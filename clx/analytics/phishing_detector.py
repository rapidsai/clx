import cudf
from cuml.preprocessing.model_selection import train_test_split
# from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
from transformers import BertTokenizer, AdamW , BertForSequenceClassification
from tqdm import tqdm, trange
import numpy as np
import os
import tokenizer

log = logging.getLogger(__name__)


class PhishingDetector:

    def __init__(self):
        self._device = None
        self._tokenizer = None
        self._model = None
        self._optimizer = None

    def init_model(self, model_or_path="bert-base-uncased"):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(model_or_path, num_labels=2)
        self._model.cuda()

    def train_model(self, emails, labels, max_num_sentences=1000000, max_num_chars=100000000, max_rows_tensor=1000000, learning_rate=3e-5, max_len=128, batch_size=32, epochs=5):

        emails["label"] = labels
        train_emails, validation_emails, train_labels, validation_labels = train_test_split(emails, 'label', train_size=0.8, random_state=2)

        # Save train_emails and validation_emails to files for tokenizer. This will change to cudf inputs.
        train_emails_pd = train_emails.to_pandas()
        train_emails_pd.iloc[:, 0].to_csv(path="train_emails.txt", header=False, index=False)

        validation_emails_pd = validation_emails.to_pandas()
        validation_emails_pd.iloc[:, 0].to_csv(path="validation_emails.txt", header=False, index=False)

        # Now get tokenize training and validation
        train_inputs, train_masks, _ = tokenizer.tokenizer("train_emails.txt", "../../hash_table.txt", max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)
        validation_inputs, validation_masks, _ = tokenizer.tokenizer("validation_emails.txt", "../../hash_table.txt", max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

        # training fails because if it doesn't have longs here
        train_inputs = train_inputs.type(torch.LongTensor)
        validation_inputs = validation_inputs.type(torch.LongTensor)

        # convert labels to tensors
        train_labels = torch.tensor(train_labels.to_array())
        validation_labels = torch.tensor(validation_labels.to_array())

        # create dataloaders
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        self._config_optimizer(learning_rate)

        self._model = self._train(train_dataloader, validation_dataloader, self._model, epochs)


    def evaluate_model(self, emails, labels, max_num_sentences=1000000, max_num_chars=100000000, max_rows_tensor=1000000, max_len=128, batch_size=32):

        # Save test emails to files for tokenizer. This will change to cudf input.
        test_emails_pd = emails.to_pandas()
        test_emails_pd.iloc[:, 0].to_csv(path="test_emails.txt", header=False, index=False)

        test_inputs, test_masks, _ = tokenizer.tokenizer("test_emails.txt", "../../hash_table.txt", max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

        test_inputs = test_inputs.type(torch.LongTensor)
        test_labels = torch.tensor(labels.to_array())
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        tests, true_labels = self._evaluate_testset(test_dataloader)

        flat_tests = [item for sublist in tests for item in sublist]
        flat_tests = np.argmax(flat_tests, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = accuracy_score(flat_true_labels, flat_tests)

        return accuracy

    def save_model(self, save_to_path="."):
        self._model.save_pretrained(save_to_path)

    def load_model(self, model_path):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(model_path)
        self._model.cuda()
        

    def predict(self, emails, max_num_sentences=1000000, max_num_chars=100000000, max_rows_tensor=1000000, max_len=128, batch_size=32):

        predict_emails_pd = emails.to_pandas()
        predict_emails_pd.to_csv(path="predict_emails.txt", header=False, index=False)

        predict_inputs, predict_masks, _ = tokenizer.tokenizer("predict_emails.txt", "../../hash_table.txt", max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

        predict_inputs = predict_inputs.type(torch.LongTensor)
        predict_data = TensorDataset(predict_inputs, predict_masks)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=batch_size)

        self._model.eval()

        results = []
        for batch in predict_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask = batch

            with torch.no_grad():

                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()

            results.append(logits)

        preds = [item for sublist in results for item in sublist]
        preds = np.argmax(preds, axis=1).flatten()
        preds = cudf.Series(preds.tolist())

        return preds


    def _tokenize(self, X_, y_):
        emails = X_.to_records()
        emails = ["[CLS] " + str(email) + " [SEP]" for email in emails]#add cls and sep so they are recognized by the tokenizer
        labels = y_.to_array()
        tokenized_emails = [self._tokenizer.tokenize(email) for email in emails]
        return tokenized_emails, labels

    def _create_att_mask(self, input_ids):
        attention_masks = []
        for a in input_ids:
            a_mask = [float(i>0) for i in a]
            attention_masks.append(a_mask)
        return attention_masks

    def _testset_loader(self, input_ids, attention_masks, labels, batch_size):
        test_inputs = torch.tensor(input_ids)
        test_masks = torch.tensor(attention_masks)
        test_labels = torch.tensor(labels)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        return test_dataloader

    def _create_data_loaders(self, train_inputs,validation_inputs,train_labels,validation_labels,train_masks,validation_masks,batch_size):
        ## Convert all of our data into torch tensors, 
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
        ## Create an iterator of our data with torch DataLoader. more memory effective
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        return train_dataloader,validation_dataloader

    def _config_optimizer(self, learning_rate):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, learning_rate)

    def _create_input_id(self, tokenized_emails, max_len):
        input_ids = [self._tokenizer.convert_tokens_to_ids(email) for email in tokenized_emails]
        input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
        return input_ids

    def _flatten_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def _train(self, train_dataloader, validation_dataloader, model, epochs):
        train_loss_set = []# Store loss and accuracy
    
        for _ in trange(epochs, desc="Epoch"):
            model.train() # enable training mode
            tr_loss = 0  # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self._device) for t in batch)# Add batch to GPU
                b_input_ids, b_input_mask, b_labels = batch# Unpack the inputs from dataloader
                self._optimizer.zero_grad()# Clear out the gradients
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]#forwardpass

                train_loss_set.append(loss.item())

                loss.backward()
                self._optimizer.step()#update parameters
                tr_loss += loss.item()#get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            model.eval()# Put model in evaluation mode to evaluate loss on the validation set

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in validation_dataloader:
                batch = tuple(t.to(self._device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():# Telling the model not to compute or store gradients, saving memory and speeding up validation
                    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]# Forward pass, calculate logit predictions
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                temp_eval_accuracy = self._flatten_accuracy(logits, label_ids)

                eval_accuracy += temp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

        return model
    
    def _evaluate_testset(self, test_dataloader):
        
        self._model.eval()

        tests , true_labels = [], []

        for batch in test_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tests.append(logits)
            true_labels.append(label_ids)

        return tests,true_labels
