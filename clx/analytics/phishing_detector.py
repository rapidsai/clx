import cudf
from cuml.preprocessing.model_selection import train_test_split as cuml_train_test_split
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
from clx.analytics.detector import Detector
from transformers import BertTokenizer, AdamW , BertForSequenceClassification
from tqdm import tqdm, trange
import numpy as np


log = logging.getLogger(__name__)


class PhishingDetector:

    def __init__(self):
        self._device = None
        self._tokenizer = None
        self._model = None
        self._optimizer = None

    def init_model(self):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self._model.cuda()

    def train_model(self, emails_train_gdf, labels_train_gdf, learning_rate=3e-5, max_len=128, batch_size=32, epochs=5):
        tokenized_emails, labels = self._tokenize(emails_train_gdf, labels_train_gdf)
        input_ids = self._create_input_id(tokenized_emails, max_len)
        attention_masks = self._create_att_mask(input_ids)
        train_inputs, validation_inputs, train_labels, validation_labels = sklearn_train_test_split(input_ids, labels, test_size=0.20, random_state=2)
        train_masks, validation_masks, _, _ = sklearn_train_test_split(attention_masks, input_ids, test_size=0.20, random_state=2)

        train_dataloader,validation_dataloader = self._create_data_loaders(train_inputs,validation_inputs,train_labels,validation_labels,train_masks,validation_masks,batch_size)

        self._config_optimizer(learning_rate)

        self._model = self._train(train_dataloader, validation_dataloader, self._model, epochs)


    def evaluate_model(self, emails_test_gdf, labels_test_gdf, max_len=128, batch_size=32):
        tokenized_emails,labels = self._tokenize(emails_test_gdf, labels_test_gdf)
        input_ids = self._create_input_id(tokenized_emails, max_len)
        attention_masks = self._create_att_mask(input_ids)
        test_dataloader = self._testset_loader(input_ids, attention_masks,labels, batch_size)
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
        

    def predict(self, emails, max_len=128, batch_size=32):
        emails = emails.to_array()
        emails = ["[CLS] " + str(email) + " [SEP]" for email in emails]#add cls and sep so they are recognized by the tokenizer
        tokenized_emails = [self._tokenizer.tokenize(email) for email in emails]
        input_ids = self._create_input_id(tokenized_emails,max_len)
        attention_masks = self._create_att_mask(input_ids)

        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        input_data = TensorDataset(inputs, masks)
        test_sampler = SequentialSampler(input_data)
        input_dataloader = DataLoader(input_data, sampler=test_sampler, batch_size=batch_size)

        self._model.eval()

        results = []
        for batch in input_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask = batch

            with torch.no_grad():

                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()

            results.append(logits)

        preds = [item for sublist in results for item in sublist]
        preds = np.argmax(preds, axis=1).flatten()

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
                # print(type(train_loss_set))

                train_loss_set.append(loss.item())
                # print("loss.item",loss.item())

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
