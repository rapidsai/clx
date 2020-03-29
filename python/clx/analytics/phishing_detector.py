import cudf
from cuml.preprocessing.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
from transformers import BertTokenizer, AdamW , BertForSequenceClassification
from tqdm import tqdm, trange
import numpy as np
import os
from clx.analytics import tokenizer

log = logging.getLogger(__name__)


class PhishingDetector:

    def __init__(self):
        self._device = None
        self._model = None
        self._optimizer = None
        self._hashpath = self._get_hash_table_path()


    def init_model(self, model_or_path="bert-base-uncased"):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = BertForSequenceClassification.from_pretrained(model_or_path, num_labels=2)
        self._model.cuda()


    def train_model(self, emails, labels, max_num_sentences=1000000, max_num_chars=100000000, max_rows_tensor=1000000, learning_rate=3e-5, max_len=128, batch_size=32, epochs=5):

        emails["label"] = labels.reset_index(drop=True)
        train_emails, validation_emails, train_labels, validation_labels = train_test_split(emails, 'label', train_size=0.8, random_state=2)

        # Tokenize training and validation
        train_inputs, train_masks, _ = tokenizer.tokenize_df(train_emails, self._hashpath, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)
        validation_inputs, validation_masks, _ = tokenizer.tokenize_df(validation_emails, self._hashpath, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

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

        test_inputs, test_masks, _ = tokenizer.tokenize_df(emails, self._hashpath, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

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

    def predict(self, emails, max_num_sentences=1000000, max_num_chars=100000000, max_rows_tensor=1000000, max_len=128, batch_size=32):

        predict_inputs, predict_masks, _ = tokenizer.tokenize_df(emails, self._hashpath, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

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

    def _get_hash_table_path(self):
        hash_table_path = "%s/resources/bert_hash_table.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        return hash_table_path

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

    def _flatten_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def _train(self, train_dataloader, validation_dataloader, model, epochs):
        train_loss_set = []# Store loss and accuracy
    
        for _ in trange(epochs, desc="Epoch"):
            model.train() # Enable training mode
            tr_loss = 0  # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self._device) for t in batch) # Add batch to GPU
                b_input_ids, b_input_mask, b_labels = batch # Unpack the inputs from dataloader
                self._optimizer.zero_grad() # Clear out the gradients
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0] # forwardpass

                train_loss_set.append(loss.item())

                loss.backward()
                self._optimizer.step() # update parameters
                tr_loss += loss.item() # get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            model.eval() # Put model in evaluation mode to evaluate loss on the validation set

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in validation_dataloader:
                batch = tuple(t.to(self._device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad(): # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0] # Forward pass, calculate logit predictions
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
