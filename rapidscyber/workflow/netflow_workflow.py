import glob
import os
import logging
import torch
import torch.nn as nn
from rapidscyber.workflow.workflow import Workflow
from rapidscyber.ml.manager.rnn_classifier_service import RNNClassifierService
from rapidscyber.ml.manager.rnn_classifier_builder import RNNClassifierBuilder

log = logging.getLogger("NetflowWorkflow")


class NetflowWorkflow(Workflow):
    def __init__(self, name, source=None, destination=None):
        super().__init__(name, source, destination)
        model_filepath = self.get_latest_modelpath("./trained_models/rnn_classifier_*")
        model = RNNClassifierBuilder.load_model(model_filepath)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        self.rnn_classifier_service = RNNClassifierService(model, optimizer, criterion)

    def workflow(self, dataframe):
        dataframe = self.add_predictions(dataframe)
        return dataframe

    def add_predictions(self, dataframe):
        domains = dataframe["domain"].to_array()
        type_ids = self.rnn_classifier_service.predict(domains)
        dataframe["prediction"] = type_ids
        return dataframe

    def get_latest_modelpath(self, models_filepath):
        log.info("models_filepath %s" % (models_filepath))
        list_of_files = glob.glob(models_filepath)
        log.info(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
