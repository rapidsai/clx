import glob
import os
import logging
from rapidscyber.workflow.workflow import Workflow
from rapidscyber.ml.provider.dga_detector import DGADetector

log = logging.getLogger("NetflowWorkflow")


class NetflowWorkflow(Workflow):
    def __init__(self, name, source=None, destination=None):
        super().__init__(name, source, destination)
        model_filepath = self.get_latest_modelpath()
        self.dd = DGADetector()
        self.dd.load_model(model_filepath)

    def workflow(self, dataframe):
        dataframe = self.add_predictions(dataframe)
        return dataframe

    def add_predictions(self, dataframe):
        domains = dataframe["domain"].to_array()
        type_ids = self.dd.predict(domains)
        dataframe["prediction"] = type_ids
        return dataframe

    def get_latest_modelpath(self):
        models_filepath = "./trained_models/rnn_classifier_*"
        log.info("models_filepath %s" % (models_filepath))
        list_of_files = glob.glob(models_filepath)
        log.info(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
