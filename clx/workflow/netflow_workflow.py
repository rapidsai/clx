import logging
from clx.workflow.workflow import Workflow

log = logging.getLogger(__name__)

class NetflowWorkflow(Workflow):
    def workflow(self, dataframe):
        """TODO: Implement netflow dataframe enrichment"""
        log.debug("Processing netflow workflow data...")
        dataframe["netflow_enriched"] = "netflow_enriched"
        return dataframe
