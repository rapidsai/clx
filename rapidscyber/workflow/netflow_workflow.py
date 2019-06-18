class NetflowWorkflow(Workflow):
    def workflow(self, dataframe):
        """TODO: Implement netflow dataframe enrichment"""
        dataframe["netflow_enriched"] = "netflow_enriched"
        return dataframe
