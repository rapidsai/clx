from workflow.workflow import Workflow


class TestWorkflowImpl(Workflow):
    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


class TestWorkflow(object):
    def setup(self):
        # Create Test Workflow Implementation
        test_pipeline = TestWorkflowImpl()

    def test_pipeline(self):
        pass
