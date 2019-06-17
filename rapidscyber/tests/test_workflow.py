from rapidscyber.workflow.workflow import Workflow


class TestWorkflowImpl(Workflow):
    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


class TestWorkflow(object):
    def setup(self):
        # Create Test Workflow Implementation
        self.test_workflow = TestWorkflowImpl()

    def test_workflow(self):
        self.test_workflow.run_workflow()
