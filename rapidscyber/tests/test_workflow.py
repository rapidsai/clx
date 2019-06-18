import csv
import os
import pytest
from rapidscyber.workflow.workflow import Workflow


class TestWorkflowImpl(Workflow):
    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


cwd = os.getcwd()
input_path = cwd + "/tests/input/person.csv"
output_path = cwd + "/tests/output/output_parameters.csv"


@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path])
def test_workflow_parameters(input_path, output_path):
    source = {
        "type": "fs",
        "input_format": "csv",
        "input_path": input_path,
        "schema": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "required_cols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
    }
    destination = {"type": "fs", "output_format": "csv", "output_path": output_path}
    test_workflow = TestWorkflowImpl(
        source=source, destination=destination, name="my-new-workflow-name"
    )
    test_workflow.run_workflow()
    with open(output_path) as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    assert data[0] == ["firstname", "lastname", "gender", "enriched"]
    assert data[1] == ["Emma", "Olivia", "F", "enriched"]
    assert data[2] == ["Ava", "Isabella", "F", "enriched"]
    assert data[3] == ["Sophia", "Charlotte", "F", "enriched"]
    os.remove(output_path)


def test_workflow_config():
    output_path = "tests/output/output_config.csv"
    if os.path.exists(output_path):
        os.remove(output_path)
    test_workflow = TestWorkflowImpl()
    test_workflow.run_workflow()
    with open(output_path) as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    assert data[0] == ["firstname", "lastname", "gender", "enriched"]
    assert data[1] == ["Emma", "Olivia", "F", "enriched"]
    assert data[2] == ["Ava", "Isabella", "F", "enriched"]
    assert data[3] == ["Sophia", "Charlotte", "F", "enriched"]
    os.remove(output_path)
