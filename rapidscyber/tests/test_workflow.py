import csv
import os
import pytest
import yaml
from rapidscyber.workflow.workflow import Workflow


class TestWorkflowImpl(Workflow):
    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


dirname, filename = os.path.split(os.path.abspath(__file__))
input_path = dirname + "/input/person.csv"
output_path_param = dirname + "/output/output_parameters.csv"
output_path_config = dirname + "/output/output_config.csv"
source = {
    "type": "fs",
    "input_format": "csv",
    "input_path": "/path/to/input",
    "schema": ["firstname", "lastname", "gender"],
    "delimiter": ",",
    "required_cols": ["firstname", "lastname", "gender"],
    "dtype": ["str", "str", "str"],
    "header": 0,
}
destination = {"type": "fs", "output_format": "csv", "output_path": "/path/to/output"}


@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path_param])
def test_workflow_parameters(input_path, output_path):
    """Tests the initialization and running of a workflow with passed in parameters"""
    # Create source and destination configurations
    source_config = source
    source_config["input_path"] = input_path
    dest_config = destination
    dest_config["output_path"] = output_path

    # Create new workflow with source and destination configurations
    test_workflow = TestWorkflowImpl(
        source=source_config, destination=dest_config, name="my-new-workflow-name"
    )

    # Run workflow and check output data
    if os.path.exists(output_path):
        os.remove(output_path)
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


@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path_config])
def test_workflow_config(input_path, output_path):
    """Tests the initialization and running of a workflow with a configuration yaml file"""
    # Write workflow.yaml file
    workflow_config = {}
    workflow_config["source"] = source
    workflow_config["source"]["input_path"] = input_path
    workflow_config["destination"] = destination
    workflow_config["destination"]["output_path"] = output_path
    workflow_config["name"] = "my-workflow"
    workflow_yaml_file = dirname + "/workflow.yaml"
    with open(workflow_yaml_file, "w") as f:
        yaml.dump(workflow_config, f)
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run workflow
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
