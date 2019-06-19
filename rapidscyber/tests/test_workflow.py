import csv
import os
import pytest
import yaml
from rapidscyber.workflow.workflow import Workflow


class TestWorkflowImpl(Workflow):
    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


dirname = os.path.split(os.path.abspath(__file__))[0]
input_path = dirname + "/input/person.csv"
output_path_param_test = dirname + "/output/output_parameters.csv"
output_path_config_test = dirname + "/output/output_config.csv"


@pytest.fixture
def set_workflow_config():
    """Sets the workflow config dictionary used for the unit tests"""
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
    destination = {
        "type": "fs",
        "output_format": "csv",
        "output_path": "/path/to/output",
    }
    workflow_config = {"source": source, "destination": destination}


@pytest.fixture
def mock_env_home(monkeypatch):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    monkeypatch.setenv("HOME", dirname)


@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path_param_test])
def test_workflow_parameters(
    mock_env_home, set_workflow_config, input_path, output_path
):
    """Tests the initialization and running of a workflow with passed in parameters"""
    # Create source and destination configurations
    source["input_path"] = input_path
    destination["output_path"] = output_path
    # Create new workflow with source and destination configurations
    test_workflow = TestWorkflowImpl(
        source=source, destination=destination, name="test-workflow"
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
@pytest.mark.parametrize("output_path", [output_path_config_test])
def test_workflow_config(mock_env_home, set_workflow_config, input_path, output_path):
    """Tests the initialization and running of a workflow with a configuration yaml file"""
    # Write workflow.yaml file
    workflow_config["destination"]["output_path"] = output_path
    workflow_config["input"]["input_path"] = input_path
    write_config_file(workflow_config, "test-workflow-config")

    if os.path.exists(output_path):
        os.remove(output_path)

    # Run workflow
    test_workflow = TestWorkflowImpl(name)
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


def test_workflow_config_error(mock_env_home, set_workflow_config):
    """Tests the error handling on incomplete workflow.yaml configuration file"""
    # Write workflow.yaml file
    test_config = {}
    test_config["source"] = source
    write_config_file(test_config, "test-workflow-error")
    with pytest.raises(IOError):
        test_workflow = TestWorkflowImpl(name)

    test_config = {}
    test_config["destination"] = destination
    write_config_file(test_config, "test-workflow-error")
    with pytest.raises(IOError):
        test_workflow = TestWorkflowImpl(name)


def write_config_file(workflow_config, workflow_name):
    """Helper function to write workflow.yaml configuration file"""
    workflow_dir = "{0}/.config/rapidscyber/{1}".format(dirname, workflow_name)
    if not os.path.exists(workflow_dir):
        os.makedirs(workflow_dir)
    with open(workflow_dir + "/workflow.yaml", "w") as f:
        yaml.dump(workflow_config, f)
