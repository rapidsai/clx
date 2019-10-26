import csv
import os
import pytest
import yaml
from clx.workflow.workflow import Workflow
from mockito import spy,verify
from cudf import DataFrame

class TestWorkflowImpl(Workflow):
    def __init__(self,name, source=None, destination=None, custom_workflow_param=None):
        self.custom_workflow_param = custom_workflow_param
        Workflow.__init__(self, name, source, destination)

    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe

dirname = os.path.split(os.path.abspath(__file__))[0]
input_path = dirname + "/input/person.csv"
output_path_param_test = dirname + "/output/output_parameters.csv"
output_path_benchmark_test = dirname + "/output/output_benchmark.csv"
output_path_config_test = dirname + "/output/output_config.csv"

@pytest.fixture
def set_workflow_config():
    """Sets the workflow config dictionary used for the unit tests"""
    source = {
        "type": "fs",
        "input_format": "csv",
        "filepath": "/path/to/input",
        "names": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "usecols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
    }
    destination = {
        "type": "fs",
        "output_format": "csv",
        "filepath": "/path/to/output",
        "index": False
    }
    workflow_config = {"source": source, "destination": destination, "custom_workflow_param": "param_value"}
    return workflow_config, source, destination


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
    source = set_workflow_config[1]
    destination = set_workflow_config[2]
    source["filepath"] = input_path
    destination["filepath"] = output_path
    # Create new workflow with source and destination configurations
    test_workflow = TestWorkflowImpl(
        source=source, destination=destination, name="test-workflow", custom_workflow_param="test_param"
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

    assert test_workflow.custom_workflow_param == "test_param"


@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path_config_test])
def test_workflow_config(mock_env_home, set_workflow_config, input_path, output_path):
    """Tests the initialization and running of a workflow with a configuration yaml file"""
    # Write workflow.yaml file
    workflow_name = "test-workflow-config"
    workflow_config = set_workflow_config[0]
    workflow_config["destination"]["filepath"] = output_path
    workflow_config["destination"]["index"] = False
    workflow_config["source"]["filepath"] = input_path
    workflow_config["custom_workflow_param"] = "param_value"
    write_config_file(workflow_config, workflow_name)

    if os.path.exists(output_path):
        os.remove(output_path)

    # Run workflow
    test_workflow = TestWorkflowImpl(workflow_name)
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

    # Check that custom workflow parameter was set from config file
    assert test_workflow.custom_workflow_param == "param_value"


def test_workflow_config_error(mock_env_home, set_workflow_config):
    """Tests the error handling on incomplete workflow.yaml configuration file"""
    workflow_name = "test-workflow-error"
    test_config = {}
    test_config["source"] = set_workflow_config[1]
    write_config_file(test_config, workflow_name)
    with pytest.raises(Exception):
        test_workflow = TestWorkflowImpl(workflow_name)

    test_config = {}
    test_config["destination"] = set_workflow_config[2]
    write_config_file(test_config, workflow_name)
    with pytest.raises(Exception):
        test_workflow = TestWorkflowImpl(workflow_name)

@pytest.mark.parametrize("input_path", [input_path])
@pytest.mark.parametrize("output_path", [output_path_benchmark_test])
def test_benchmark_decorator(mock_env_home, set_workflow_config, input_path, output_path):
    #Dummy function
    def func(self):
       return DataFrame()
    benchmarked_func = Workflow.benchmark(func)

    source = set_workflow_config[1]
    destination = set_workflow_config[2]
    source["filepath"] = input_path
    destination["filepath"] = output_path
    # Create new workflow with source and destination configurations
    tb = spy(TestWorkflowImpl(
        source=source, destination=destination, name="test-workflow"
    ))
    benchmarked_func(tb.run_workflow)
    
    #Verify that run_workflow was not called, instead expect that benchmark wrapper function will be called
    verify(tb, times=0).run_workflow(...)

def write_config_file(workflow_config, workflow_name):
    """Helper function to write workflow.yaml configuration file"""
    workflow_dir = "{0}/.config/clx/{1}".format(dirname, workflow_name)
    if not os.path.exists(workflow_dir):
        os.makedirs(workflow_dir)
    with open(workflow_dir + "/workflow.yaml", "w") as f:
        yaml.dump(workflow_config, f)
