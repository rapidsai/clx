# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import os
import pytest
import yaml
from clx.workflow.workflow import Workflow
from mockito import spy, verify, when
from cudf import DataFrame

input_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)

empty_df = DataFrame()


class TestWorkflowImpl(Workflow):
    def __init__(self, name, source=None, destination=None, custom_workflow_param=None):
        self.custom_workflow_param = custom_workflow_param
        Workflow.__init__(self, name, source, destination)

    def workflow(self, dataframe):
        dataframe["enriched"] = "enriched"
        return dataframe


dirname = os.path.split(os.path.abspath(__file__))[0]


@pytest.fixture
def set_workflow_config():
    """Sets the workflow config dictionary used for the unit tests"""
    source = {
        "type": "fs",
        "input_format": "csv",
        "input_path": "/path/to/input",
        "names": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "usecols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
    }
    destination = {
        "type": "fs",
        "output_format": "csv",
        "output_path": "/path/to/output",
        "index": False
    }
    workflow_config = {
        "source": source,
        "destination": destination,
        "custom_workflow_param": "param_value",
    }
    return workflow_config, source, destination


@pytest.fixture
def mock_env_home(monkeypatch):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    monkeypatch.setenv("HOME", dirname)


@pytest.mark.parametrize("input_df", [input_df])
def test_workflow_parameters(
    tmpdir, mock_env_home, set_workflow_config, input_df
):
    """Tests the initialization and running of a workflow with passed in parameters"""
    source = set_workflow_config[1]
    destination = set_workflow_config[2]

    test_dir = tmpdir.mkdir("tmp_test_workflow")
    input_path = str(test_dir.join("person.csv"))
    input_df.to_csv(input_path, index=False)
    output_path = str(test_dir.join("output_parameters.csv"))
    source["input_path"] = input_path
    destination["output_path"] = output_path

    # Create new workflow with source and destination configurations
    test_workflow = TestWorkflowImpl(
        source=source,
        destination=destination,
        name="test-workflow",
        custom_workflow_param="test_param",
    )

    test_workflow.run_workflow()
    expected_df = input_df
    expected_df["enriched"] = "enriched"
    result_df = cudf.read_csv(output_path)

    assert result_df.equals(expected_df)
    assert test_workflow.custom_workflow_param == "test_param"


def test_workflow_config(tmpdir, mock_env_home, set_workflow_config):
    """Tests the initialization and running of a workflow with a configuration yaml file"""
    test_dir = tmpdir.mkdir("tmp_test_workflow")
    input_path = str(test_dir.join("person.csv"))
    input_df.to_csv(input_path, index=False)
    output_path = str(test_dir.join("output_config.csv"))

    # Write workflow.yaml file
    workflow_name = "test-workflow-config"
    workflow_config = set_workflow_config[0]
    workflow_config["destination"]["output_path"] = output_path
    workflow_config["destination"]["index"] = False
    workflow_config["source"]["input_path"] = input_path
    workflow_config["custom_workflow_param"] = "param_value"
    write_config_file(workflow_config, workflow_name)

    # Run workflow
    test_workflow = TestWorkflowImpl(workflow_name)
    test_workflow.run_workflow()

    expected_df = input_df
    expected_df["enriched"] = "enriched"
    result_df = cudf.read_csv(output_path)
    assert result_df.equals(expected_df)

    # Check that custom workflow parameter was set from config file
    assert test_workflow.custom_workflow_param == "param_value"


def test_workflow_config_error(mock_env_home, set_workflow_config):
    """Tests the error handling on incomplete workflow.yaml configuration file"""
    workflow_name = "test-workflow-error"
    test_config = {}
    test_config["source"] = set_workflow_config[1]
    write_config_file(test_config, workflow_name)
    with pytest.raises(Exception):
        TestWorkflowImpl(workflow_name)

    test_config = {}
    test_config["destination"] = set_workflow_config[2]
    write_config_file(test_config, workflow_name)
    with pytest.raises(Exception):
        TestWorkflowImpl(workflow_name)


def test_workflow_no_data(tmpdir, mock_env_home, set_workflow_config):
    """ Test confirms that workflow is not run and output not written if no data is returned from the workflow io_reader
    """
    # Create source and destination configurations
    source = set_workflow_config[1]
    destination = set_workflow_config[2]

    test_dir = tmpdir.mkdir("tmp_test_workflow")
    input_path = str(test_dir.join("input_empty.csv"))
    empty_df.to_csv(input_path)
    output_path = str(test_dir.join("output_empty.csv"))
    source["input_path"] = input_path
    destination["output_path"] = output_path

    # Create new workflow with source and destination configurations
    test_workflow = spy(TestWorkflowImpl(
        source=source, destination=destination, name="test-workflow-no-data", custom_workflow_param="test_param"
    ))
    test_workflow.run_workflow()

    # Verify workflow not run
    verify(test_workflow, times=0).workflow(...)

    # Verify that no output file created.
    assert not os.path.exists(output_path)


def test_workflow_no_enriched_data(tmpdir, mock_env_home, set_workflow_config):
    """ Test confirms that if workflow produces no enriched data that no output file is created
    """
    # Create source and destination configurations
    source = set_workflow_config[1]
    destination = set_workflow_config[2]

    test_dir = tmpdir.mkdir("tmp_test_workflow")
    input_path = str(test_dir.join("person.csv"))
    input_df.to_csv(input_path, index=False)
    output_path = str(test_dir.join("output_empty.csv"))
    source["input_path"] = input_path
    destination["output_path"] = output_path

    # Create new workflow with source and destination configurations
    test_workflow = spy(TestWorkflowImpl(
        source=source, destination=destination, name="test-workflow-no-data", custom_workflow_param="test_param"
    ))
    io_writer = spy(test_workflow._io_writer)

    # Return empty dataframe when workflow runs
    when(test_workflow).workflow(...).thenReturn(DataFrame())

    # Verify io_writer does not write data
    verify(io_writer, times=0).write_data(...)

    # Verify that no output file created.
    assert not os.path.exists(output_path)


def test_benchmark_decorator(
    tmpdir, mock_env_home, set_workflow_config
):
    # Dummy function
    def func(self):
        return DataFrame()

    benchmarked_func = Workflow.benchmark(func)

    source = set_workflow_config[1]
    destination = set_workflow_config[2]

    test_dir = tmpdir.mkdir("tmp_test_workflow")
    input_path = str(test_dir.join("person.csv"))
    input_df.to_csv(input_path, index=False)
    output_path = str(test_dir.join("output_benchmark.csv"))
    source["input_path"] = input_path
    destination["output_path"] = output_path
    # Create new workflow with source and destination configurations
    tb = spy(
        TestWorkflowImpl(source=source, destination=destination, name="test-workflow")
    )
    benchmarked_func(tb.run_workflow)

    # Verify that run_workflow was not called, instead expect that benchmark wrapper function will be called
    verify(tb, times=0).run_workflow(...)


def write_config_file(workflow_config, workflow_name):
    """Helper function to write workflow.yaml configuration file"""
    workflow_dir = "{0}/.config/clx/{1}".format(dirname, workflow_name)
    if not os.path.exists(workflow_dir):
        os.makedirs(workflow_dir)
    with open(workflow_dir + "/workflow.yaml", "w") as f:
        yaml.dump(workflow_config, f)
