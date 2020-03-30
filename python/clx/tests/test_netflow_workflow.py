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
from clx.workflow.netflow_workflow import NetflowWorkflow


def test_netflow_workflow():
    """Tests the netflow dataframe enrichment"""
    netflow_workflow = NetflowWorkflow("netflow-workflow")
    input_df = cudf.DataFrame(
        {
            "ts time": ["12345678900.12345"],
            "uid string": ["123ABC"],
            "id.orig_h": ["123.456.789"],
            "id.orig_p": ["1000"],
            "id.resp_h": ["987.654.321"],
            "id.resp_p": ["80"],
            "proto": ["tcp"],
            "service": ["-"],
            "duration": ["2.015"],
            "orig_bytes": ["0"],
            "resp_bytes": ["0"],
            "conn_state": ["SH"],
            "local_orig": ["-"],
            "local_resp": ["-"],
            "missed_bytes": ["0"],
            "history": ["F"],
            "orig_pkts count": ["2"],
            "orig_ip_bytes": ["80"],
            "resp_pkts": ["0"],
            "resp_ip_bytes": ["0"],
            "tunnel_parents": ["-"],
        }
    )
    actual_df = netflow_workflow.workflow(input_df)
    expected_df = cudf.DataFrame(
        {
            "ts time": ["12345678900.12345"],
            "uid string": ["123ABC"],
            "id.orig_h": ["123.456.789"],
            "id.orig_p": ["1000"],
            "id.resp_h": ["987.654.321"],
            "id.resp_p": ["80"],
            "proto": ["tcp"],
            "service": ["-"],
            "duration": ["2.015"],
            "orig_bytes": ["0"],
            "resp_bytes": ["0"],
            "conn_state": ["SH"],
            "local_orig": ["-"],
            "local_resp": ["-"],
            "missed_bytes": ["0"],
            "history": ["F"],
            "orig_pkts count": ["2"],
            "orig_ip_bytes": ["80"],
            "resp_pkts": ["0"],
            "resp_ip_bytes": ["0"],
            "tunnel_parents": ["-"],
            "netflow_enriched": ["netflow_enriched"],
        }
    )

    assert actual_df.equals(expected_df)
