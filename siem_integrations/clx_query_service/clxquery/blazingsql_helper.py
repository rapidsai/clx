# Copyright (c) 2020, NVIDIA CORPORATION.
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

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from blazingsql import BlazingContext
import logging

log = logging.getLogger(__name__)
"""
This class provides functionality to run blazingSQL queires and drop tables.
"""


class BlazingSQLHelper:
    def __init__(self):
        cluster = LocalCUDACluster()
        client = Client(cluster)
        self._bc = BlazingContext(dask_client = client, network_interface = 'lo')

    """This function runs blazingSQL query. 
    
    :param config: Query related tables configuration.
    :type config: dict
    :return: Query results.
    :rtype: cudf.DataFrame
    """

    def run_query(self, config):
        for table in config["tables"]:
            table_name = table["table_name"]
            file_path = table["input_path"]
            kwargs = table.copy()
            del kwargs["table_name"]
            del kwargs["input_path"]
            self._bc.create_table(table_name, file_path, **kwargs)
        sql = config["sql"]
        log.debug("Executing query: %s" % (sql))
        result = self._bc.sql(sql)
        result = result.compute()
        return result

    """This function drops blazingSQL tables.
    :param table_names: List of table names to drop.
    :type table_names: List
    """

    def drop_table(self, table_names):
        for table_name in table_names:
            log.debug("Drop table: %s" % (table_name))
            self._bc.drop_table(table_name)