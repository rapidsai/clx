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


class Resources:

    _instance = None

    @staticmethod
    def get_instance():
        if Resources._instance is None:
            Resources()
        return Resources._instance

    def __init__(self):
        if Resources._instance is not None:
            raise Exception("This is a singleton class")
        else:
            Resources._instance = self
            Resources._instance._iana_lookup_df = self._load_iana_lookup_df()

    @property
    def iana_lookup_df(self):
        return self._iana_lookup_df

    def _load_iana_lookup_df(self):
        iana_path = "%s/resources/iana_port_lookup.csv" % os.path.dirname(
            os.path.realpath(__file__)
        )
        colNames = ["port", "service"]
        colTypes = ["int64", "str"]
        iana_lookup_df = cudf.read_csv(
            iana_path,
            delimiter=',',
            names=colNames,
            dtype=colTypes,
            skiprows=1
        )
        iana_lookup_df = iana_lookup_df.dropna()
        iana_lookup_df = iana_lookup_df.groupby(["port"]).min().reset_index()

        return iana_lookup_df


def major_ports(addr_col, port_col, min_conns=1, eph_min=10000):
    """Find major ports for each address. This is done by computing the mean number of connections across all
    ports for each address and then filters out all ports that don't cross this threshold. Also adds column
    for IANA service name correspondingto each port.

    :param addr_col: Column of addresses as strings
    :type addr_col: cudf.Series
    :param port_col: Column of corresponding port numbers as ints
    :type port_col: cudf.Series
    :param min_conns: Filter out ip:port rows that don't have at least this number of connections (default: 1)
    :type min_conns: int
    :param eph_min: Ports greater than or equal to this will be labeled as an ephemeral service (default: 10000)
    :type eph_min: int
    :return: DataFrame with columns for address, port, IANA service corresponding to port, and number of connections
    :rtype: cudf.DataFrame

    Examples
    --------
    >>> import clx.heuristics.ports as ports
    >>> import cudf
    >>> input_addr_col = cudf.Series(["10.0.75.1","10.0.75.1","10.0.75.1","10.0.75.255","10.110.104.107", "10.110.104.107"])
    >>> input_port_col = cudf.Series([137,137,7680,137,7680, 7680])
    >>> ports.major_ports(input_addr_col, input_port_col, min_conns=2, eph_min=7000)
                addr  port     service  conns
    0      10.0.75.1   137  netbios-ns      2
    1 10.110.104.107  7680   ephemeral      2
    """

    # Count the number of connections across each src ip-port pair
    gdf = cudf.DataFrame({"addr": addr_col, "port": port_col})
    gdf["conns"] = 1.0
    gdf = gdf.groupby(["addr", "port"], as_index=False).count()

    # Calculate average number of connections across all ports for each ip
    cnt_avg_gdf = gdf[["addr", "conns"]]
    cnt_avg_gdf = cnt_avg_gdf.groupby(["addr"], as_index=False).mean()
    cnt_avg_gdf = cnt_avg_gdf.rename(columns={"conns": "avg"})

    # Merge averages to dataframe
    gdf = gdf.merge(cnt_avg_gdf, on=['addr'], how='left')

    # Filter out all ip-port pairs below average
    gdf = gdf[gdf.conns >= gdf.avg]

    if min_conns > 1:
        gdf = gdf[gdf.conns >= min_conns]

    gdf = gdf.drop(['avg'], axis=1)

    resources = Resources.get_instance()
    iana_lookup_df = resources.iana_lookup_df

    # Add IANA service names to node lists
    gdf = gdf.merge(iana_lookup_df, on=['port'], how='left')

    gdf.loc[gdf["port"] >= eph_min, "service"] = "ephemeral"

    gdf = gdf.groupby(["addr", "port", "service"], dropna=False, as_index=False, sort=True).sum()

    return gdf
