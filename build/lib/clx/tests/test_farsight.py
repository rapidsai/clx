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

import pytest
import requests
from mockito import when, mock
from clx.osi.farsight import FarsightLookupClient

ip = "100.0.0.1"
server = "https://localhost"
apikey = "dummy-api-key"

ip_response = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{"count":69435,"time_first":1428433465,"time_last":1538014110,"rrname":"io.","rrtype":"A","rdata":"100.0.0.1"}',
    },
    spec=requests.Response,
)

rrset_response = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{"count":81556,"time_first":1374184718,"time_last":1564909243,"rrname":"www.dnsdb.info.","rrtype":"CNAME","bailiwick":"dnsdb.info.","rdata":["dnsdb.info."]}',
    },
    spec=requests.Response,
)

rdata_name_response = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{"count":497,"time_first":1386638408,"time_last":1561176503,"rrname":"81.64-26.140.160.66.in-addr.arpa.","rrtype":"PTR","rdata":"www.farsightsecurity.com."}',
    },
    spec=requests.Response,
)


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("ip", [ip])
@pytest.mark.parametrize("ip_response", [ip_response])
def test_query_rdata_ip(server, apikey, ip, ip_response):
    client = FarsightLookupClient(server, apikey, limit=1)
    when(requests).get(...).thenReturn(ip_response)
    result = client.query_rdata_ip(ip)
    assert len(result) == 1


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("ip", [ip])
@pytest.mark.parametrize("ip_response", [ip_response])
def test_query_rdata_ip2(server, apikey, ip, ip_response):
    client = FarsightLookupClient(server, apikey, limit=1)
    when(requests).get(...).thenReturn(ip_response)
    result = client.query_rdata_ip(ip, before=1428433465, after=1538014110)
    assert len(result) == 1


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("rrset_response", [rrset_response])
def test_query_rrset(server, apikey, rrset_response):
    client = FarsightLookupClient(server, apikey)
    when(requests).get(...).thenReturn(rrset_response)
    result = client.query_rrset("www.dnsdb.info")
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert "bailiwick" in result
    assert "rdata" in result
    assert result["bailiwick"] == "dnsdb.info."


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("rrset_response", [rrset_response])
def test_query_rrset2(server, apikey, rrset_response):
    client = FarsightLookupClient(server, apikey)
    when(requests).get(...).thenReturn(rrset_response)
    result = client.query_rrset(
        "www.dnsdb.info",
        rrtype="CNAME",
        bailiwick="dnsdb.info.",
        before=1374184718,
        after=1564909243,
    )
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert "bailiwick" in result
    assert "rdata" in result
    assert result["bailiwick"] == "dnsdb.info."


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("rdata_name_response", [rdata_name_response])
def test_query_rdata_name(server, apikey, rdata_name_response):
    client = FarsightLookupClient(server, apikey)
    when(requests).get(...).thenReturn(rdata_name_response)
    result = client.query_rdata_name("www.farsightsecurity.com")
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert result["rdata"] == "www.farsightsecurity.com."


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("rdata_name_response", [rdata_name_response])
def test_query_rdata_name2(server, apikey, rdata_name_response):
    client = FarsightLookupClient(server, apikey)
    when(requests).get(...).thenReturn(rdata_name_response)
    result = client.query_rdata_name(
        "www.farsightsecurity.com", rrtype="PTR", before=1386638408, after=1561176503
    )
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert result["rdata"] == "www.farsightsecurity.com."
