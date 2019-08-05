import pytest
import requests
from mockito import when, mock
from rapidscyber.osi.farsight import FarsightLookupClient

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
def test_query_rdata_ip(server, apikey, ip):
    client = FarsightLookupClient(server, apikey, limit=1)
    when(client).get(...).thenReturn(ip_response)
    result = client.query_rdata_ip(ip)
    assert len(result) == 1


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
def test_query_rrset(server, apikey):
    client = FarsightLookupClient(server, apikey)
    when(client).get(...).thenReturn(rrset_response)
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
def test_query_rdata_name(server, apikey):
    client = FarsightLookupClient(server, apikey)
    when(client).get(...).thenReturn(rdata_name_response)
    result = client.query_rdata_name("www.farsightsecurity.com")
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert result["rdata"] == "www.farsightsecurity.com."
