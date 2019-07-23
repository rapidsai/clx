import pytest
from rapidscyber.osi.farsight import FarsightLookupClient

ip = "104.244.13.104"
server = "https://api.dnsdb.info"
# note: generate your own API key by subscribing to Farsight service.
# https://www.farsightsecurity.com
apikey = "< add api key here >"


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
@pytest.mark.parametrize("ip", [ip])
def test_query_rdata_ip(server, apikey, ip):
    client = FarsightLookupClient(server, apikey, limit=1)
    result = client.query_rdata_ip(ip)
    assert len(result) == 1
    assert result[0]["rdata"] == ip


@pytest.mark.parametrize("server", [server])
@pytest.mark.parametrize("apikey", [apikey])
def test_query_rrset(server, apikey):
    client = FarsightLookupClient(server, apikey)
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
    result = client.query_rdata_name("www.farsightsecurity.com")
    result = result[0]
    assert "count" in result
    assert "time_first" in result
    assert "time_last" in result
    assert "rrname" in result
    assert "rrtype" in result
    assert result["rdata"] == "www.farsightsecurity.com."
