import os
import pytest
import requests
from mockito import when, mock, verify
from rapidscyber.osi.virus_total import VirusTotalClient

api_key = "dummy-api-key"
client = VirusTotalClient(api_key=api_key)
response = mock(spec=requests.Response)
response.status_code = 200
response.text = {"response_code": 1}
resp_dict = dict(status_code=response.status_code, json_resp=response.text)

test_input_base_path = "%s/input" % os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
def test_file_scan(client, resp_dict, test_input_base_path):
    input_file = "%s/person.csv" % (test_input_base_path)
    when(client).post(...).thenReturn(resp_dict)
    resp = client.file_scan(input_file)
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_file_rescan(client, resp_dict):
    when(client).post(...).thenReturn(resp_dict)
    resp = client.file_rescan(
        [
            "75efd85cf6f8a962fe016787a7f57206ea9263086ee496fc62e3fc56734d4b53",
            "9f101483662fc071b7c10f81c64bb34491ca4a877191d464ff46fd94c7247115",
        ]
    )
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_file_report(client, resp_dict):
    when(client).get(...).thenReturn(resp_dict)
    resp = client.file_report(
        [
            "75efd85cf6f8a962fe016787a7f57206ea9263086ee496fc62e3fc56734d4b53-1555351539",
            "9f101483662fc071b7c10f81c64bb34491ca4a877191d464ff46fd94c7247115",
        ]
    )
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_url_scan(client, resp_dict):
    when(client).post(...).thenReturn(resp_dict)
    resp = client.url_scan(["nvidia.com", "github.com"])
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_url_report(client, resp_dict):
    when(client).post(...).thenReturn(resp_dict)
    resp = client.url_report(["nvidia.com"])
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_ipaddress_report(client, resp_dict):
    when(client).get(...).thenReturn(resp_dict)
    resp = client.ipaddress_report("90.156.201.27")
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("resp_dict", [resp_dict])
def test_domain_report(client, resp_dict):
    when(client).get(...).thenReturn(resp_dict)
    resp = client.domain_report("027.ru")
    assert resp["status_code"] == 200
    assert resp["json_resp"]["response_code"] == 1
