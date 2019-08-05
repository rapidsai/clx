import pytest
import datetime
from rapidscyber.osi.whois import WhoIsLookupClient
from mockito import when, mock


domains = ["nvidia.com"]
datetime_1 = datetime.datetime(2020, 5, 17)
datetime_2 = datetime.datetime(2020, 5, 18)
client = WhoIsLookupClient()

response_dict = {
    "domain_name": "NVIDIA.COM",
    "registrar": "Safenames Ltd",
    "emails": [
        "abuse@safenames.net",
        "wadmpfvzi5ei@idp.email",
        "hostmaster@safenames.net",
    ],
}


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("domains", [domains])
def test_whois(client, domains):
    when(client).request_server(...).thenReturn(response_dict)
    result = client.whois(domains)
    assert result[0]["domain_name"] == "NVIDIA.COM"
    assert len(result) == len(domains)


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("datetime_1", [datetime_1])
@pytest.mark.parametrize("datetime_2", [datetime_2])
def test_flatten_str_array(client, datetime_1, datetime_2):
    response = {
        "domain_name": "NVIDIA.COM",
        "registrar": "Safenames Ltd",
        "emails": [
            "abuse@safenames.net",
            "wadmpfvzi5ei@idp.email",
            "hostmaster@safenames.net",
        ],
        "updated_date": [
            datetime.datetime(2020, 5, 17),
            datetime.datetime(2020, 5, 18),
        ],
    }
    resp_keys = response.keys()
    actual_output = client.flatten_str_array(response, resp_keys)
    expected_output = {
        "domain_name": "NVIDIA.COM",
        "registrar": "Safenames Ltd",
        "emails": "abuse@safenames.net,wadmpfvzi5ei@idp.email,hostmaster@safenames.net",
        "updated_date": [datetime_1, datetime_2],
    }
    assert actual_output == expected_output


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("datetime_1", [datetime_1])
@pytest.mark.parametrize("datetime_2", [datetime_2])
def test_flatten_datetime_array(client, datetime_1, datetime_2):
    response = {
        "domain_name": "NVIDIA.COM",
        "registrar": "Safenames Ltd",
        "emails": [
            "abuse@safenames.net",
            "wadmpfvzi5ei@idp.email",
            "hostmaster@safenames.net",
        ],
        "updated_date": [datetime_1, datetime_2],
    }
    resp_keys = response.keys()
    actual_output = client.flatten_datetime_array(response, resp_keys)
    expected_output = {
        "domain_name": "NVIDIA.COM",
        "registrar": "Safenames Ltd",
        "emails": [
            "abuse@safenames.net",
            "wadmpfvzi5ei@idp.email",
            "hostmaster@safenames.net",
        ],
        "updated_date": "05-17-2020 00:00:00,05-18-2020 00:00:00",
    }
    assert actual_output == expected_output
