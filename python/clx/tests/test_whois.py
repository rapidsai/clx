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
import datetime
import whois
from clx.osi.whois import WhoIsLookupClient
from mockito import when


domains = ["nvidia.com"]
datetime_1 = datetime.datetime(2020, 5, 17)
datetime_2 = datetime.datetime(2020, 5, 18)
client = WhoIsLookupClient()

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


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("domains", [domains])
def test_whois(client, domains):
    expected_output = [{
        "domain_name": "NVIDIA.COM",
        "registrar": "Safenames Ltd",
        "emails": "abuse@safenames.net,wadmpfvzi5ei@idp.email,hostmaster@safenames.net",
        "updated_date": "05-17-2020 00:00:00,05-18-2020 00:00:00",
    }]
    when(whois).whois(...).thenReturn(response)
    actual_output = client.whois(domains)
    assert actual_output[0]["domain_name"] == "NVIDIA.COM"
    assert len(actual_output) == len(domains)
    assert actual_output == expected_output
