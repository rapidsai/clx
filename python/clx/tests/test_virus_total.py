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
import cudf
from mockito import when, mock
from clx.osi.virus_total import VirusTotalClient

api_key = "dummy-api-key"
client = VirusTotalClient(api_key=api_key)

test_input_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)


ipaddress_report_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{   "response_code": 1,   "verbose_msg": "IP address found in dataset",   "asn": "25532",   "country": "RU",   "resolutions": [{     "last_resolved": "2013-04-08 00:00:00",     "hostname": "90.156.201.27"   }, {     "last_resolved": "2013-04-08 00:00:00",     "hostname": "auto.rema-tiptop.ru"   }],   "detected_urls": [{     "url": "http://027.ru/",     "positives": 2,     "total": 37,     "scan_date": "2013-04-07 07:18:09"   }],   "detected_downloaded_samples": [{     "date": "2018-03-29 18:38:05",     "positives": 2,     "total": 59,     "sha256": "d9cacb75a3fd126762f348d00fb6e3809ede2c13b2ad251831e130bcb7ae7a84"   }, {     "date": "2018-03-29 08:52:38",     "positives": 2,     "total": 59,     "sha256": "416751ebbd5d6c37bb20233a39ade80db584057f3d5c4bbf976ce9c332836707"   }],   "undetected_downloaded_samples": [{     "date": "2018-03-28 06:36:55",     "positives": 0,     "total": 0,     "sha256": "4a91398fd21f2d0b09fc7478d016d4a8fc9fe6f1c01e10b8e7c725542260cd9f"   }],   "undetected_urls": [     "http://zadiplomom.ru/",     "3aafd5a54bb034882b8f5544bb647b6841bcb6ce938c40fb92be4cb84f2f0983",     0,     67,     "2018-02-19 18:04:15"   ] }',
    },
    spec=requests.Response,
)

put_comment_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{"response_code": 1, "verbose_msg": "Your comment was successfully posted"}',
    },
    spec=requests.Response,
)

domain_report_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{   "undetected_referrer_samples": [{     "date": "2018-03-04 16:38:06",     "positives": 0,     "total": 66,     "sha256": "ce08cf22949b6b6fcd4e61854ce810a4f9ee04529340dd077fa354d759dc7a95"   }, {     "positives": 0,     "total": 53,     "sha256": "b8f5db667431d02291eeec61cf9f0c3d7af00798d0c2d676fde0efb0cedb7741"   }],   "whois_timestamp": 1520586501,   "detected_downloaded_samples": [{     "date": "2013-06-20 18:51:30",     "positives": 2,     "total": 46,     "sha256": "cd8553d9b24574467f381d13c7e0e1eb1e58d677b9484bd05b9c690377813e54"   }],   "detected_referrer_samples": [],   "undetected_downloaded_samples": [{     "date": "2018-01-14 22:34:24",     "positives": 0,     "total": 70,     "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"   }],   "resolutions": [{     "last_resolved": "2018-03-09 00:00:00",     "ip_address": "185.53.177.31"   }, {     "last_resolved": "2013-06-20 00:00:00",     "ip_address": "90.156.201.97"   }],   "subdomains": [     "test.027.ru",     "www.027.ru"   ],   "categories": [     "parked",     "uncategorized"   ],   "domain_siblings": [],   "undetected_urls": [],   "response_code": 1,   "verbose_msg": "Domain found in dataset",   "detected_urls": [{     "url": "http://027.ru/",     "positives": 2,     "total": 67,     "scan_date": "2018-04-01 15:51:22"   }, {     "url": "http://027.ru/adobe/flash_install_v10x1.php",     "positives": 5,     "total": 67,     "scan_date": "2018-03-26 09:22:43"   }, {     "url": "http://027.ru/track.php",     "positives": 4,     "total": 66,     "scan_date": "2018-01-14 22:39:41"   }, {     "url": "http://027.ru/track.php?domain=027.ru&caf=1&toggle=answercheck",     "positives": 2,     "total": 66,     "scan_date": "2018-01-09 22:19:43"   }, {     "url": "https://027.ru/",     "positives": 1,     "total": 66,     "scan_date": "2016-02-08 13:25:40"   }] }',
    },
    spec=requests.Response,
)

url_scan_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{  "response_code": 1,  "verbose_msg": "Scan request successfully queued, come back later for the report",  "scan_id": "1db0ad7dbcec0676710ea0eaacd35d5e471d3e11944d53bcbd31f0cbd11bce31-1320752364",  "scan_date": "2011-11-08 11:39:24",  "url": "http://www.virustotal.com/",  "permalink": "http://www.virustotal.com/url/1db0ad7dbcec0676710ea0eaacd35d5e471d3e11944d53bcbd31f0cbd11bce31/analysis/1320752364/" }',
    },
    spec=requests.Response,
)

url_report_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{  "response_code": 1,  "verbose_msg": "Scan finished, scan information embedded in this object",  "scan_id": "1db0ad7dbcec0676710ea0eaacd35d5e471d3e11944d53bcbd31f0cbd11bce31-1390467782",  "permalink": "https://www.virustotal.com/url/__urlsha256__/analysis/1390467782/",  "url": "http://www.virustotal.com/",  "scan_date": "2014-01-23 09:03:02",  "filescan_id": null,  "positives": 0,  "total": 51,  "scans": {     "CLEAN MX": {       "detected": false,        "result": "clean site"     },     "MalwarePatrol": {       "detected": false,        "result": "clean site"     }   } }',
    },
    spec=requests.Response,
)

file_report_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{  "response_code": 1,  "verbose_msg": "Scan finished, scan information embedded in this object",  "resource": "99017f6eebbac24f351415dd410d522d",  "scan_id": "52d3df0ed60c46f336c131bf2ca454f73bafdc4b04dfa2aea80746f5ba9e6d1c-1273894724",  "md5": "99017f6eebbac24f351415dd410d522d",  "sha1": "4d1740485713a2ab3a4f5822a01f645fe8387f92",  "sha256": "52d3df0ed60c46f336c131bf2ca454f73bafdc4b04dfa2aea80746f5ba9e6d1c",  "scan_date": "2010-05-15 03:38:44",  "permalink": "https://www.virustotal.com/file/52d3df0ed60c46f336c131bf2ca454f73bafdc4b04dfa2aea80746f5ba9e6d1c/analysis/1273894724/",  "positives": 40,  "total": 40,  "scans": {    "nProtect": {      "detected": true,       "version": "2010-05-14.01",       "result": "Trojan.Generic.3611249",       "update": "20100514"    },    "CAT-QuickHeal": {      "detected": true,       "version": "10.00",       "result": "Trojan.VB.acgy",       "update": "20100514"    },    "McAfee": {      "detected": true,       "version": "5.400.0.1158",       "result": "Generic.dx!rkx",       "update": "20100515"    },    "TheHacker": {      "detected": true,       "version": "6.5.2.0.280",       "result": "Trojan/VB.gen",       "update": "20100514"    },       "VirusBuster": {     "detected": true,      "version": "5.0.27.0",      "result": "Trojan.VB.JFDE",      "update": "20100514"    }  } }',
    },
    spec=requests.Response,
)

file_scan_resp = mock(
    {
        "status_code": 200,
        "raise_for_status": lambda: None,
        "text": '{   "permalink": "https://www.virustotal.com/file/d140c...244ef892e5/analysis/1359112395/",   "resource": "d140c244ef892e59c7f68bd0c6f74bb711032563e2a12fa9dda5b760daecd556",   "response_code": 1,   "scan_id": "d140c244ef892e59c7f68bd0c6f74bb711032563e2a12fa9dda5b760daecd556-1359112395",   "verbose_msg": "Scan request successfully queued, come back later for the report",   "sha256": "d140c244ef892e59c7f68bd0c6f74bb711032563e2a12fa9dda5b760daecd556" }',
    },
    spec=requests.Response,
)


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("ipaddress_report_resp", [ipaddress_report_resp])
def test_ipaddress_report(client, ipaddress_report_resp):
    when(requests).get(...).thenReturn(ipaddress_report_resp)
    result = client.ipaddress_report("90.156.201.27")
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1
    assert json_resp["country"] == "RU"
    assert json_resp["asn"] == "25532"
    assert json_resp["resolutions"][0]["last_resolved"] == "2013-04-08 00:00:00"


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("put_comment_resp", [put_comment_resp])
def test_put_comment(client, put_comment_resp):
    when(requests).post(...).thenReturn(put_comment_resp)
    result = client.put_comment(
        "75efd85cf6f8a962fe016787a7f57206ea9263086ee496fc62e3fc56734d4b53",
        "This is a test comment",
    )
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1
    assert json_resp["verbose_msg"] == "Your comment was successfully posted"


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("domain_report_resp", [domain_report_resp])
def test_domain_report(client, domain_report_resp):
    when(requests).get(...).thenReturn(domain_report_resp)
    result = client.domain_report("027.ru")
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["detected_urls"][0]["url"] == "http://027.ru/"
    assert json_resp["undetected_referrer_samples"][0]["positives"] == 0
    assert json_resp["undetected_referrer_samples"][0]["total"] == 66


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("url_scan_resp", [url_scan_resp])
def test_url_scan(client, url_scan_resp):
    when(requests).post(...).thenReturn(url_scan_resp)
    result = client.url_scan(["virustotal.com"])
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1
    assert json_resp["url"] == "http://www.virustotal.com/"
    assert (
        json_resp["verbose_msg"] == "Scan request successfully queued, come back later for the report"
    )


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("url_report_resp", [url_report_resp])
def test_url_report(client, url_report_resp):
    when(requests).post(...).thenReturn(url_report_resp)
    result = client.url_report(["virustotal.com"])
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1
    assert json_resp["url"] == "http://www.virustotal.com/"
    assert not json_resp["scans"]["CLEAN MX"]["detected"]
    assert json_resp["scans"]["CLEAN MX"]["result"] == "clean site"


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("file_report_resp", [file_report_resp])
def test_file_report(client, file_report_resp):
    when(requests).get(...).thenReturn(file_report_resp)
    result = client.file_report(["99017f6eebbac24f351415dd410d522d"])
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["resource"] == "99017f6eebbac24f351415dd410d522d"
    assert json_resp["sha1"] == "4d1740485713a2ab3a4f5822a01f645fe8387f92"
    assert json_resp["scan_date"] == "2010-05-15 03:38:44"
    assert json_resp["scans"]["nProtect"]["detected"]


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("file_scan_resp", [file_scan_resp])
@pytest.mark.parametrize("test_input_df", [test_input_df])
def test_file_scan(tmpdir, client, file_scan_resp, test_input_df):
    fname = str(tmpdir.mkdir("tmp_test_virus_total").join("person.csv"))
    test_input_df.to_csv(fname, index=False)

    when(requests).post(...).thenReturn(file_scan_resp)
    result = client.file_scan(fname)
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1


@pytest.mark.parametrize("client", [client])
@pytest.mark.parametrize("file_rescan_resp", [file_scan_resp])
@pytest.mark.parametrize("test_input_df", [test_input_df])
def test_file_rescan(tmpdir, client, file_rescan_resp, test_input_df):
    fname = str(tmpdir.mkdir("tmp_test_virus_total").join("person.csv"))
    test_input_df.to_csv(fname, index=False)
    when(requests).post(...).thenReturn(file_rescan_resp)
    result = client.file_rescan(fname)
    json_resp = result["json_resp"]
    assert result["status_code"] == 200
    assert json_resp["response_code"] == 1
