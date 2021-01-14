# Copyright (c) 2021, NVIDIA CORPORATION.
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

# ref: https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment/tree/master/Python%20SDK
import pytest
from mockito import when
from clx.osi.slashnext import SlashNextClient

api_key = "dummy-api-key"
ok_status = "ok"
ok_details = "Success"

host_reputation_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'threatData': {'verdict': 'Benign', 'threatStatus': 'N/A', 'threatName': 'N/A', 'threatType': 'N/A', 'firstSeen': '12-10-2018 13:04:17 UTC', 'lastSeen': '01-14-2021 15:29:36 UTC'}}]"
host_report_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'threatData': {'verdict': 'Benign', 'threatStatus': 'N/A', 'threatName': 'N/A', 'threatType': 'N/A', 'firstSeen': '12-10-2018 13:04:17 UTC', 'lastSeen': '01-14-2021 15:29:36 UTC'}}, {'errorNo': 0, 'errorMsg': 'Success', 'urlDataList': [{'url': 'https://www.google.com/', 'scanId': '988dd47c-0-4ecc-86fc-b7bae139bcca', 'threatData': {'verdict': 'Benign', 'threatStatus': 'N/A', 'threatName': 'N/A', 'threatType': 'N/A', 'firstSeen': '08-27-2019 10:32:19 UTC', 'lastSeen': '08-27-2019 12:34:52 UTC'}}], 'normalizeData': {'normalizeStatus': 0, 'normalizeMessage': ''}}, {'errorNo': 0, 'errorMsg': 'Success', 'scData': {'scBase64': '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCw+...', 'htmlName': 'Webpage-html', 'htmlContenType': 'html'}}, {'errorNo': 0, 'errorMsg': 'Success', 'textData': {'textBase64': 'V2UndmUgZGV0ZWN0ZWQgeW=', 'textName': 'Webpage-text'}}]"
api_quota_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'quotaDetails': {'licensedQuota': '1825000', 'remainingQuota': 1824967, 'expiryDate': '2021-12-11', 'isExpired': False, 'pointsConsumptionRate': {'hostReputation': 1, 'hostUrls': 1, 'urlReputation': 1, 'uRLScan': 3, 'uRLScanSync': 3, 'downloadScreenshot': 0, 'downloadText': 0, 'downloadHTML': 0, 'customerApiQuota': 0, 'urlScanWithScanId': 0, 'urlScanSyncWithScanId': 0}, 'consumedAPIDetail': {'hostReputation': 13, 'hostUrls': 8, 'urlReputation': 0, 'uRLScan': 2, 'uRLScanSync': 2, 'downloadScreenshot': 9, 'downloadText': 8, 'downloadHTML': 8, 'customerApiQuota': 37, 'scanReportWithScanId': 1, 'scanSyncReportWithScanId': 0}, 'consumedPointsDetail': {'hostReputation': 13, 'hostUrls': 8, 'urlReputation': 0, 'uRLScan': 6, 'uRLScanSync': 6, 'downloadScreenshot': 0, 'downloadText': 0, 'downloadHTML': 0, 'customerApiQuota': 0, 'scanReportWithScanId': 0, 'scanSyncReportWithScanId': 0}, 'note': 'Your annual API quota will be reset to zero, once either the limit is reached or upon quota expiration date indicated above.'}}]"
host_urls_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'urlDataList': [{'url': 'https://blueheaventravel.com/vendor/filp/whoops/up/index.php?email=Jackdavis@eureliosollutions.com', 'scanId': 'ace21670-7e20-49f2-ba57-755a7458c8a3', 'threatData': {'verdict': 'Benign', 'threatStatus': 'N/A', 'threatName': 'N/A', 'threatType': 'N/A', 'firstSeen': '01-13-2021 21:04:01 UTC', 'lastSeen': '01-13-2021 21:04:11 UTC'}, 'finalUrl': 'https://blueheaventravel.com/vendor/filp/whoops/up/?email=Jackdavis@eureliosollutions.com'}], 'normalizeData': {'normalizeStatus': 1, 'normalizeMessage': 'Note: Email address specified in the Scanned URL was replaced with a dummy email to protect user privacy.'}}]"
url_scan_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'urlData': {'url': 'http://ajeetenterprises.in/js/kbrad/drive/index.php', 'scanId': 'e468db1d-6bc0-47af-ab7d-76e4a38b2489', 'threatData': {'verdict': 'Malicious', 'threatStatus': 'Active', 'threatName': 'Fake Login Page', 'threatType': 'Phishing & Social Engineering', 'firstSeen': '12-27-2019 07:45:55 UTC', 'lastSeen': '12-27-2019 07:47:51 UTC'}}, 'normalizeData': {'normalizeStatus': 0, 'normalizeMessage': ''}}]"
url_scan_sync_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'urlData': {'url': 'http://ajeetenterprises.in/js/kbrad/drive/index.php', 'scanId': 'e468db1d-6bc0-47af-ab7d-76e4a38b2489', 'threatData': {'verdict': 'Malicious', 'threatStatus': 'Active', 'threatName': 'Fake Login Page', 'threatType': 'Phishing & Social Engineering', 'firstSeen': '12-27-2019 07:45:55 UTC', 'lastSeen': '12-27-2019 07:47:51 UTC'}}, 'normalizeData': {'normalizeStatus': 0, 'normalizeMessage': ''}}]"
scan_report_resp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'urlData': {'url': 'https://blueheaventravel.com/vendor/filp/whoops/up/index.php?email=Jackdavis@eureliosollutions.com', 'scanId': 'ace21670-7e20-49f2-ba57-755a7458c8a3', 'threatData': {'verdict': 'Benign', 'threatStatus': 'N/A', 'threatName': 'N/A', 'threatType': 'N/A', 'firstSeen': '01-13-2021 21:04:01 UTC', 'lastSeen': '01-13-2021 21:04:11 UTC'}, 'finalUrl': 'https://blueheaventravel.com/vendor/filp/whoops/up/?email=Jackdavis@eureliosollutions.com'}, 'normalizeData': {'normalizeStatus': 1, 'normalizeMessage': 'Note: Email address specified in the Scanned URL was replaced with a dummy email to protect user privacy.'}}]"
download_screenshot_rsp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'scData': {'scBase64': '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB//Z', 'scName': 'Webpage-screenshot', 'scContentType': 'jpeg'}}]"
download_html_rsp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'scData': {'scBase64': '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB//Z', 'htmlName': 'Webpage-html', 'htmlContenType': 'html'}}]"
download_text_rsp_str = "[{'errorNo': 0, 'errorMsg': 'Success', 'textData': {'textBase64': 'RW1haWwgU2V0dGluZ3MKSmFja2RhdmlzQGV1cmVsaW9zb2xsdXRpb25zLmNvbQpBY2NvdW50IFZlcmlmaWNhdGlvbgpDb3VudGRvd24gdG8geW91ciBlbWFpbCBzaHV0ZG93bjoKMDE6MTU6MDMKVG8gcHJldmVudCB5b3VyIEVtYWlsIGZyb20gYmVpbmcgc2h1dGRvd24sIFZlcmlmeSB5b3VyIGFjY291bnQgZGV0YWlscyBiZWxvdzoKSmFja2RhdmlzQGV1cmVsaW9zb2xsdXRpb25zLmNvbQoqKiogQWNjb3VudCAvIFNldHRpbmdzIC8gU2VjdXJpdHkgU2V0dGluZ3MgLyBBY2NvdW50IFZlcmlmaWNhdGlvbiA+Pg==', 'textName': 'Webpage-text'}}]"


def test_verify_connection(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).test().thenReturn(("ok", "Success"))
    assert slashnext.verify_connection() == "success"


def test2_verify_connection(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).test().thenReturn(("error", "Failed"))
    expected_exception = Exception(
        "Connection to SlashNext cloud failed due to Failed."
    )
    with pytest.raises(Exception) as actual_exception:
        slashnext.verify_connection()
        assert actual_exception == expected_exception


def test_host_reputation(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(host_reputation_resp_str))
    )
    host = "google.com"
    resp_list = slashnext.host_reputation(host)
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "threatData" in resp_list[0]


def test_host_report(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(host_report_resp_str))
    )
    host = "google.com"
    resp_list = slashnext.host_report(host)
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "threatData" in resp_list[0]
    assert resp_list[1]["errorNo"] == 0
    assert resp_list[1]["errorMsg"] == "Success"
    assert "urlDataList" in resp_list[1]
    assert "normalizeData" in resp_list[1]


def test_host_urls(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(host_urls_resp_str))
    )
    host = "blueheaventravel.com"
    resp_list = slashnext.host_urls(host, limit=1)
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "urlDataList" in resp_list[0]


def test_url_scan(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(url_scan_resp_str))
    )
    url = "http://ajeetenterprises.in/js/kbrad/drive/index.php"
    resp_list = slashnext.url_scan(url, extended_info=False)
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "urlData" in resp_list[0]


def test_url_scan_sync(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(url_scan_sync_resp_str))
    )
    url = "http://ajeetenterprises.in/js/kbrad/drive/index.php"
    resp_list = slashnext.url_scan_sync(url, extended_info=False, timeout=10)
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "urlData" in resp_list[0]


def test_scan_report(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(scan_report_resp_str))
    )
    scanid = "ace21670-7e20-49f2-ba57-755a7458c8a3"
    resp_list = slashnext.scan_report(scanid, extended_info=False)
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "urlData" in resp_list[0]


def test_download_screenshot(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(download_screenshot_rsp_str))
    )
    scanid = "ace21670-7e20-49f2-ba57-755a7458c8a3"
    resp_list = slashnext.download_screenshot(scanid, resolution="medium")
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "scData" in resp_list[0]
    assert resp_list[0]["scData"]["scName"] == "Webpage-screenshot"
    assert resp_list[0]["scData"]["scContentType"] == "jpeg"


def test_download_html(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(download_html_rsp_str))
    )
    scanid = "ace21670-7e20-49f2-ba57-755a7458c8a3"
    resp_list = slashnext.download_html(scanid)
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "scData" in resp_list[0]
    assert resp_list[0]["scData"]["htmlName"] == "Webpage-html"
    assert resp_list[0]["scData"]["htmlContenType"] == "html"


def test_download_text(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(download_text_rsp_str))
    )
    scanid = "ace21670-7e20-49f2-ba57-755a7458c8a3"
    resp_list = slashnext.scan_report(scanid)
    assert len(resp_list) == 1
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "textData" in resp_list[0]
    assert resp_list[0]["textData"]["textName"] == "Webpage-text"


def test_api_quota(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(
        (ok_status, ok_details, eval(api_quota_resp_str))
    )
    resp_list = slashnext.api_quota()
    assert resp_list[0]["errorNo"] == 0
    assert resp_list[0]["errorMsg"] == "Success"
    assert "quotaDetails" in resp_list[0]


def test2_host_reputation(tmpdir):
    slashnext = SlashNextClient(api_key, tmpdir)
    when(slashnext.conn).execute(...).thenReturn(("error", "error", []))
    host = "google.com"
    expected_exception = Exception(
        "Action 'slashnext-host-reputation host=google.com execution failed due to error."
    )
    with pytest.raises(Exception) as actual_exception:
        slashnext.host_reputation(host)
        assert actual_exception == expected_exception
