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
import pytest
from clx.workflow.splunk_alert_workflow import SplunkAlertWorkflow


@pytest.mark.parametrize("threshold", [2.0])
@pytest.mark.parametrize("interval", ["day"])
@pytest.mark.parametrize("window", [7])
def test_splunk_alert_workflow(threshold, interval, window):
    """Tests the splunk alert analysis workflow"""
    sa_workflow = SplunkAlertWorkflow(
        "splunk-alert-workflow",
        threshold=threshold,
        interval=interval,
        window=window,
        raw_data_col_name="raw",
    )
    TEST_DATA = [
        '1515699589, search_name="Test Search Name", orig_time="1515699589", info_max_time="1566346500.000000000", info_min_time="1566345300.000000000", info_search_time="1566305689.361160000", message.description="Test Message Description", message.hostname="msg.test.hostname", message.ip="100.100.100.123", message.user_name="user@test.com", severity="info", urgency="medium"',
        '1515705792, search_name="Test Search Name 2", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", src_ip="10.01.01.123", src_ip="10.01.01.1, count="19", info_max_time="1548772200.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1515867169, search_name="Test Search Name 3", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", count="19", info_max_time="1548234811.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1515943943, search_name="Endpoint - Brute Force against Known User - Rule", orig_source="100.20.2.21", orig_source="FEDEX-MA", orig_source="localhost.com", failure="1104", first="Pattrick", identity="pjame", info_max_time="1546382400.000000000", info_min_time="1546378800.000000000", info_search_time="1546382850.589570000", success="8", user="pjame',
        '1515983612, search_name="Manual Notable Event - Rule", _time="1554290847", app="SplunkEnterpriseSecuritySuite", creator="test@nvidia.com", info_max_time="+Infinity", info_min_time="0.000", info_search_time="1554290847.423961000", owner="test@nvidia.com", rule_description="FireEye NX alert for Incident Review with Major Severity", rule_title="FireEye NX alert for Incident Review(Majr)", security_domain="endpoint", status="0", urgency="medium"',
        '1516034744, search_name="Endpoint - FireEye NX alert for Incident Review (Minor) - Rule", category="riskware-callback", dest_ip="10.15.90.150", occurred="Mar 09 2019 02:36:00 UTC", signature="Android.Adware.Batmobil", src_ip="10.15.90.151", dest_port="80", src_port="40472", orig_time="1552098960", info_max_time="1552099380.000000000", info_min_time="1552098780.000000000", info_search_time="1552052094.393543000", severity="minr", src_host="ip-10.5.13.compute.internal"',
        '1516112793, search_name=\\"Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule\\", count=\\"1\\", dest=\\"TEST-01\\", dest_priority=\\"medium\\", info_max_time=\\"1511389440.000000000\\", info_min_time=\\"1511388840.000000000\\", info_search_time=\\"1511389197.841039000\\", info_sid=\\"rt_scheduler_dGNhcnJvbGxAbnZpZGlhLmNvbQ__SplunkEnterpriseSecuritySuite__RMD5c5145919d43bdffc_at_1511389196_22323\\", lastTime=\\"1511388996.202094\\"',
        '1516238826, search_name="Test Search Name", orig_time="1516238826", info_max_time="1566346500.000000000", info_min_time="1566345300.000000000", info_search_time="1566305689.361160000", message.description="Test Message Description", message.hostname="msg.test.hostname", message.ip="100.100.100.123", message.user_name="user@test.com", severity="info", urgency="medium"',
        '1516381833, search_name="Test Search Name 2", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", src_ip="10.01.01.123", src_ip="10.01.01.1, count="19", info_max_time="1548772200.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1516515000, search_name="Test Search Name 3", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", count="19", info_max_time="1548234811.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1516618560, search_name="Endpoint - Brute Force against Known User - Rule", orig_source="100.20.2.21", orig_source="FEDEX-MA", orig_source="localhost.com", failure="1104", first="Pattrick", identity="pjame", info_max_time="1546382400.000000000", info_min_time="1546378800.000000000", info_search_time="1546382850.589570000", success="8", user="pjame',
        '1516797485, search_name="Manual Notable Event - Rule", _time="1554290847", app="SplunkEnterpriseSecuritySuite", creator="test@nvidia.com", info_max_time="+Infinity", info_min_time="0.000", info_search_time="1554290847.423961000", owner="test@nvidia.com", rule_description="FireEye NX alert for Incident Review with Major Severity", rule_title="FireEye NX alert for Incident Review(Majr)", security_domain="endpoint", status="0", urgency="medium"',
        '1516988701, search_name="Endpoint - FireEye NX alert for Incident Review (Minor) - Rule", category="riskware-callback", dest_ip="10.15.90.150", occurred="Mar 09 2019 02:36:00 UTC", signature="Android.Adware.Batmobil", src_ip="10.15.90.151", dest_port="80", src_port="40472", orig_time="1552098960", info_max_time="1552099380.000000000", info_min_time="1552098780.000000000", info_search_time="1552052094.393543000", severity="minr", src_host="ip-10.5.13.compute.internal"',
        '1517106577, search_name=\\"Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule\\", count=\\"1\\", dest=\\"TEST-01\\", dest_priority=\\"medium\\", info_max_time=\\"1511389440.000000000\\", info_min_time=\\"1511388840.000000000\\", info_search_time=\\"1511389197.841039000\\", info_sid=\\"rt_scheduler_dGNhcnJvbGxAbnZpZGlhLmNvbQ__SplunkEnterpriseSecuritySuite__RMD5c5145919d43bdffc_at_1511389196_22323\\", lastTime=\\"1511388996.202094\\"',
        '1517236429, search_name="Test Search Name", orig_time="1517236429", info_max_time="1566346500.000000000", info_min_time="1566345300.000000000", info_search_time="1566305689.361160000", message.description="Test Message Description", message.hostname="msg.test.hostname", message.ip="100.100.100.123", message.user_name="user@test.com", severity="info", urgency="medium"',
        '1517304151, search_name="Test Search Name 2", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", src_ip="10.01.01.123", src_ip="10.01.01.1, count="19", info_max_time="1548772200.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1517627976, search_name="Test Search Name 3", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", count="19", info_max_time="1548234811.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1517772505, search_name="Endpoint - Brute Force against Known User - Rule", orig_source="100.20.2.21", orig_source="FEDEX-MA", orig_source="localhost.com", failure="1104", first="Pattrick", identity="pjame", info_max_time="1546382400.000000000", info_min_time="1546378800.000000000", info_search_time="1546382850.589570000", success="8", user="pjame',
        '1517798946, search_name="Manual Notable Event - Rule", _time="1554290847", app="SplunkEnterpriseSecuritySuite", creator="test@nvidia.com", info_max_time="+Infinity", info_min_time="0.000", info_search_time="1554290847.423961000", owner="test@nvidia.com", rule_description="FireEye NX alert for Incident Review with Major Severity", rule_title="FireEye NX alert for Incident Review(Majr)", security_domain="endpoint", status="0", urgency="medium"',
        '1517811562, search_name="Endpoint - FireEye NX alert for Incident Review (Minor) - Rule", category="riskware-callback", dest_ip="10.15.90.150", occurred="Mar 09 2019 02:36:00 UTC", signature="Android.Adware.Batmobil", src_ip="10.15.90.151", dest_port="80", src_port="40472", orig_time="1552098960", info_max_time="1552099380.000000000", info_min_time="1552098780.000000000", info_search_time="1552052094.393543000", severity="minr", src_host="ip-10.5.13.compute.internal"',
        '1518083921, search_name=\\"Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule\\", count=\\"1\\", dest=\\"TEST-01\\", dest_priority=\\"medium\\", info_max_time=\\"1511389440.000000000\\", info_min_time=\\"1511388840.000000000\\", info_search_time=\\"1511389197.841039000\\", info_sid=\\"rt_scheduler_dGNhcnJvbGxAbnZpZGlhLmNvbQ__SplunkEnterpriseSecuritySuite__RMD5c5145919d43bdffc_at_1511389196_22323\\", lastTime=\\"1511388996.202094\\"',
        '1518119960, search_name="Test Search Name 3", signature="Android.Adware.Batmobil", signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", src="10.01.01.1", count="19", info_max_time="1548234811.000000000", info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"',
        '1518157223, search_name="Endpoint - Brute Force against Known User - Rule", orig_source="100.20.2.21", orig_source="FEDEX-MA", orig_source="localhost.com", failure="1104", first="Pattrick", identity="pjame", info_max_time="1546382400.000000000", info_min_time="1546378800.000000000", info_search_time="1546382850.589570000", success="8", user="pjame',
        '1518261255, search_name="Manual Notable Event - Rule", _time="1554290847", app="SplunkEnterpriseSecuritySuite", creator="test@nvidia.com", info_max_time="+Infinity", info_min_time="0.000", info_search_time="1554290847.423961000", owner="test@nvidia.com", rule_description="FireEye NX alert for Incident Review with Major Severity", rule_title="FireEye NX alert for Incident Review(Majr)", security_domain="endpoint", status="0", urgency="medium"',
    ]
    raw_df = cudf.DataFrame({"raw": TEST_DATA})
    actual_df = sa_workflow.workflow(raw_df)
    expected_df = cudf.DataFrame()
    expected_df["time"] = [
        1517702400,
        1516924800,
        1517097600,
        1517788800,
        1517184000,
        1517270400,
        1517616000
    ]
    expected_df["rule"] = [
        "Endpoint - Brute Force against Known User - Rule",
        "Endpoint - FireEye NX alert for Incident Review (Minor) - Rule",
        "Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule",
        "Manual Notable Event - Rule",
        "Test Search Name",
        "Test Search Name 2",
        "Test Search Name 3",
    ]
    for col in expected_df.columns:
        assert expected_df[col].equals(actual_df[col])


@pytest.mark.parametrize("threshold", [1.5, 3.05, 1.0])
@pytest.mark.parametrize("interval", ["hour"])
@pytest.mark.parametrize("window", [24, 48])
def test_splunk_alert_workflow_hour(threshold, interval, window):
    SplunkAlertWorkflow(
        "splunk-alert-workflow", threshold=threshold, interval=interval
    )


@pytest.mark.parametrize("threshold", [2.0])
@pytest.mark.parametrize("interval", ["minute"])
def test_splunk_alert_workflow_min(threshold, interval):
    with pytest.raises(Exception):
        SplunkAlertWorkflow(
            "splunk-alert-workflow", threshold=threshold, interval=interval
        )
