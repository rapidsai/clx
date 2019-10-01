import cudf
import pytest
from clx.workflow.splunk_alert_workflow import SplunkAlertWorkflow

@pytest.mark.parametrize("threshold", [2.0])
@pytest.mark.parametrize("interval", ["day"])
@pytest.mark.parametrize("window", [7])
def test_splunk_alert_workflow(threshold, interval, window):
    """Tests the splunk alert analysis workflow"""
    sa_workflow = SplunkAlertWorkflow("splunk-alert-workflow", threshold=threshold, interval=interval, window=window)
    input_df = cudf.DataFrame(
        [
            ("severity", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("time", ["1515699589","1515705792","1515867169","1515943943","1515983612","1516034744","1516112793","1516238826","1516381833","1516515000","1516618560","1516797485","1516988701","1517106577","1517236429","1517304151","1517627976","1517772505","1517798946","1517811562","1518083921","1518119960","1518157223","1518261255",]),
            ("urgency", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("rule_name", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("security_domain", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("owner", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("savedsearch_description", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("search_name", ["Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule","Threat - HX events to Incident Review - Rule","Access - Privileged user accessing more than expected number of machines in period - Rule","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen","Threat - Source And Destination Matches - Threat Gen",]),
            ("src_port", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("dest_port", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("user", [None,None,"brianyork",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("src_ip", [None,"192.168.0.123",None,"192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123",]),
            ("dest_ip", ["192.168.0.123",None,None,"192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123","192.168.0.123",]),
            ("host", ["my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com",]),
            ("splunk_server", ["my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com","my.splunkcloud.com",]),
            ("event_type", ["Threat"," "," ",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("src_mac", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("dest_mac", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("message_hostname", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("message_ip", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("message_user_name", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]),
            ("message_description", [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,])
        ]
    )
    actual_df = sa_workflow.workflow(input_df)
    expected_df = cudf.DataFrame()
    expected_df["day"] = [1515888000,1515974400,1516060800,1516233600,1516320000,1516492800,1516579200,1516752000,1516924800,1517097600,1517184000,1517270400,1517616000,1517702400,1517788800,1518048000,1518134400,1518220800,1515628800,1515801600]
    expected_df["Access - Privileged user accessing more than expected number of machines in period - Rule_flag"] = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True]
    expected_df["Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule_flag"] = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False]
    expected_df["Threat - HX events to Incident Review - Rule_flag"] = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False]
    expected_df["Threat - Source And Destination Matches - Threat Gen_flag"] = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True, True,False,False,True,False]
    for col in expected_df.columns:
        assert expected_df[col].equals(actual_df[col])

@pytest.mark.parametrize("threshold", [1.5, 3.05, 1.0])
@pytest.mark.parametrize("interval", ["hour"])
@pytest.mark.parametrize("window", [24, 48])
def test_splunk_alert_workflow_hour(threshold, interval, window):
    sa_workflow = SplunkAlertWorkflow("splunk-alert-workflow", threshold=threshold, interval=interval)

@pytest.mark.parametrize("threshold", [2.0])
@pytest.mark.parametrize("interval", ["minute"])
def test_splunk_alert_workflow_min(threshold, interval):
    with pytest.raises(Exception) as e:
        sa_workflow = SplunkAlertWorkflow("splunk-alert-workflow", threshold=threshold, interval=interval)
    