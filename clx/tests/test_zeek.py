import pytest
import pandas as pd
import numpy as np

import cudf
import rapidscyber.parsers.zeek as zeek

def test_parse_log_file(tmpdir):

    header = "#separator\t\\x09\n\
        #set_separator\t,\n\
        #empty_field\t(empty)\n\
        #unset_field\t-\n\
        #path\tconn\n\
        #open\t2015-01-24-16-49-04\n\
        #fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents\n\
        #types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tcount\tstring\tcount\tcount\tcount\tcount\tset[string]\n"

    actual = pd.DataFrame()
    actual["ts"] = [1421927450.370337, 1421927658.777193]
    actual["ts"] = actual["ts"].astype("float64")
    actual["uid"] = ["CFlyqZgM1g71BYPB6", "CnKVxKIj403JsAK5k"]
    actual["id.orig_h"] = ["175.45.176.3", "175.45.176.1"]
    actual["id.orig_p"] = [7177, 24809]
    actual["id.orig_p"] = actual["id.orig_p"].astype("int32")
    actual["id.resp_h"] = ["149.171.126.16", "149.171.126.14"]
    actual["id.resp_p"] = [80, 443]
    actual["id.resp_p"] = actual["id.resp_p"].astype("int32")
    actual["proto"] = ["tcp", "tcp"]
    actual["service"] = ["http", "http"]
    actual["duration"] = [0.214392, 2.37679]
    actual["duration"] = actual["duration"].astype("float64")
    actual["orig_bytes"] = [194, 188]
    actual["orig_bytes"] = actual["orig_bytes"].astype("int64")
    actual["resp_bytes"] = [12282, 0]
    actual["resp_bytes"] = actual["resp_bytes"].astype("int64")
    actual["conn_state"] = ["SF", "SF"]
    actual["local_orig"] = [False, False]
    actual["missed_bytes"] = [12282, 0]
    actual["missed_bytes"] = actual["missed_bytes"].astype("int64")
    actual["history"] = ["ShADdFfa", "ShADFfa"]
    actual["orig_pkts"] = [12, 14]
    actual["orig_pkts"] = actual["orig_pkts"].astype("int64")
    actual["orig_ip_bytes"] = [900, 1344]
    actual["orig_ip_bytes"] = actual["orig_ip_bytes"].astype("int64")
    actual["resp_pkts"] = [24, 6]
    actual["resp_pkts"] = actual["resp_pkts"].astype("int64")
    actual["resp_ip_bytes"] = [25540, 256]
    actual["resp_ip_bytes"] = actual["resp_ip_bytes"].astype("int64")
    actual["tunnel_parents"] = ["(empty)", "(empty)"]

    footer = "#close^I2015-01-24-16-50-35"

    fname = tmpdir.mkdir("tmp_clx_zeek_test").join("tst_zeek_conn_log.csv")
    actual.to_csv(fname, sep="\t", index=False, header=False)

    with open(fname, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header + content + footer)

    parsed = zeek.parse_log_file(fname)
    assert np.allclose(parsed["ts"].to_pandas(), actual["ts"])
    assert parsed["uid"].to_pandas().equals(actual["uid"])
    assert parsed["id.orig_h"].to_pandas().equals(actual["id.orig_h"])
    assert parsed["id.orig_p"].to_pandas().equals(actual["id.orig_p"])
    assert parsed["id.resp_h"].to_pandas().equals(actual["id.resp_h"])
    assert parsed["id.resp_p"].to_pandas().equals(actual["id.resp_p"])
    assert parsed["proto"].to_pandas().equals(actual["proto"])
    assert parsed["service"].to_pandas().equals(actual["service"])
    assert np.allclose(parsed["duration"].to_pandas(), actual["duration"])
    assert parsed["orig_bytes"].to_pandas().equals(actual["orig_bytes"])
    assert parsed["resp_bytes"].to_pandas().equals(actual["resp_bytes"])
    assert parsed["conn_state"].to_pandas().equals(actual["conn_state"])
    assert parsed["local_orig"].to_pandas().equals(actual["local_orig"])
    assert parsed["missed_bytes"].to_pandas().equals(actual["missed_bytes"])
    assert parsed["history"].to_pandas().equals(actual["history"])
    assert parsed["orig_pkts"].to_pandas().equals(actual["orig_pkts"])
    assert parsed["orig_ip_bytes"].to_pandas().equals(actual["orig_ip_bytes"])
    assert parsed["resp_pkts"].to_pandas().equals(actual["resp_pkts"])
    assert parsed["resp_ip_bytes"].to_pandas().equals(actual["resp_ip_bytes"])
    assert parsed["tunnel_parents"].to_pandas().equals(actual["tunnel_parents"])