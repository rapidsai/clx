import cudf
from clx.heuristics import ports

def test_major_ports():
    input_addr_col = cudf.Series(["10.0.75.1","10.0.75.1","10.0.75.1","10.0.75.255","10.110.104.107"])
    input_port_col = cudf.Series([137,137,7680,137,7680])

    expected = cudf.DataFrame()
    expected["addr"] = ["10.0.75.1","10.0.75.255","10.110.104.107"]
    expected["port"] = [137,137,7680]
    expected["service"] = ["netbios-ns","netbios-ns","pando-pub"]
    expected["conns"] = [2,1,1]
    
    actual = ports.major_ports(input_addr_col, input_port_col)
    
    assert actual.equals(expected)

def test_major_ports_ephemeral():
    input_addr_col = cudf.Series(["10.0.75.1","10.0.75.2","10.0.75.3","10.0.75.4"])
    input_port_col = cudf.Series([50000,60000,20000,80])

    expected = cudf.DataFrame()
    expected["addr"] = ["10.0.75.1","10.0.75.2","10.0.75.2"]
    expected["port"] = [50000,80000,20000,80]
    expected["service"] = ["ephemeral","ephemeral","dnp","http"]
    expected["conns"] = [1,1,1]

    actual = ports.major_ports(input_addr_col, input_port_col, eph_min=50000)
    
    assert actual.equals(expected)



def test_major_ports_minn_conns():
    input_addr_col = cudf.Series(["10.0.75.1","10.0.75.1","10.0.75.1","10.0.75.255","10.110.104.107"])
    input_port_col = cudf.Series([137,137,7680,137,7680])

    expected = cudf.DataFrame()
    expected["addr"] = ["10.0.75.1"]
    expected["port"] = [137]
    expected["service"] = ["netbios-ns"]
    expected["conns"] = [2]
    
    actual = ports.major_ports(input_addr_col, input_port_col, min_conns=2)

    assert actual.equals(expected)