import cudf
import os

class Resources:

    _instance = None
    
    @staticmethod
    def get_instance():
        if Resources._instance == None:
            Resources()
        return Resources._instance

    def __init__(self):
        if Resources._instance != None:
            raise Exception("This is a singleton class")
        else:
            Resources._instance = self
            Resources._instance._iana_lookup_df = self._load_iana_lookup_df()
            
    @property
    def iana_lookup_df(self):
        return self._iana_lookup_df
    
    
    def _load_iana_lookup_df(self):
        iana_path = "%s/resources/iana_port_lookup.csv" % os.path.dirname(
            os.path.realpath(__file__)
        )
        colNames = ["port","service"]
        colTypes = ["int64","str"]
        iana_lookup_df = cudf.read_csv(iana_path, delimiter=',',
                names=colNames,
                dtype=colTypes,
                skiprows=1)
        iana_lookup_df = iana_lookup_df.dropna()
        iana_lookup_df = iana_lookup_df.groupby(["port"]).min().reset_index()

        return iana_lookup_df

def major_ports(addr_col, port_col, min_conns=1, eph_min=10000):

    # Count the number of connections across each src ip-port pair
    nodes_gdf = cudf.DataFrame([("addr", addr_col), ("port", port_col)])
    nodes_gdf["conns"] = 1.0
    nodes_gdf = nodes_gdf.groupby(["addr", "port"], as_index=False).count()

    # Calculate average number of connections across all ports for each ip
    cnt_avg_gdf = nodes_gdf[["addr", "conns"]]
    cnt_avg_gdf = cnt_avg_gdf.groupby(["addr"], as_index=False).mean()
    cnt_avg_gdf = cnt_avg_gdf.rename(columns={"conns":"avg"})

    # Merge averages to dataframe
    nodes_gdf = nodes_gdf.merge(cnt_avg_gdf, on=['addr'], how='left')

    # Filter out all ip-port pairs below average
    nodes_gdf = nodes_gdf[nodes_gdf.conns>=nodes_gdf.avg]
    
    if min_conns > 1:
        nodes_gdf = nodes_gdf[nodes_gdf.conns >= min_conns]

    nodes_gdf = nodes_gdf.drop(['avg'])

    resources = Resources.get_instance()
    iana_lookup_df = resources.iana_lookup_df

    # Add IANA service names to node lists
    nodes_gdf = nodes_gdf.merge(iana_lookup_df, on=['port'], how='left')

    nodes_gdf.loc[nodes_gdf["port"] >= eph_min, "service"] = "ephemeral"

    nodes_gdf = nodes_gdf.groupby(["addr", "port", "service"], dropna=False, as_index=False).sum()

    return nodes_gdf
