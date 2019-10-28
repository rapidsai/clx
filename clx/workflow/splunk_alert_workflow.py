import logging
import clx
import clx.mlstats
import cudf
from clx.workflow.workflow import Workflow
from clx.parsers.splunk_notable_parser import SplunkNotableParser

log = logging.getLogger(__name__)


class SplunkAlertWorkflow(Workflow):
    def __init__(
        self,
        name,
        source=None,
        destination=None,
        interval="day",
        threshold=2.5,
        window=7,
        raw_data_col_name="_raw",
    ):
        self.interval = interval
        self._threshold = threshold
        self._window = window
        self._snp = SplunkNotableParser()
        self._raw_data_col_name = raw_data_col_name
        Workflow.__init__(self, name, source, destination)

    @property
    def interval(self):
        """Interval can be set to day or hour by which z score will be calculated"""
        return self._interval

    @interval.setter
    def interval(self, interval):
        if interval != "day" and interval != "hour":
            raise Exception(
                "interval='" + interval + "': interval must be set to 'day' or 'hour'"
            )
        else:
            self._interval = interval

    @property
    def threshold(self):
        """Threshold by which to flag z score. Threshold will be flagged for scores >threshold or <-threshold"""
        return self._threshold

    @property
    def window(self):
        """Window by which to calculate rolling z score"""
        return self._window

    @property
    def raw_data_col_name(self):
        """Dataframe column name containing raw splunk alert data"""
        return self._raw_data_col_name

    def workflow(self, dataframe):
        log.debug("Processing splunk alert workflow data...")
        parsed_df = self._snp.parse(dataframe, self._raw_data_col_name)
        interval = self._interval
        threshold = float(self._threshold)

        # Create alerts dataframe
        alerts_gdf = parsed_df
        alerts_gdf["time"] = alerts_gdf["time"].astype("int")
        alerts_gdf = alerts_gdf.rename(columns={"search_name": "rule"})
        if interval == "day":
            alerts_gdf[interval] = alerts_gdf.time.applymap(self.__round2day)
        else:  # hour
            alerts_gdf[interval] = alerts_gdf.time.applymap(self.__round2hour)

        # Group alerts by interval and pivot table
        day_rule_df = (
            alerts_gdf[["rule", interval, "time"]]
            .groupby(["rule", interval])
            .count()
            .reset_index()
        )
        day_rule_df.columns = ["rule", interval, "count"]
        day_rule_piv = self.__pivot_table(
            day_rule_df, interval, "rule", "count"
        ).fillna(0)

        # Calculate rolling zscore
        r_zscores = cudf.DataFrame()
        for rule in day_rule_piv.columns:
            x = day_rule_piv[rule]
            r_zscores[rule] = clx.mlstats.rzscore(x, self._window)

        # Flag z score anomalies
        output = self.__flag_anamolies(r_zscores, threshold)
        output[interval] = day_rule_piv.index
        log.debug(output)
        return output

    def __flag_anamolies(self, zc_df, threshold):
        for col in zc_df.columns:
            if col != self._interval:
                zc_df[col] = zc_df[col].abs()
                zc_df[col] = zc_df[col].nans_to_nulls()
                zc_df[col] = zc_df[col].where(zc_df[col] > threshold, None)
                zc_df[col + "_flag"] = zc_df[col].notna()
        return zc_df

    # cuDF Feature request: https://github.com/rapidsai/cudf/issues/1214
    def __pivot_table(self, gdf, index_col, piv_col, v_col):
        index_list = gdf[index_col].unique()
        piv_gdf = cudf.DataFrame({index_col: list(range(len(index_list)))})
        piv_gdf[index_col] = index_list
        for group in gdf[piv_col].unique():
            temp_df = gdf[gdf[piv_col] == group]
            temp_df = temp_df[[index_col, v_col]]
            temp_df.columns = [index_col, group]
            piv_gdf = piv_gdf.merge(temp_df, on=[index_col], how="left")
        piv_gdf = piv_gdf.set_index(index_col)
        piv_gdf = piv_gdf.sort_index()
        return piv_gdf

    def __round2day(self, epoch_time):
        return int(epoch_time / 86400) * 86400

    def __round2hour(self, epoch_time):
        return int(epoch_time / 3600) * 3600
