import logging
import cudf
import pandas as pd
log = logging.getLogger(__name__)

def rzscore(input_series, window):
    """ Captures rolling z score for a given column and window size"""
    log.debug("Calculating rolling zscore...")

    # Using pandas due to std rolling window feature pending in cudf https://github.com/rapidsai/cudf/issues/2757
    input_series_pdf = input_series.to_frame().to_pandas()
    
    # Calculate rolling zscore
    r_window = input_series_pdf.rolling(window=window) # get data from rolling time window
    mean = r_window.mean() # calculate mean in window
    std_window = r_window.std(ddof=0) # calculate std in window
    z_score = (input_series_pdf-mean)/std_window # calculate z score in window
    z_score.columns = ['zscore']

    # Convert pandas Series to gpu Series
    zscore_gdf = cudf.from_pandas(z_score)
    zscore_series = zscore_gdf['zscore']

    print(zscore_series)

    return zscore_series
