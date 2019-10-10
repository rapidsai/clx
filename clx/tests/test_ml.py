import clx.ml
import cudf
import cupy as cp

def test_rzscore():
    sequence = [3,4,5,6,1,10,34,2,1,11,45,34,2,9,19,43,24,13,23,10,98,84,10]
    series = cudf.Series(sequence)
    zscores_df = cudf.DataFrame()
    zscores_df['zscore'] = clx.ml.rzscore(series, 7)
    expected_zscores_arr = [float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'),2.374423424, -0.645941275, -0.683973734, 0.158832461, 1.847751909, 0.880026019, -0.950835449, -0.360593742, 0.111407599, 1.228914145, -0.074966331, -0.570321249, 0.327849973, -0.934372308, 2.296828498, 1.282966989, -0.795223674]
    expected_zscores_df = cudf.DataFrame()
    expected_zscores_df['zscore'] = expected_zscores_arr

    # Check that columns are equal
    expected_zscores_df['zscore'] = expected_zscores_df['zscore'].fillna(0)
    zscores_df['zscore'] = zscores_df['zscore'].fillna(0)
    assert cp.allclose(expected_zscores_df['zscore'], zscores_df['zscore'])