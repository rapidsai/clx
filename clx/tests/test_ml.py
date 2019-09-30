import clx.ml
import cudf

def test_rzscore():
    sequence = [3,4,5,6,1,10,34,2,1,11,45,34,2,9,19,43,24,13,23,10,98,84,10]
    series = cudf.Series(sequence)
    zscores_df = cudf.DataFrame()
    zscores_df['zscore'] = clx.ml.rzscore(series, 7)
    expected_zscores_arr = [float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'),-0.66483855880834, -0.7401410439505697, 0.23676013879409535, 3.3090095948710756, 1.1734538188742984, -1.071714854816058, -0.5456968661530742, 0.2550541099647308, 1.6711139882517698, 0.04452587482515405, -0.7965172698391981, 0.18293323079510138, -0.7376624385994839, 7.172294479823011, 1.8032118470366378, -0.9852135580890902]
    expected_zscores_df = cudf.DataFrame()
    expected_zscores_df['zscore'] = expected_zscores_arr
    # TODO: Revisit, this assertion fails. Checking equality by individual element
    # assert expected_zscores_df['zscore'].equals(zscores_df['zscore'])
    for i in range(len(expected_zscores_arr)):
      assert expected_zscores_df['zscore'][i] == zscores_df['zscore'][i]
