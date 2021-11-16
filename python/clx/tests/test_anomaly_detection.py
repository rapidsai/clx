import cudf

import clx.analytics.anomaly_detection
import clx.features


def test_anomaly_detection():
    df = cudf.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "user": [
                "u1",
                "u5",
                "u4",
                "u2",
                "u3",
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
                "u1",
            ],
            "computer": [
                "c1",
                "c1",
                "c5",
                "c1",
                "c1",
                "c3",
                "c1",
                "c1",
                "c2",
                "c3",
                "c1",
                "c1",
                "c4",
                "c5",
            ],
        }
    )
    fdf = clx.features.frequency(df, "user", "computer")  # Create feature data
    actual = clx.analytics.anomaly_detection.dbscan(fdf, min_samples=2, eps=0.5)
    expected = cudf.Series([-1, -1], dtype="int32", index=None)
    expected.index = cudf.Series(["u1", "u4"])
    assert actual.equals(expected)
