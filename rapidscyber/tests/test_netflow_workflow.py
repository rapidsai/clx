import cudf
import pytest
from rapidscyber.workflow.netflow_workflow import NetflowWorkflow


def test_netflow_workflow():
    """Tests the netflow dataframe enrichment"""
    netflow_workflow = NetflowWorkflow("netflow-workflow")
    input_df = cudf.DataFrame(
        [
            ("firstname", ["Emma", "Ava", "Sophia"]),
            ("lastname", ["Olivia", "Isabella", "Charlotte"]),
            ("gender", ["F", "F", "F"]),
        ]
    )
    actual_df = netflow_workflow.workflow(input_df)
    expected_df = cudf.DataFrame(
        [
            ("firstname", ["Emma", "Ava", "Sophia"]),
            ("lastname", ["Olivia", "Isabella", "Charlotte"]),
            ("gender", ["F", "F", "F"]),
            (
                "netflow_enriched",
                ["netflow_enriched", "netflow_enriched", "netflow_enriched"],
            ),
        ]
    )
    # Equality checks issue: https://github.com/rapidsai/cudf/issues/1750
    assert actual_df.to_pandas().equals(expected_df.to_pandas())
