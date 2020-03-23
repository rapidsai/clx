import cudf
import torch
from clx.analytics.cybert import Cybert

cy = Cybert()

def test_preprocess():
    df = cudf.DataFrame()
    df['test'] = ['This is a test', 'This is another test']
    input_ids, attention_masks = cy.preprocess(df)
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_masks, torch.Tensor)