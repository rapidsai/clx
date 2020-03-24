import cudf
import torch
from clx.analytics.cybert import Cybert

cy = Cybert()

def test_preprocess():
    """ Test preprocessing of data into cybert. Data (within a dataframe) will first be tokenized by the tokenizer, 
    then output input_ids, attention masks, and meta_data"""
    df = cudf.DataFrame()
    df['test'] = ['This is a test', 'This is another test']
    input_ids, attention_masks, meta_data = cy.preprocess(df)
    assert isinstance(input_ids, torch.Tensor)
    assert len(input_ids) == len(df['test'])
    assert cy.max_seq_len == len(input_ids[0])
    assert isinstance(attention_masks, torch.Tensor)
    assert len(attention_masks) == len(df['test'])
    assert cy.max_seq_len == len(attention_masks[0])