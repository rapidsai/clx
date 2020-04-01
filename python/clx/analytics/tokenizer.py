from clx.analytics import tokenizer_wrapper


def tokenize_file(input_file, hash_file="default", max_sequence_length=64, stride=48, do_lower=True, do_truncate=False, max_num_sentences=100, max_num_chars=100000, max_rows_tensor=500):
    """
    Run CUDA BERT wordpiece tokenizer on file. Encodes words to token ids using vocabulary from a pretrained tokenizer.

    :param input_file: path to input file, each line represents one sentence to be encoded
    :type input_file: str
    :param hash_file: path to hash file containing vocabulary and ids from a pretrained tokenizer
    :type hash_file: str
    :param max_sequence_length: Limits the length of the sequence returned. If tokenized sentence is shorter than max_sequence_length, output will be padded with 0s. If the tokenized sentence is longer than max_sequence length and do_truncate is set to false, there will be multiple returned sequences containing the overflowing token ids.
    :type max_sequence_length: int
    :param stride: If do_truncate is set to false and the tokenized sentence is larger than max_sequence_length, the sequences containing the overflowing token ids can contain duplicated token ids from the main sequence. If max_sequence_length is equal to stride there are no duplicated id tokens. If stride is 80% of max_sequence_length, 20% of the first sequence chunk will be repeated on the second sequence chunk and so on until the entire sentence is encoded.
    :type stride: int
    :param do_lower: If set to true, original text will be lowercased before encoding.
    :type do_lower: bool
    :param do_truncate: If set to true, sentences will be truncated and padded to max_sequence_length. Each input sentence will result in exactly one output sequence. If set to false, there will be multiple output sequences when the max_sequence_length is smaller than the tokenized sentence.
    :type do_truncate: bool
    :param max_num_sentences: max num sentences to be encoded in one batch
    :type max_num_sentences: int
    :param max_num_chars: max num characters in file
    :type max_num_chars: int
    :param max_rows_tensor: max num of rows in an output tensor
    :type max_rows_tensor: int
    :return: tokens: token ids encoded from sentences padded with 0s to max_sequence_length
    :rtype: torch.Tensor
    :return: attention_masks: binary tensor indicating the position of the padded indices so that the model does not attend to them
    :rtype: torch.Tensor
    :return: metadata: for each row of the output tensors, the meta_data contains the index id of the original sentence encoded, and the first and last index of the token ids that are non-padded and non-overlapping
    :rtype: torch.Tensor

    Examples
    --------
    >>> from clx.analytics import tokenizer
    >>> tokens, masks, metadata = tokenizer.tokenize_file("input.txt")
    """
    tokens, masks, metadata = tokenizer_wrapper.tokenize_file(input_file, hash_file, max_sequence_length, stride, do_lower, do_truncate, max_num_sentences, max_num_chars, max_rows_tensor)
    return tokens, masks, metadata


def tokenize_df(input_df, hash_file="default", max_sequence_length=64, stride=48, do_lower=True, do_truncate=False, max_num_sentences=100, max_num_chars=100000, max_rows_tensor=500):
    """
    Run CUDA BERT wordpiece tokenizer on cuDF dataframe. Encodes words to token ids using vocabulary from a pretrained tokenizer.

    :param input_df: input dataframe, each row represents one sentence to be encoded
    :type input_df: cudf.DataFrame
    :param hash_file: path to hash file containing vocabulary and ids from a pretrained tokenizer
    :type hash_file: str
    :param max_sequence_length: Limits the length of the sequence returned. If tokenized sentence is shorter than max_sequence_length, output will be padded with 0s. If the tokenized sentence is longer than max_sequence length and do_truncate is set to false, there will be multiple returned sequences containing the overflowing token ids.
    :type max_sequence_length: int
    :param stride: If do_truncate is set to false and the tokenized sentence is larger than max_sequence_length, the sequences containing the overflowing token ids can contain duplicated token ids from the main sequence. If max_sequence_length is equal to stride there are no duplicated id tokens. If stride is 80% of max_sequence_length, 20% of the first sequence chunk will be repeated on the second sequence chunk and so on until the entire sentence is encoded.
    :type stride: int
    :param do_lower: If set to true, original text will be lowercased before encoding.
    :type do_lower: bool
    :param do_truncate: If set to true, sentences will be truncated and padded to max_sequence_length. Each input sentence will result in exactly one output sequence. If set to false, there will be multiple output sequences when the max_sequence_length is smaller than the tokenized sentence.
    :type do_truncate: bool
    :param max_num_sentences: max num sentences to be encoded in one batch
    :type max_num_sentences: int
    :param max_num_chars: max num characters in dataframe
    :type max_num_chars: int
    :param max_rows_tensor: max num of rows in an output tensor
    :type max_rows_tensor: int
    :return: tokens: token ids encoded from sentences padded with 0s to max_sequence_length
    :rtype: torch.Tensor
    :return: attention_masks: binary tensor indicating the position of the padded indices so that the model does not attend to them
    :rtype: torch.Tensor
    :return: metadata: for each row of the output tensors, the meta_data contains the index id of the original sentence encoded, and the first and last index of the token ids that are non-padded and non-overlapping
    :rtype: torch.Tensor

    Examples
    --------
    >>> from clx.analytics import tokenizer
    >>> import cudf
    >>> df = cudf.read_csv("input.txt")
    >>> tokens, masks, metadata = tokenizer.tokenize_file(df)
    """
    tokens, masks, metadata = tokenizer_wrapper.tokenize_df(input_df, hash_file, max_sequence_length, stride, do_lower, do_truncate, max_num_sentences, max_num_chars, max_rows_tensor)
    return tokens, masks, metadata
