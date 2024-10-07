import torch
import numpy as np

def word_embeddings (sentences, tokenizer, model):
    """
    Contextualized embeddings are based on tokens, which do not necessarily
    correspond to words. Therefore, it is a routine to combine and average out
    the embeddings of the tokens that constitute a certain word of interest.
    This routine is achieved by this function.

    Parameters
    ----------
    sentences : A list-like object
        A list of sentences/contexts. Each sentence is expected to contain
        several words. For each of these words, tokens of the word will be
        identified, embedding vectors for these tokens will be retrieved, and
        these token embedding vectors will be combined by averaging each
        dimension.
    tokenizer : transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
        A tokenizer object created by
        transformers.AutoTokenizer.from_pretrained.
    model : transformers.models.gpt2.modeling_gpt2.GPT2Model
        A model created by transformers.AutoModel.from_pretrained.

    Returns
    -------
    m : numpy.ndarray
    A three-dimensional matrix of word-embeddings. Its first, second, and third
    dimensions represent sentences (contexts), words in each sentence, and
    (semantic) dimensions, respectively. Each sentence contains a
    two-dimensional matrix, where rows are words and columns are dimensions (in
            that particular sentence/context). These two-dimensional matrices
    are padded with np.nan if the sentence contains fewer words than the
    longest sentence, so that each of these two-dimensional matrices has the
    same dimensions.
    """
    t = token_ids(sentences, tokenizer)
    m = token_embeddings(sentences, tokenizer, model)
    w = [ t.word_ids(i) for i in range(len(sentences)) ]
    w = [ np.unique([ j for j in i if not (j is None) ]).tolist() for i in w ]
    w = [ [ t.word_to_tokens(i, k) for k in j ] for i,j in enumerate(w) ]
    w = [ [ slice(*list(j)) for j in i ] for i in w ]
    m = [ np.array([ m[i, k, :].mean(0) for k in j ]) for i,j in enumerate(w) ]
    l = max([ i.shape[0] for i in m ])
    m = [ np.pad(i, [(0, l-(i.shape[0])),(0,0)], mode='constant', constant_values=np.nan) for i in m ]
    m = np.array(m)
    return m

def words (sentences, padding=True):
    """
    This function returns how words are recognized in sentences, ensuring the
    same length of elements for each sentence by padding (if padding=True).
    This function is intended to obtain word labels for a word-level embeddings
    such as returned by word_embeddings.

    Parameters
    ----------
    sentences : A list-like object
        A list of sentences. Each sentence will be split by spaces into words.
    padding : bool
        If True, all the first-level elements of the returned nested list will
        have the same length of elements, namely words, by padding with np.nan.

    Returns
    -------
    s : A (nested) list
        The first-level elements are sentences (e.g., s[0]). Each sentence
        contains words as its elements. If padding=True, all the sentences will
        have the same length of elements with padding with np.nan.
    """
    s = [ i.split(' ') for i in sentences ]
    if padding:
        l = max([ len(i) for i in s ])
        s = [ np.pad(np.array(i), (0, l-len(i)), mode='constant', constant_values='').tolist()  for i in s ]
        s = [ [ np.nan if j=='' else j for j in i ] for i in s ]
    return s

def token_embeddings (sentences, tokenizer, model):
    """
    It converts sentences to a token-based matrix of embedding vectors.

    Parameters
    ----------
    sentences : A list-like object
        A list of sentences. Each sentence will be parsed into tokens (not
        words). For each token, an embedding vector will be retrieved.
    tokenizer : transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
        A tokenizer object created by transformers.AutoTokenizer.from_pretrained.
    model : transformers.models.gpt2.modeling_gpt2.GPT2Model
        A model created by transformers.AutoModel.from_pretrained.

    Returns
    -------
    t : numpy.ndarray
        A 3-d matrix of embeddings. The first dimension represents sentences.
        The second dimension represents words (in a particular sentence). The
        third dimension represents embedding-dimensions (semantic dimensions).
    """
    t = token_ids(sentences, tokenizer)
    with torch.no_grad():
      m = model(**t, output_hidden_states=True).hidden_states[-1].numpy()
    return m

def tokens (sentences, tokenizer):
    """
    This function shows how sentences are parsed into tokens.

    Parameters
    ----------
    sentences : A list-like object
        A list of sentences. Each sentence will be parsed into tokens, which
        will be returned.
    tokenizer : transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
        A tokenizer object created by transformers.AutoTokenizer.from_pretrained.

    Returns
    -------
    t : A (nested) list
        Each element of the first level corresponds to a sentence (e.g., t[0]).
        Each sentence contains the same number of elements, which are tokens.
    """
    t = token_ids(sentences, tokenizer)
    t = [ tokenizer.convert_ids_to_tokens(i) for i in t.input_ids ]
    return t

def token_ids (sentences, tokenizer):
    """
    In contextualized embeddings, tokens are made use of, instead of words.
    Tokens are not necessarily the same as words. This function takes sentences
    and returns the IDs of the tokens constituting the words in the sentences.

    Parameters
    ----------
    sentences : A list-like object
        A list of sentences. Each sentence will be parsed into tokens (not
        words). The IDs of the tokens will be returned.
    tokenizer : transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
        A tokenizer object created by transformers.AutoTokenizer.from_pretrained.

    Returns
    -------
    t : transformers.tokenization_utils_base.BatchEncoding
        This is a dictionary-like object. It contains token IDs and
        attention_masks. The former can be accessed by the key "input_ids", and
        the latter can be accessed by the key "attention_mask". Attention masks
        indicate which token IDs are padding-tokens (i.e., 0 for padding
        tokens).
    """
    if tokenizer.pad_token is None:
        raise ValueError('Run "context_embed_utils.padding" first.')
    else:
        t = tokenizer(sentences, return_tensors="pt", padding=True)
    return t

def padding (tokenizer, model):
    """
    Sets the padding token for a tokenizer and resize the token embedding size
    for a model.

    Parameters
    ----------
    tokenizer : transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
        A tokenizer object created by transformers.AutoTokenizer.from_pretrained.
    model : transformers.models.gpt2.modeling_gpt2.GPT2Model
        A model created by transformers.AutoModel.from_pretrained.

    Returns
    -------
    tokenizer_model : tuple
        A tuple of "tokenizer" and "model", where the padding token is defined
        as the end-of-sentence token.
    """
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer_model = (tokenizer, model)
    return tokenizer_model

