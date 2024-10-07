import pickle
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer, AutoModel

import context_embed_utils as ce

TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
model = AutoModel.from_pretrained("gpt2")
sentences = ["university city", "a great university library"]

tokenizer_de = AutoTokenizer.from_pretrained("dbmdz/german-gpt2", clean_up_tokenization_spaces=True)
model_de = AutoModel.from_pretrained("dbmdz/german-gpt2")
sentences_de = ["Ich bin ein Mensch", "Krankenversicherungskarte"]

def test_padding ():
    t, m = ce.padding(tokenizer_de, model_de)
    assert len(t) == m.config.vocab_size

def test_word_embeddings():
    with pytest.raises(Exception) as e:
        ce.token_ids(sentences, tokenizer)
    assert str(e.value)=='Run "context_embed_utils.padding" first.'

tokenizer, model = ce.padding(tokenizer, model)
tokenizer_de, model_de = ce.padding(tokenizer_de, model_de)

def test_words ():
    ans = [['university', 'city', np.nan, np.nan], ['a', 'great', 'university', 'library']]
    tst = ce.words(sentences)
    assert (np.array(tst)==np.array(ans)).all()

def test_word_embeddings ():
    ans = np.load(RESOURCES / 'test_word_embeddings_01.npy')
    tst = ce.word_embeddings(sentences, tokenizer, model)
    assert np.allclose(ans, tst, equal_nan=True)

def test_tokens ():
    tst = ce.tokens(sentences, tokenizer)
    ans = [['un', 'iversity', 'Ġcity', '<|endoftext|>'], ['a', 'Ġgreat', 'Ġuniversity', 'Ġlibrary']]
    assert (np.array(tst)==np.array(ans)).all()

def test_token_ids ():
    tst = ce.token_ids(sentences, tokenizer)
    with open (RESOURCES / 'test_token_ids_01.pkl', 'rb') as f:
        ans = pickle.load(f)
    assert (tst['input_ids'] == ans['input_ids']).all()
    assert (tst['attention_mask'] == ans['attention_mask']).all()

def test_token_embeddings ():
    ans = np.load(RESOURCES / 'test_token_embeddings_01.npy')
    tst = ce.token_embeddings(sentences, tokenizer, model)
    assert np.allclose(ans, tst, equal_nan=True)
