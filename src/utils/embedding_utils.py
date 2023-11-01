import openai
import tiktoken
import numpy as np
from itertools import islice
from typing import List
from tenacity import *

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'


@retry(stop=stop_after_attempt(7))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(
        input=text_or_tokens, model=model if openai.organization else None, engine=EMBEDDING_MODEL
        if openai.organization is None else None)["data"][0]["embedding"]


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def embed_query(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    """get the embedding for a string

    Args:
        text (str): input text
        model: embedding model. Defaults to EMBEDDING_MODEL.
        max_tokens: max tokens for the model. Defaults to EMBEDDING_CTX_LENGTH.
        encoding_name: name of the encoding for tiktoken. Defaults to EMBEDDING_ENCODING.
        average (bool, optional): Average the embeddings. Defaults to True
    """
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(
            chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / \
            np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


def embed_documents(texts, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    embeddings: List[List[float]] = [[] for _ in range(len(texts))]
    for i in range(len(texts)):
        embeddings[i] = embed_query(
            texts[i], model, max_tokens, encoding_name, average=True)
    return embeddings
