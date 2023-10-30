# -*- coding: utf-8 -*-
"""RAG_pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bSiSjyKwxhCEZansvBH9SkIMuhlqqYco
"""

!pip install chromadb sentence-transformers -q

from transformers.models.esm.openfold_utils.tensor_utils import List
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List
from uuid import uuid4

import chromadb
client = chromadb.Client()


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] ## output is a tuple and you select the first element for the embeddings , shape - (n,seq_len,384)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_doc_embeddings(sentence_embedding_model: str, docs: list, use_hf: bool) -> List:
    if use_hf:
        tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model)
        embedding_model = AutoModel.from_pretrained(sentence_embedding_model)
        encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
        query_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings
    else:
        return NotImplementedError('Non HF models not supported currently')

def store_doc_embeddings(collection_name: str, docs: list, doc_embeddings: list):
    assert len(docs) == len(doc_embeddings), 'Length of docs and doc embeddings is not equal'
    collection_metadata={"hnsw:space": "cosine"}
    doc_collection = client.get_or_create_collection(collection_name, metadata=collection_metadata)
    doc_collection.upsert(ids=[str(uuid4()) for i in range(len(docs))],
                          embeddings=doc_embeddings,
                          metadatas=[{'doc_name': f'Doc {i}'} for i in range(len(docs))],
                          documents=docs)
    print(f"Successfully inserted {len(docs)} Docs into Collection: {collection_name}")




def get_query_embeddings(query_embedding_model: str, query_sents: list, use_hf: bool) -> List:
    if use_hf:
        tokenizer = AutoTokenizer.from_pretrained(query_embedding_model)
        embedding_model = AutoModel.from_pretrained(query_embedding_model)
        encoded_input = tokenizer(query_sents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
        query_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings
    else:
        return NotImplementedError('Non HF models not supported currently')

from transformers import pipeline, set_seed

set_seed(32)


generator = pipeline('text-generation', model="facebook/opt-1.3b", do_sample=True)
print(f'Generation without relevant context:\n')
generator("Who lives in the Imperial Palace in Tokyo?")

docs = ['The Imperial Palace is the main residence of the Emperor of Tokyo. It is a large park-like area located in the Chiyoda district of the Chiyoda ward of Tokyo', 'Tokyo is the capital of Japan. Japan is a country in Asia.']
doc_embeddings = get_doc_embeddings('sentence-transformers/all-MiniLM-L6-v2', docs, True)
store_doc_embeddings('My_Doc_Collection', docs, doc_embeddings.numpy().tolist())

collection = client.get_collection('My_Doc_Collection')
query_texts = ['Who lives in the Imperial Palace in Tokyo?']
relevant_documents = collection.query(query_embeddings=None, query_texts=query_texts)['documents']

top_k = 1
relevant_documents = relevant_documents[0][:top_k]

context = " ".join(relevant_documents)
question = query_texts[0]

print(f'Generation with relevant context:\n')
generator(f"Given {context}\n Now, answer this question: {question}", max_new_tokens=100)
