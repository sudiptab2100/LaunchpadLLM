from get_hf_token import hf_token
import json
import pandas as pd
import requests
from sentence_transformers.util import semantic_search
import torch

with open('files/settings.json') as f:
    settings = json.load(f)

model_id = settings['embeddings']['model_id']
vectors_file = f"files/{settings['embeddings']['vectors_file']}"
data_file = f"files/{settings['data_file']}"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

# Read a CSV file
def read_as_csv(file_loc):
    f = pd.read_csv(file_loc)
    return f

# Get texts of the indexes from dataset
def get_data_texts(idxs):
    dataset = read_as_csv(data_file)
    q = dataset['text'] # Get text row only
    res = list()
    for idx in idxs:
        res.append(q[idx])
    return res

# Get vectors of dataset (Use API)
def data_vectors(texts, store=True):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}}) # get vectors from huggingface  
    response = response.json()
    if store: # Store the file 
        vectors = pd.DataFrame(response)
        vectors.to_csv(vectors_file, index=False)
    else: return response

# Get vectors of dataset (Use Local Model)
def data_vectors(texts, model, store=True):
    response = model.encode(texts)
    if store: # Store the file 
        vectors = pd.DataFrame(response)
        vectors.to_csv(vectors_file, index=False)
    else: return response

# Search top 'k' similar examples of query in dataset embeddings
def search_similar(query, model, k):
    query_vectors = data_vectors([query], model, False)
    query_vectors = torch.FloatTensor(query_vectors)
    dataset_embeddings = read_as_csv(vectors_file)
    dataset_embeddings = torch.from_numpy(dataset_embeddings.to_numpy()).to(torch.float)
    hits = semantic_search(query_vectors, dataset_embeddings, top_k=k)
    idxs = [] # store indexes of the texts
    for _ in hits:
        for h in _: idxs.append(h['corpus_id'])
    return idxs