from embeddings_lib import data_vectors, read_as_csv
import json

with open('files/settings.json') as f:
    settings = json.load(f)

data_file = f"files/{settings['data_file']}"
dataset = read_as_csv(data_file)
texts = list()
for q in dataset['text']: texts.append(q)
print(len(texts))
data_vectors(texts=texts) # Store vector embeddings in directory