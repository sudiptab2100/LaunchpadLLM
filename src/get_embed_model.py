from sentence_transformers import SentenceTransformer

def embed_model(path):
    model = SentenceTransformer(path)
    return model