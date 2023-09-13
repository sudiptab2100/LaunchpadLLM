from embeddings_lib import search_similar, get_data_texts, read_as_csv
from get_llm_model import get_model
from get_embed_model import embed_model
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json


with open('files/settings.json') as f:
    settings = json.load(f)

app = FastAPI()
chain = get_model()
embed = embed_model(f"files/{settings['embeddings']['local_model']}")

class MSG(BaseModel):
    msg: str

@app.post("/chat/")
async def create_item(item: MSG):
    query = f"### Query: {item.msg}\n### Expression: "
    idxs = search_similar(query, embed, 10)
    similar_texts = get_data_texts(idxs)
    
    # 'prompt_q' is the prompt that will be sent to the LLM model
    '''
    Prompt Format:
        [INST] <<SYS>>
        You're a AI expression generator who uses it's own knowledge and '### Query' & '### Expression' examples below to generate appropiate 'expresion' for the given 'query'. Return 'FAILED' if unable to generate correct 'expression' for the given query.Not equal to sign is '<>'.
        <</SYS>>
        
        {INPUT} [/INST]
    '''
    prompt_q = "[INST] <<SYS>>\nYou're a AI expression generator who uses it's own knowledge and '### Query' & '### Expression' examples below to generate appropiate 'expresion' for the given 'query'. Return 'FAILED' if unable to generate correct 'expression' for the given query.Not equal to sign is '<>'.\n<</SYS>>\n\n"
    for _ in similar_texts: prompt_q += _ + '\n'
    prompt_q += query + "[/INST]" # Final prompt
    print(f"Prompt:\n{prompt_q}")
    
    result = chain.run(prompt_q)
    return result


uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")