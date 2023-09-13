from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.chains import LLMChain
import json


with open('files/settings.json') as f:
    settings = json.load(f)
model_path = f"files/{settings['model']}"

def get_model():
    n_gpu_layers = 0
    n_batch = 512
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
        temperature=.6,
        stop=["### Query", "\n"]
    )
    
    template = "{concept}"
    
    prompt = PromptTemplate(
        input_variables=["concept"],
        template=template,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
