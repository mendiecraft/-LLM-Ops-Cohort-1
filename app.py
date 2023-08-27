import os
import openai

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
import chainlit as cl


openai.api_key = os.environ.get("OPENAI_API_KEY")

# try:
#     # rebuild storage context
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     # load index
#     index = load_index_from_storage(storage_context)
# except:
#     from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

#     documents = SimpleDirectoryReader("./data").load_data()
#     index = GPTVectorStoreIndex.from_documents(documents)
#     index.storage_context.persist()
documents = SimpleDirectoryReader(
    input_files=["hitchhikers.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(documents)
ft_model_id='ft:gpt-3.5-turbo-0613:handshake::7raTNuPU'

@cl.on_chat_start
async def factory():
    msg = cl.Message(content=f"Building Index...")
    await msg.send()
    # llm_predictor = LLMPredictor(
    #     llm=ChatOpenAI(
    #         temperature=0,
    #         model_name="gpt-3.5-turbo",
    #         streaming=True,
    #     ),
    # )
    # service_context = ServiceContext.from_defaults(
    #     llm_predictor=llm_predictor,
    #     chunk_size=512,
    #     callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    # )

    gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
     callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    ft_context = ServiceContext.from_defaults(
    llm=OpenAI(model=ft_model_id, temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
  callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
)

    query_engine = index.as_query_engine(
        service_context=gpt_35_context
    )

    finetune_query_engine = index.as_query_engine(service_context=ft_context)

    msg.content = f"Index built!"
    await msg.send()

    cl.user_session.set("query_engine", query_engine)

    cl.user_session.set("finetune_query_engine", finetune_query_engine)


@cl.on_message
async def main(message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message)

    response_message = cl.Message(content="")
    response_message.content = "Base Model Response"

    elements = [
    cl.Text(content=response.response, display="inline")
]
    response_message.elements = elements

    await response_message.send()

    finetune_query_engine = cl.user_session.get("finetune_query_engine")  # type: RetrieverQueryEngine
    finetune_response = await cl.make_async(finetune_query_engine.query)(message)
    
    finetune_response_message = cl.Message(content="")
    finetune_response_message.content = 'Finetune Model Response'

    elements = [
    cl.Text( content=finetune_response.response, display="inline")
]
    finetune_response_message.elements = elements

    await finetune_response_message.send()

