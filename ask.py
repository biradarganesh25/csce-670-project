import os
import pickle
import common_data
import faiss
import gradio as gr
import numpy as np
import openai
import tiktoken
from langchain.vectorstores.faiss import FAISS

openai.api_key = os.getenv("OPENAI_API_KEY")

# load faiss index
index = faiss.read_index("faiss_store.pkl")
# load docs.pkl list of documents
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
# load metadatas.pkl list of metadata
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

def get_similar_docs(query):
    # query = "what are tamu iss advising zoom meeting hours?"

    EMBEDDING_MODEL = common_data.EMBEDDING_MODEL
    # MODEL_PRICE_PER_TOKEN = "0.002"
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # tokens = encoding.encode(query)
    # print("price: ", len(tokens) * MODEL_PRICE_PER_TOKEN)
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = np.array(query_embedding_response["data"][0]["embedding"])
    # normalize query embedding
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

    distance, indices = index.search(query_embedding_normalized.reshape(1, -1), 2)
    print("similar docs:")
    for i in indices[0]:
        print(docs[i])
        print(metadatas[i])
        print("#########################")

    return [docs[i] for i in indices[0]], [metadatas[i] for i in indices[0]]

def build_prompt(query, docs):
    introduction = "Use the below paragraphs to answer the following question. If you are not sure about the answer, say 'I don't know', don't try to guess.\n"
    question = f"\n\nQuestion: {query}"
    prompt = introduction
    for doc in docs:
        prompt += f"\n\n{doc}"
    prompt += question
    print("using prompt: ", prompt)
    return prompt

def ask(query):
    docs, metadats = get_similar_docs(query)
    prompt = build_prompt(query, docs)
    if len(prompt) > common_data.TOKEN_BUDGET:
        print("prompt too long")
        return
    messages = [
        {"role": "system", "content": "You answer questions about the ISS department of TAMU."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=common_data.CHAT_MODEL,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

def conversation_history(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ''.join(s)
    output = ask(inp)
    history.append((input, output))
    return history, history

blocks = gr.Blocks()
prompt = "Hi I am Contextify! \n Ask anything about ISSS!!"
with blocks:
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit = gr.Button("Send")
    submit.click(conversation_history, inputs=[message, state], outputs=[chatbot, state])
blocks.launch(debug=True)
