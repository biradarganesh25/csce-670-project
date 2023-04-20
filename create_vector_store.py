import os
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import openai
import faiss
import numpy as np
#read open ai api key from env
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORGANIZATION")
# print(openai.api_key)
# print(openai.organization)

def build_text_and_sources(txt_files):
	text = []
	sources = []
	with open('links.txt', 'r') as f:
		urls = f.readlines()
	for i,txt_file in enumerate(txt_files):
		with open(txt_file, encoding="utf8") as f:
			data = f.read()
			text.append(data)
			sources.append(urls[i][:-1])
	return text, sources

def get_docs_and_metadata(text,sources,txt_files):
	text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
	docs = []
	metadatas = []
	split_dir = "./split_data"
	for i, d in enumerate(text):
		splits = text_splitter.split_text(d)
		docs.extend(splits)
		metadatas.extend([{"source": sources[i]}] * len(splits))
		with open(os.path.join(split_dir, f"{txt_files[i]}"), "w") as f:
			f.write("\n#######\n".join(splits))
	with open("docs.pkl", "wb") as f:
		pickle.dump(docs, f)
	with open("metadatas.pkl", "wb") as f:
		pickle.dump(metadatas, f)
	return docs, metadatas

current_dir = os.getcwd() #TODO: change this to directory with cleaned .txt files 
data_folder = "./data"
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
#sort txt files based on number in the name 
txt_files.sort(key=lambda f: int(f.split('.')[0]))
text, sources = build_text_and_sources(txt_files)
docs, metadatas = get_docs_and_metadata(text,sources,txt_files)

# print("Documents:{}\n".format(docs[100]))
# print("metadata_files:{}\n".format(metadatas[0]))
# print(len(metadatas))

def create_embeddings(docs):
	batch_size = 100
	embeddings = []
	EMBEDDING_MODEL = "text-embedding-ada-002"
	for batch_start in range(0, len(docs), batch_size):
		batch_end = batch_start + 100
		batch = docs[batch_start:batch_end]
		response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
		for i, be in enumerate(response["data"]):
			assert i == be["index"]  # double check embeddings are in same order as input
		batch_embeddings = [e["embedding"] for e in response["data"]]
		embeddings.extend(batch_embeddings)
	with open("embeddings.pkl", "wb") as f:
		pickle.dump(embeddings, f)

def normalize_embeddings():
	with open("embeddings.pkl", "rb") as f:
		embeddings = pickle.load(f)
	#normalize embeddings using numpy
	embeddings = np.array(embeddings)
	embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
	with open("embeddings_normalized.pkl", "wb") as f:
		pickle.dump(embeddings_normalized, f)	

def create_vector_store():
	# read the faiss_store.pkl file
	with open("embeddings_normalized.pkl", "rb") as f:
		embeddings_normalized = pickle.load(f)
	#save embeddings to faiss vector store
	index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
	index.add(embeddings_normalized)
	faiss.write_index(index, "faiss_store.pkl")

# create_embeddings(docs)
# normalize_embeddings()
create_vector_store()

#save embeddings

# Store the embeddings
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(store, f)
    
# print("Documents:{}\n".format(docs[0]))
# print("metadata_files:{}\n".format(metadatas[0]))
