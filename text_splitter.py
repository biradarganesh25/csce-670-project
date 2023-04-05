import os
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

def build_text_and_sources(txt_files):
	text = []
	sources = []
	for txt_file in txt_files:
		with open(txt_file, encoding="utf8") as f:
			text.append(f.read())
			sources.append(txt_file)
	return text, sources

def get_docs_and_metadata(text,sources):
	text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
	docs = []
	metadatas = []
	for i, d in enumerate(text):
		splits = text_splitter.split_text(d)
		docs.extend(splits)
		metadatas.extend([{"source": sources[i]}] * len(splits))
	return docs, metadatas


current_dir = os.getcwd() #TODO: change this to directory with cleaned .txt files 
txt_files = [f for f in os.listdir(current_dir) if f.endswith('.txt')]
text, sources = build_text_and_sources(txt_files)
docs, metadatas = get_docs_and_metadata(text,sources)

print("Documents:{}\n".format(docs[0]))
# print("metadata_files:{}\n".format(metadatas[0]))
print(len(metadatas))


# Store the embeddings
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)