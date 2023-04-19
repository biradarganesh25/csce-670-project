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
	with open('links.txt', 'r') as f:
		urls = f.readlines()
	for i,txt_file in enumerate(txt_files):
		with open(txt_file, encoding="utf8") as f:
			data = f.read()
			text.append(data[1:])
			sources.append(urls[i])
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
	return docs, metadatas


current_dir = os.getcwd() #TODO: change this to directory with cleaned .txt files 
data_folder = "./data"
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
text, sources = build_text_and_sources(txt_files)
docs, metadatas = get_docs_and_metadata(text,sources,txt_files)

print("Documents:{}\n".format(docs[100]))
# print("metadata_files:{}\n".format(metadatas[0]))
print(len(metadatas))


# Store the embeddings
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(store, f)
    
# print("Documents:{}\n".format(docs[0]))
# print("metadata_files:{}\n".format(metadatas[0]))
