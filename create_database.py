# # from langchain.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# # from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# import openai 
# from dotenv import load_dotenv
# import os
# import shutil

# # Load environment variables. Assumes that project contains .env file with API keys
# load_dotenv()
# #---- Set OpenAI API key 
# # Change environment variable name from "OPENAI_API_KEY" to the name given in 
# # your .env file.
# openai.api_key = os.environ['OPENAI_API_KEY']

# CHROMA_PATH = "chroma"
# DATA_PATH = "data/books"


# def main():
#     generate_data_store()


# def generate_data_store():
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)


# def load_documents():
#     loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
#     documents = loader.load()
#     return documents


# def split_text(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

#     document = chunks[10]
#     print(document.page_content)
#     print(document.metadata)

#     return chunks


# def save_to_chroma(chunks: list[Document]):
#     # Clear out the database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# if __name__ == "__main__":
#     main()
import argparse
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found.")
        return

    chunks = split_text(documents)
    if not chunks:
        print("No chunks created.")
        return

    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, loader_cls=PyPDFLoader, glob="*.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increase chunk size
        chunk_overlap=150,  # Increase chunk overlap
        length_function=len,
        add_start_index=True,
    )
    
    chunks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_document = {executor.submit(text_splitter.split_documents, [doc]): doc for doc in documents}
        for future in as_completed(future_to_document):
            try:
                doc_chunks = future.result()
                chunks.extend(doc_chunks)
            except Exception as exc:
                print(f"Document {future_to_document[future].metadata['source']} generated an exception: {exc}")

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if chunks:
        document = chunks[0]
        print(document.page_content)
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main_query():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=20)  # Increase k for more results
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(api_key=openai_api_key)
    response = model.invoke(prompt)

    # Extract detailed content from the response
    detailed_response = ""
    for doc, _score in results:
        detailed_response += f"Chapter {doc.metadata['chapter_number']}: {doc.page_content}\n"

    print(detailed_response)

if __name__ == "__main__":
    main()
    # To query the data, run: main_query()
