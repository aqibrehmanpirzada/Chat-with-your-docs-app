# import streamlit as st
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import os

# # Load environment variables. Assumes that project contains .env file with API keys
# load_dotenv()
# # Set OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def main(query_text):
#     # Prepare the DB.
#     embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)
#     if len(results) == 0 or results[0][1] < 0.7:
#         st.warning("Unable to find matching results.")
#         return

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     model = ChatOpenAI(api_key=openai_api_key)
#     response = model.invoke(prompt)

#     return response.content

# def run_streamlit_app():
#     st.title("Interactive Query Application")

#     query_text = st.text_input("Enter your query:", "")

#     if st.button("Submit"):
#         if query_text:
#             st.subheader("Query Results:")
#             result = main(query_text)
#             st.write(result)

# if __name__ == "__main__":
#     run_streamlit_app()

#--------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main(query_text):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        st.warning("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(api_key=openai_api_key)
    response = model.invoke(prompt)

    return response.content

def run_streamlit_app():
    st.title("Interactive Query Application")
    st.markdown("---")

    query_text = st.text_input("Enter your query:", "")
    show_results = st.button("Show Results")

    if show_results and query_text:
        st.info("Processing your query...")
        result = main(query_text)

        st.subheader("Query Results:")
        st.write(result)

    # Add a section for additional functionalities or explanations
    st.markdown("---")
    st.subheader("Additional Information")
    st.write("This application uses AI to retrieve relevant information based on your query.")
    st.write("Please enter a query and click 'Show Results' to see the response.")

if __name__ == "__main__":
    run_streamlit_app()


