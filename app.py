from flask import Flask, render_template, jsonify, request
from src.helper import get_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Get the embedding model
embeddings = get_embeddings()

index_name = "medical-chatbot-knowledge-base"
# Store all the text chunks in the Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

# Retriever for similarity search and retrieving required docs
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

# Load the HuggingFace model for text generation
llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task = "test-generation"
)
chat_model = ChatHuggingFace(llm = llm)

# Prompt tremplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

# Chains
qa_Chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_Chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)