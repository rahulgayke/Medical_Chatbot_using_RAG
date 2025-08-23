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

@app.route("/")
def index():
    return render_template('chat.html')
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)