from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to include all relevant details. 
    If the answer is not in the provided context, follow these steps:  
    1. Acknowledge that the answer is not in the provided context.  
    2. Provide any relevant general information related to the question if possible.  
    3. Suggest similar topics or related questions that might be helpful.  
    4. Encourage the user to ask another question if they need further assistance.  

    Context:\n {context}?\n

    Question:\n {question}\n

    Answer:

    If the answer is not in the provided context:
    - "The answer is not in the provided context. However, here is some general information that may help:"
    - "give some information from the browser or any source"
    - "You might also find these topics useful: "
    - " Give some topics similar to the topics in which the question is asked"
    - "If you need further assistance, please leave your question, and I'll be happy to help!"
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('pdf_files')
    pdf_paths = []
    for file in files:
        path = os.path.join('uploads', file.filename)
        file.save(path)
        pdf_paths.append(path)

    try:
        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return jsonify({"message": "PDF processed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question')
    try:
        response_text = user_input(user_question)
        return jsonify({"answer": response_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True,use_reloader=False)
