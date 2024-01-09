__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv  import load_dotenv
load_dotenv()
import google.generativeai as genai
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv  import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCnfxeSjZ4YzCUM8u5jNsdrJGcmhBIwIZo"
genai.configure(api_key="AIzaSyCnfxeSjZ4YzCUM8u5jNsdrJGcmhBIwIZo")
model=genai.GenerativeModel(model_name="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_index = None
app = FastAPI()
chat =None
class Item(BaseModel):
    filename: str
    filetype: str
    filesize: int 
    data: str

class question(BaseModel):
    querie:str

@app.post("/qna/")
async def upload_file(item: question):
    """
    Send a question to the chatbot.

    Args:
        item (Question): The question to be sent to the chatbot.

    Returns:
        dict: A dictionary containing the response from the chatbot.
    """
    global vector_index  # Use the global variable
    if vector_index is None:
        return {"error": "Vector index not initialized"}
    
    docs = vector_index.get_relevant_documents(item.querie)
    for doc in docs:
        response = chat.send_message(doc.to_json()['kwargs']['page_content'])
    response = chat.send_message(item.querie)  
        
    return {"response": response.text}
    

@app.post("/upload/")
async def upload_file(item: Item):
    global vector_index 
    global chat
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text= item.data
    texts = text_splitter.split_text(text)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
    chat = model.start_chat(history=[])
    return {"response": "done"}

@app.get("/")
async def root():
    return {"message": "Hello World"}
#uvicorn back:app --reload  
#docker compose up -d --build  --scale server=1      
 