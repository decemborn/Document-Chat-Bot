from fastapi import FastAPI,UploadFile,File,Form,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger
import os

load_dotenv()

app=FastAPI(title="RagBot2.0")

# allow frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500,content={"error":str(exc)})
    
@app.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"recieved {len(files)} files")
        load_vectorstore(files)
        logger.info("documents added to ChromaDB")
        return {"message":"Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500,content={"error":str(e)})


# @app.post("/ask/")
# async def ask_quyestion(question:str=Form(...)):
#     try:
#         logger.info("fuser query:{question}")
#         from langchain.vectorstores import Chroma
#         from langchain.embeddings import HuggingFaceBgeEmbeddings
#         from modules.load_vectorstore import PERSIST_DIR

#         vectorstore=Chroma(
#             persist_directory=PERSIST_DIR,
#             embedding_function=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L12-v2")
#         )
#         chain=get_llm_chain(vectorstore)
#         result=query_chain(chain,question)
#         logger.info("query successfull")
#         return result
#     except Exception as e:
#         logger.exception("error processing question")
#         return JSONResponse(status_code=500,content={"error":str(e)})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from modules.llm import get_llm_chain
        from modules.query_handlers import query_chain

        PERSIST_DIR = "./chroma_store"
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Load Chroma
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM + RetrievalQA chain
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}