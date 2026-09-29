import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

print("Loading PDF document...")

loader = PyPDFLoader("TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()


# --------------------------------------------------
# 2. Split document into chunks
# --------------------------------------------------

print("Chunking text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(document)


# --------------------------------------------------
# 3. Create embeddings and vector database
# --------------------------------------------------

print("Creating vector database...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# 5. Create prompt
# --------------------------------------------------

template = """
Use the following pieces of retrieved context to answer the question.

If you don't know the answer, just say that you don't know.

Use three sentences maximum and keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)


# --------------------------------------------------
# 6. Initialize Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# --------------------------------------------------
# 7. Format retrieved documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# --------------------------------------------------
# 8. Create RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# --------------------------------------------------
# 9. Start chatbot
# --------------------------------------------------

print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop.")


while True:

    user_question = input("\nYour Question: ")

    if user_question.lower() in ["exit", "quit"]:
        print("Shutting down chatbot. Goodbye!")
        break

    try:

        response = rag_chain.invoke(user_question)

        if isinstance(response.content, list):

            clean_answer = ""

            for block in response.content:
                if isinstance(block, dict):
                    if "text" in block:
                        clean_answer += block["text"]

        else:
            clean_answer = response.content

        print(f"\nAnswer: {clean_answer}")

    except Exception as e:

        print("\nError occurred:")
        print(e)