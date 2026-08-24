import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


#Load Environment Variables
load_dotenv()

##Initialize Groq LLM (Llama 3 70B - high-reasoning accuracy)
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

##Initialize Local Text Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(pdf_path):
    """Loads a resume, chunks the text, and builds an in-memory vector database."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def screen_resume(resume_pdf, job_description):
    """Agentic workflow to retrieve resume data and evaluate fit."""
    retriever = build_vector_store(resume_pdf)
    
    # Define agent instructions & system persona
    system_prompt = (
        "You are an expert HR Screening Agent. Analyze the candidate's resume context provided "
        "and measure their alignment against the Job Description query.\n\n"
        "Provide a structured response:\n"
        "1. MATCH SCORE: (0 to 100)\n"
        "2. CORE STRENGTHS: (Bullet points of matching skills/experience)\n"
        "3. GAPS & MISSING REQUIREMENTS: (Bullet points of missing skills)\n"
        "4. FINAL VERDICT: (Short summary statement evaluating if they should proceed)\n\n"
        "Resume Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": f"Evaluate this resume against this job description: {job_description}"})
    return response["answer"]

# ---Run a local trial---
if __name__ == "__main__":
    # Place a sample resume PDF in your 'resumes' folder and name it 'candidate_sample.pdf'
    sample_resume = "resumes/sample.pdf"
    
    sample_jd = """
    We are looking for a Python Developer with 2+ years of experience. 
    Required skills: FastAPI, PostgreSQL, Git, Docker, and experience working with REST APIs.
    """
    
    if os.path.exists(sample_resume):
        print("Analyzing candidate profile with Groq RAG Agent...")
        evaluation = screen_resume(sample_resume, sample_jd)
        print("\n=== AGENT EVALUATION REPORT ===\n")
        print(evaluation)
    else:
        print(f"Please place a sample PDF file at '{sample_resume}' to run the screening process.")
