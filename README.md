# Resume Analyzer AI

An AI-powered Resume Screening and Analysis system that uses RAG (Retrieval-Augmented Generation) to evaluate a candidate's resume against a given Job Description.
The system extracts information from a PDF resume, retrieves relevant sections, and uses an LLM to identify strengths, skill gaps, and overall alignment.

## Features
- Upload and process PDF resumes
- Retrieve relevant resume information using RAG
- AI-powered resume and job description comparison
- Generate a match score
- Identify candidate strengths
- Highlight missing skills and requirements
- Provide a final screening verdict

## Architecture
```
Resume PDF
    │
PyPDFLoader
    │
Text Splitting
    │
HuggingFace Embeddings
    │
Chroma Vector Store
    │
Retriever
    │
Relevant Resume Context
    │
Groq LLM
    │
Resume Evaluation Report
```
## Tech Stack
- Python
- LangChain
- Groq
- GPT-OSS Model
- HuggingFace Embeddings
- ChromaDB
- PyPDF

## Example Output
<img width="80%" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/1c4e407e-916e-4973-927b-d4db35da0ec2" />
