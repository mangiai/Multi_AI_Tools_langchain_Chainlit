
# PDF Chatbot with Pinecone and OpenAI

This project demonstrates how to build a PDF chatbot using LangChain, Pinecone for vector search, OpenAI embeddings, and various tools like Tavily Search and YouTube Search. The chatbot can read PDF files, chunk the content, add it to a vector store, and assist with answering queries using the stored data.


## Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- Pinecone API
- OpenAI API
- Groq API
- Tavily Search API
- YouTube Data API


## Run Locally

Clone the project

```bash
  git clone https://github.com/mangiai/Multi_AI_Tools_langchain_Chainlit.git
```

Go to the project directory

```bash
  python -m venv my env
  myenv/Scripts/Activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the Chainlit Application

```bash
  chainlit run chainlit_agent.py
```

