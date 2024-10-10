# IMPORTING LIBRARIES

import os
import time
import getpass
import openai
import langchain
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

# LOADING ENVIRONMENTAL VARIABLES

from dotenv import load_dotenv
load_dotenv()
import os

# READING DOCUMENT HELPER FUNCTION

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Load documents from 'PDFs/' directory
doc = read_doc('PDFs/')
len(doc)

# CHUNKING TECHNIQUE FOR DOCUMENTS

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs

# Chunk the documents
documents = chunk_data(docs=doc)

# SETTING UP PINECONE

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# SETTING UP PINECONE INDEX

index_name = "ytpdfchat"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV")),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# SETTING UP OPENAI EMBEDDINGS

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# SETTING UP VECTOR STORE

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ADDING DOCUMENTS TO VECTOR STORE

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

# INITIALIZING TAVILY SEARCH TOOL

tavirly_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    name="Tavily_search",
    description="This tool searches the web using Tavirly and returns up to 5 results with advanced search depth, including raw content and images."
)

# INITIALIZING YOUTUBE SEARCH TOOL

youtube_search_tool = YouTubeSearchTool(
    name="youtube_search",
    description="This tool searches YouTube for relevant videos based on a query and returns video URLs.",
)

# Example of running the YouTube search tool
response = youtube_search_tool.invoke("lex fridman, 5")
print(response)

# SETTING UP GROQ API KEY

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# INITIALIZING CODE INTERPRETER TOOL

code_interpreter = ChatGroq(
    name="chat_groq",
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=8000,
    timeout=60,
    max_retries=2,
)

# SETTING UP TOOLS
# INITIALIZING CODE INTERPRETER TOOL WITH SHORT DESCRIPTION

code_interpreter_tool = code_interpreter.as_tool(
    arg_types={"input": str},
    description="This tool interprets and executes code snippets provided by the user."
)

# UPDATE THE TOOLS LIST

tools = [
    youtube_search_tool.as_tool(),
    tavirly_search_tool.as_tool(),
    code_interpreter_tool
]




print(tools[0].name)
print(tools[1].name)
print(tools[2].name)

# DEFINING PROMPT TEMPLATE

prompt_template = """
You are an advanced AI agent with access to multiple tools and a vector database. Your task is to:
1. Understand the user's query.
2. Retrieve relevant information from the vector database if needed.
3. Use the appropriate tool to either search the web, find videos, or interpret code based on the context.
4. Synthesize the results and generate a final response that combines the vector store results and tool outputs.

**Steps**:
- If the query involves a specific technology or content search, query the vector database for related documents.
- If the query mentions video, search YouTube using the YouTube Search tool.
- If the query involves finding general information or web content, use the Tavily Search tool.
- If the query involves a technical request such as code interpretation or debugging, use the Code Interpreter tool.

Remember to combine all relevant information in a cohesive response.

**Examples**:
- For a query like "What are the recent advancements in AI?", query the vector database for related documents, then use Tavily Search to find any additional web content.
- For a query like "Show me recent videos about AI ethics," use the YouTube Search tool to find relevant videos.
- For a query like "Debug the following code...", invoke the Code Interpreter tool to assist with the query.

Now, proceed with the user's query.
"""

# INITIALIZING THE AGENT

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    prompt=prompt_template,
    vector_store=vector_store,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

# EXAMPLE QUERY

query = "Get me a Youtube tutorial for learning rust"

# RUNNING THE AGENT WITH THE INITIALIZED PROMPT AND QUERY

response = agent.invoke(query)

# OUTPUT THE RESPONSE

print("Agent Response:")
print(response)
