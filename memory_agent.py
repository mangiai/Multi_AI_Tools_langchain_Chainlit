# IMPORTING LIBRARIES

import os
import time
import getpass
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool  # Importing the tool decorator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# LOADING ENVIRONMENTAL VARIABLES

from dotenv import load_dotenv
load_dotenv()

# READING DOCUMENT HELPER FUNCTION

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Load documents from 'PDFs/' directory
doc = read_doc('PDFs/')
print(f"Number of documents loaded: {len(doc)}")

# CHUNKING TECHNIQUE FOR DOCUMENTS

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks  # Corrected return value

# Chunk the documents
documents = chunk_data(docs=doc)
print(f"Number of chunks created: {len(documents)}")

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
    print(f"Creating index '{index_name}'...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"Index '{index_name}' is ready.")

index = pc.Index(index_name)

# SETTING UP OPENAI EMBEDDINGS

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# SETTING UP VECTOR STORE

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print("Vector store initialized.")

# ADDING DOCUMENTS TO VECTOR STORE

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
print("Documents added to the vector store.")

# INITIALIZING TAVILY SEARCH TOOL

tavily_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    name="Tavily_search",
    description="This tool searches the web using Tavily and returns up to 5 results with advanced search depth, including raw content and images."
)

# INITIALIZING YOUTUBE SEARCH TOOL

youtube_search_tool = YouTubeSearchTool(
    name="youtube_search",
    description="This tool searches YouTube for relevant videos based on a query and returns video URLs.",
)

# INITIALIZING CODE INTERPRETER TOOL

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

code_interpreter = ChatGroq(
    name="chat_groq",
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=8000,
    timeout=60,
    max_retries=2,
)

# INITIALIZING CODE INTERPRETER TOOL WITH SHORT DESCRIPTION

code_interpreter_tool = code_interpreter.as_tool(
    arg_types={"input": str},
    description="This tool interprets and executes code snippets provided by the user."
)

# DEFINING TOOLS USING @tool DECORATOR

@tool
def youtube_search(query: str) -> str:
    """Search YouTube for relevant videos based on the query and return video URLs."""
    return youtube_search_tool.invoke(query)

@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily and return up to 5 results with advanced search depth, including raw content and images."""
    return tavily_search_tool.invoke(query)

# @tool
# def code_interpreter_tool_func(input_code: str) -> str:
#     """Interpret and execute code snippets provided by the user."""
#     return code_interpreter_tool.invoke(input_code)

# SETTING UP TOOLS LIST

tools = [
    youtube_search,
    tavily_search,
    code_interpreter_tool_func,
]

# INITIALIZING MEMORY

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)

# DEFINING PROMPT TEMPLATE WITH MEMORY

from langchain_core.prompts import MessagesPlaceholder

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI assistant with access to multiple tools and a vector database.

Use the following context if relevant to the user's query:
{context}

Your task is to:
1. Understand the user's query.
2. If the answer can be found in the conversation history, provide the answer based on memory.
3. If not, use the appropriate tools or the context provided to generate an answer.
4. Synthesize the results and generate a final response.

Now, proceed with the user's query.
""",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

chat_history = []

# INITIALIZING THE AGENT WITH MEMORY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "context": lambda x: x["context"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x.get("intermediate_steps", [])
        ),
        "chat_history": lambda x: x.get("chat_history", []),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# FUNCTION TO RETRIEVE CONTEXT FROM VECTOR STORE

def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant documents from the vector store based on the query."""
    results = vector_store.similarity_search(query, k=top_k)
    if not results:
        return ""
    # Concatenate the content of all retrieved documents
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(results)])
    return context

# EXAMPLE QUERY LOOP

while True:
    query = input("--!!!CHATBOT QUERY SEARCH VIDEOS, CODE COMPLETIONS OR VECTOR DB!!!-- ")

    if query.lower() in {"exit", "quit"}:
        print("Exiting the chatbot. Goodbye!")
        break

    # RETRIEVE CONTEXT FROM VECTOR STORE
    context = retrieve_context(query)
    if context:
        print("Retrieved Context:")
        print(context)
    else:
        print("No relevant context found.")

    print("\nInvoking Agent...\n")

    # Prepare the input with context
    agent_input = {"input": query, "context": context}

    # RUNNING THE AGENT WITH THE RETRIEVED CONTEXT AND USER QUERY
    response = agent_executor.invoke(agent_input)

    # OUTPUT THE RESPONSE
    print("\nAgent Response:")
    print(response)
    print("\n" + "-"*80 + "\n")
