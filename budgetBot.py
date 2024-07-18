# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from collections import OrderedDict

# Path to the FAISS vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template for the QA system
# This template structures the input for the language model
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Creates and returns a PromptTemplate object with the custom template.
    This function allows easy modification of the prompt structure if needed.
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    """
    Creates and returns a RetrievalQA chain.
    This chain combines document retrieval with question answering.
    
    :param llm: The language model to use for question answering
    :param prompt: The prompt template to structure input for the language model
    :param db: The vector database for document retrieval
    :return: A RetrievalQA chain object
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # 'stuff' chain type passes all retrieved documents to the LLM
        retriever=db.as_retriever(search_kwargs={'k': 2}),  # Retrieve top 2 most relevant documents
        return_source_documents=True,  # Include source documents in the output
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    """
    Loads and returns the language model (Llama 2 in this case).
    Adjust model parameters here for performance tuning.
    """
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",  # Path to the model file
        model_type="llama",
        max_new_tokens = 512,  # Maximum number of tokens in the generated response
        temperature = 0.5  # Controls randomness: lower values make output more deterministic
    )
    return llm

def qa_bot():
    """
    Sets up and returns the complete QA system.
    This function initializes embeddings, loads the vector database,
    sets up the language model, and creates the QA chain.
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the vector database
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load the language model
    llm = load_llm()
    
    # Set up the prompt template
    qa_prompt = set_custom_prompt()
    
    # Create and return the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Chainlit event handler for chat initiation
@cl.on_chat_start
async def start():
    """
    Initializes the chat session.
    This function is called when a new chat session starts.
    """
    # Create the QA chain
    chain = qa_bot()
    
    # Send an initial message
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Budget Bot. What is your query?"
    await msg.update()
    
    # Store the chain and initialize a cache in the user session
    cl.user_session.set("chain", chain)
    cl.user_session.set("cache", OrderedDict())

# Chainlit event handler for incoming messages
@cl.on_message
async def main(message: cl.Message):
    """
    Processes incoming user messages and generates responses.
    This function is called for each message sent by the user.
    """
    # Retrieve the QA chain and cache from the user session
    chain = cl.user_session.get("chain")
    cache = cl.user_session.get("cache")
    
    # Check if the question is in the cache
    if message.content in cache:
        await cl.Message(content=f"[Cached] {cache[message.content]}").send()
        # Move this item to the end to mark it as most recently used
        cache.move_to_end(message.content)
        return

    # Set up the callback handler for streaming the response
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    
    # Generate the response
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    # Add source information to the answer
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    
    # Add the new question and answer to the cache
    cache[message.content] = answer
    # If cache size exceeds 5, remove the least recently used item
    if len(cache) > 5:
        cache.popitem(last=False)
    
    # Send the response
    await cl.Message(content=answer).send()
