import os
from operator import itemgetter

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = PromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

print("finish")

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without Langchain expression language
    Limitations:
    - manual step-by-step execution
    - no built-in streaming support
    - no async support without additional code
    - harder to compose with other chains
    - more verbose and error-prone
    """
    # step 1: retrieval relevant documents
    docs = retriever.invoke(query)
    # step 2: format documents into context string
    context = format_docs(docs)
    # step 3: format the prompt with context and question
    messages = prompt_template.format(context=context, question=query)
    # step 4: invoke llm with the formatted message
    response = llm.invoke([messages])
    # step 5: return the content
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL 
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.stream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: More concise and readable
    - Reusable: chain can be saved, shared, and composed with other chains
    - Better debugging: Langchain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") 
            | retriever 
            | format_docs # Langchain converts python functions into RunnableLambda under the hood
        )       # result: combines original + new 
                # {
                #   "question": "...",                  # passed through unchanged
                #   "context": "doc1\ndoc2\ndoc3..."    # newly computed
                # } 
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain




if __name__ == "__main__":
    print("Retrieving...")
    query = "What is Pinecone in machine learning?"

    # ================================================
    # Option 0: Raw invocation without RAG
    # ================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("Answer:")
    print(result_raw.content)

    # ================================================
    # Option 1: Implementation without LCEL
    # ================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query=query)
    print("Answer:")
    print(result_without_lcel)

    # ================================================
    # Option 2: Implementation with LCEL
    # ================================================
    print("\n" + "="*70)
    print("IMPLEMENTATION 2: With LCEL")
    print("="*70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("Answer:")
    print(result_with_lcel)