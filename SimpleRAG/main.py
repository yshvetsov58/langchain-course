import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, EmbedQuery, SearchDocumentsRequest, SearchDocumentsResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

load_dotenv()
print("Initializing components.....")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.index(os.environ.get("PINECONE_INDEX_NAME"))

llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-5.2")

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(result):
    return "\n\n".join(doc.chunk_text for doc in result.matches)
#========================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
#========================================================
def retrieve_docs_from_vectors(query: str):
    return index.documents.search(
        namespace="blog",
        top_k=10,
        score_by=[
            EmbedQuery(field="chunk_text", query=query)
        ],
        include_fields=["chunk_text"],
    )

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats  them, and generated a response.
    Limitations:
        - Manual step-by-step execution
        - No build-in streaming support
        - Harder to compose with other chains
        - More verbose and error-prone
    """
    docs = retrieve_docs_from_vectors(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content
#========================================================
# IMPLEMENTATION 2: With LCEL (Simple Function-Based Approach)
#========================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Express Language )
    Returns a chain that can be invoked with {"question":"....."}
    Advantages over non-lcel approach:
    1) Declarative and composable
    2) Build-in streaming: chain.stream() works out of the box
    3) Build-in async: chain.ainvoke() and chain.asteam() available
    4) Batch processing chain.batch() for multiple inputs
    5) Type safety: Better integration with LangChain type system
    6) Less code: More concise and readable
    7) Reusable
    8) bBetter debugging
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=(
                    RunnableLambda(
                        lambda data: data["question"],
                        name="ExtractQuestion",
                    )
                    | RunnableLambda(
                retrieve_docs_from_vectors,
                name="SearchPinecone",
            )
                    | RunnableLambda(
                format_docs,
                name="FormatContext",
            )
            )
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving.....")
    query = "What is the Pinecone in machine learning?"
  #  print(retrieval_chain_without_lcel(query))
    input_dict = {"question": "What is the Pinecone in machine learning?"}
    print(create_retrieval_chain_with_lcel().invoke(input_dict))
