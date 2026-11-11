import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from unstructured import chunking

load_dotenv()

if __name__ == '__main__':
    loader = UnstructuredLoader(file_path=str(Path(__file__).with_name("mediumblog1.txt")),
                                chunking_strategy="basic",
                                autodetect_encoding=True,
                                max_characters=100000)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Chunks length: {len(texts)=}")
    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    #vectors = embeddings.embed_documents(
    #    [doc.page_content for doc in texts],
    #)
    #print(f"Vector length: {len(vectors)=}")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.index(os.environ.get("PINECONE_INDEX_NAME"))

    CHUNK_FIELD = "chunk_text"

    documents = [
        {
            **doc.metadata,
            "_id":f"medium_blog_{i}",
            CHUNK_FIELD: doc.page_content,
        }
        for i, doc in enumerate(texts)
    ]
    result = index.documents.batch_upsert(
        namespace="blog",
        documents=documents,
    )
    if result.errors:
        raise RuntimeError("\n".join(error.error_message for error in result.errors))


