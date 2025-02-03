import warnings
import openai
import os
import sys
import constant
import openai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


warnings.filterwarnings("ignore")
os.environ["OPENAI_API_KEY"]=constant.APIKEY



loader = TextLoader("data.txt",autodetect_encoding = True)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
all_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.4)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query=sys.argv[1]

response = rag_chain.invoke(query)

if response == "我不知道。":
    response_1 =  openai.chat.completions.create(model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": query,
        },
    ],stream=True,
)
    for chunk in response_1:
        print(chunk.choices[0].delta.content or "", end="")
else:
    for chunk in rag_chain.stream(query):
        print (chunk, end="", flush=True)


    
