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
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


warnings.filterwarnings("ignore")
os.environ["OPENAI_API_KEY"]=constant.APIKEY



loader = TextLoader("data1.txt",autodetect_encoding = True)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=80
)
all_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=4)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.4)

contextualize_q_system_prompt = """該問題依賴於上下文的內容，請精準捕捉上下文關係，\
透過上下文關係，將問題修改為能理解的形式，並補全代詞、模糊語意或上下文資訊，使問題更清楚。\
若問題或對話脫離原先上下文內容，仍有獨立理解該問題的能力。\
若問題超出資料範圍，請仍以模型的理解能力重寫為清楚而完整的獨立問題。\
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """你是一個專業的睡眠治療師，可以從資料中（context）的信息，來準確地提供專業睡眠建議。\
你是根據對話的上下文，來流暢地回答用戶的問題，\
若無法從檢索信息找到合理回答，利用自身的理解與知識生成能力回答，\
回答要清楚扼要且合理，並契合用戶問題。\
當提到更多、其他，請承接前文語意延伸，\
所有回答皆使用繁體中文，除非用戶提及特定人物，不然不要加入任何人名。\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)
query=sys.argv[1]
chat_history = []
ai_msg = rag_chain.invoke({"question": query, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=query), ai_msg])


response=rag_chain.invoke({"question": query, "chat_history": chat_history})

print(response)

