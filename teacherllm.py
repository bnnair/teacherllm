"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""
import savetodb
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# prepare the template we will use when prompting the AI
# template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

template = """Use the following pieces of information and do what the user says.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


# load the language model
llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0.01})


# load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

savetodb.saveToDB()
db = FAISS.load_local("faiss", embeddings,
                      allow_dangerous_deserialization=True)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# ask the AI chat about information in our local files
# prompt = "Who is the author of FftSharp? What is their favorite color?"
prompt = "Generate 2 new questions based on the pdf along with their answers. make sure \
    that the questions should be in line with the worksheet data."
output = qa_llm.invoke({'query': prompt})
print(output["result"])
