import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
# from pypdf import PdfReader
import pypdf
from PyPDF2 import PdfReader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"


def uploaded_data(uploaded_file, title=None):
    if uploaded_file is not None:
        docs = []
        fname = uploaded_file.name
        if not title:
            title = os.path.basename(fname)
        if fname.lower().endswith('pdf'):
            pdf_reader = PdfReader(uploaded_file)
            # loader = PyPDFLoader(uploaded_file)
            # docs.extend(loader.load())
            for num, page in enumerate(pdf_reader.pages):
                page = page.extract_text()
                doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
                docs.append(doc)

        else:
            # assume text
            doc_text = uploaded_file.read().decode()
            docs.append(doc_text)

        return docs
    else:
        return None

def load_web_page(urls):
    if len(urls) > 0:
        docs = [WebBaseLoader(url).load() for url in urls]
        print("docs",docs)
        url_docs_list = [item for sublist in docs for item in sublist]
        return url_docs_list
    else: 
        return None

st.set_page_config(page_title="Adaptive RAG ChatBot", page_icon=":robot:", layout="wide")

curr_state = []
urls = []
docs_list = []
################################FRONT-END####################################

st.title("Adaptive RAG ChatBot")

st.subheader("Ask questions about your documents and get answers from the RAG model.")

#subheader
# st.subheader('LLM Question-Answering Application')

#sidebar creator
with st.sidebar:

    #uploads file
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

    #Add URL
    url_input = st.text_input("Add a URL")
    if len(url_input)>0:
        urls.append(url_input)
    #add URL button
    if st.button("Add URL"):

        # urls.append(url_input)
        # print('urls',urls)
        st.write(f"Added URL: {url_input}")

    #number input widget
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512) #, on_change=clear_history

    #variable k input
    k = st.number_input('k', min_value=1, max_value=20, value=5) #, on_change=clear_history

    #add file button
    add_data = st.button('Add Data')

    add_vertical_space(5)
    st.write('Made with ❤️ by Mohamed Farrag')


    if add_data: # if the user browsed a file 

        with st.spinner ('Reading, chunking and embedding ...'):
            

            # if len(urls) > 0:
            #     url_docs_list = load_web_page(urls)
                
            # if uploaded_file is not None:
            #     docs = uploaded_data(uploaded_file)
            print("urls2",urls)
            url_docs_list = load_web_page(urls)
            docs = uploaded_data(uploaded_file)

            
            if url_docs_list is not None and docs is not None:
                docs_list = docs + url_docs_list
            elif url_docs_list is not None and docs is None:
                docs_list = url_docs_list
            elif url_docs_list is None and docs is not None:
                docs_list = docs
            else:
                docs_list = []

            print('docs_list',docs_list)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(docs_list)
            print('doc_splits',doc_splits)

            # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            # chunked_documents = text_splitter.split_documents(docs_list)
            # print('chunked_documents',chunked_documents)


            # Embedding arguments
            model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
            gpt4all_kwargs = {'allow_download': 'True'}

            # Add to vectorDB
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding = GPT4AllEmbeddings(
                                        model_name=model_name,
                                        gpt4all_kwargs=gpt4all_kwargs
                                                                    ),
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})




# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=chunk_size, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)


# # Embedding arguments
# model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
# gpt4all_kwargs = {'allow_download': 'True'}

# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding = GPT4AllEmbeddings(
#                             model_name=model_name,
#                             gpt4all_kwargs=gpt4all_kwargs
#                                                         ),
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": k})

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

prompt_retrieval_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt_retrieval_grader | llm | JsonOutputParser()


prompt_Generate = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)


# Chain
rag_chain = prompt_Generate | llm | StrOutputParser()

prompt_hallucination_grader = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt_hallucination_grader | llm | JsonOutputParser()

# Prompt
prompt_answer_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt_answer_grader | llm | JsonOutputParser()

prompt_question_router = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt_question_router | llm | JsonOutputParser()


# web Search
web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


### Nodes


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # st.write("---RETRIEVE---")
    curr_state.append("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    # st.write("---GENERATE---")
    curr_state.append("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    # st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    curr_state.append("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            # st.write("---GRADE: DOCUMENT RELEVANT---")
            curr_state.append("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            # st.write("---GRADE: DOCUMENT NOT RELEVANT---")
            curr_state.append("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    # st.write("---WEB SEARCH---")
    curr_state.append("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


### Conditional edge


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    # st.write("---ROUTE QUESTION---")
    curr_state.append("---ROUTE QUESTION---")
    question = state["question"]
    # st.write(question)
    curr_state.append(question)
    source = question_router.invoke({"question": question})
    # st.write(source)
    # curr_state.append(source)
    # st.write(source["datasource"])
    datasource = source["datasource"]
    curr_state.append(f"The data source is the {datasource}")
    if source["datasource"] == "web_search":
        # st.write("---ROUTE QUESTION TO WEB SEARCH---")
        curr_state.append("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        # st.write("---ROUTE QUESTION TO RAG---")
        curr_state.append("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # st.write("---ASSESS GRADED DOCUMENTS---")
    curr_state.append("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # st.write(
        #     "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        # )
        curr_state.append(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        # st.write("---DECISION: GENERATE---")
        curr_state.append("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    # st.write("---CHECK HALLUCINATIONS---")
    curr_state.append("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        # st.write("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        curr_state.append("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        # st.write("---GRADE GENERATION vs QUESTION---")
        curr_state.append("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            # st.write("---DECISION: GENERATION ADDRESSES QUESTION---")
            curr_state.append("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            # st.write("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            curr_state.append("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        # st.write("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        curr_state.append("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile
app = workflow.compile()

q = st.chat_input("Say something")
inputs = {"question": q}

if q:
    for output in app.stream(inputs):
        for key, value in output.items():
            for i in range(len(curr_state)):
                
                st.write(curr_state[i])
            
            curr_state.clear()
            st.write(f"Finished running: {key}:")
    st.write(value["generation"])