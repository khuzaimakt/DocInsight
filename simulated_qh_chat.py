import os 

from qh_chat import UploadFile, UploadFileType

import re
from typing import List

from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

import nltk
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

import pdfplumber
import llm_blender

import pdfplumber

nltk.download("punkt")

SYSTEM_PROMPT = SystemMessage(
                #"For testing purposes, " + 
                # "start all responses with the word " +
                # "'Camembert.' Then respond to the query.\n\n" +
                "You are an assistant to hospital staff, " + 
                "including doctors, nurses, and administrative staff. " +
                "You will receive queries from the hospital staff. \n\n" +
                "Sometimes these queries will contain attached context, " +
                "which may be presented to you as a file or as a " + 
                "section of text input. " + 
                "If given a context, your response should be based on " + 
                "the information in that context. " + 
                "Nonetheless, avoid using phrases like " + 
                "'Based on the given context'. \n\n" +
                "Your tone should match that of a knowledgeable colleague " +
                "in a hospital setting. If asked a clinical question, use " +
                "standard medical terminology in your response,  " +
                "as a knowledgeable doctor or nurse would. " + 
                "If asked a clinical question, assume the person " + 
                "asking knows standard medical acronyms, as a nurse " + 
                "or doctor would. " + 
                "If asked an administrative question, use standard " + 
                "administrative and financial terms, as a hospital " +
                "administrator would.\n\n" + 
                "Your responses should be concise and contain as little " +
                "extraneous detail as possible, while still " +
                "addressing the query. " + 
                "Avoid filler phrases that do not convey information.")
HUMAN_NOCONTEXT_PROMPT = HumanMessagePromptTemplate.from_template(
    "Answer this question: {question}")
HUMAN_CONTEXT_PROMPT = HumanMessagePromptTemplate.from_template(
    "Answer this question: {question}\n\nBased on this data:\n\n{context}")

class TextSplitter:
    @staticmethod
    def split_file(documents: List[Document], chunk_size: int, use_sentence_split: bool) -> List[Document]:
        """Split a file into chunks."""
        if use_sentence_split:
            return TextSplitter.split_file_into_sentences(documents, chunk_size)
        else:
            return TextSplitter.split_file_recursive(documents, chunk_size)
    
    @staticmethod
    def split_file_recursive(documents, chunk_size):
        """Split a file into chunks using recursive chunking."""
        chunks = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        return chunks

    @staticmethod
    def split_file_into_sentences(documents, chunk_size):
        """Split a file into sentences. Recursive chunking is sentences longer than `chunk_size`."""
        sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size//8)
    
        chunks = []
        for doc in documents:
            sentences = [Document(page_content=s, metadata={"source": "local"})
                         for s in sent_detector.tokenize(doc.page_content)]
            for s in sentences:
                if len(s.page_content) > chunk_size:
                    sentence_chunks = recursive_splitter.split_documents([s])
                else:
                    sentence_chunks = [s]

                chunks.extend(sentence_chunks)
    
        return chunks
    
class SimulatedQualifiedHealthTextLoader:
    """Simulated a text file loader. This class is responsible for loading a text file and chunking it.
    
    Args:
        - filepath: The path to the text file.
        - chunk_size: The size of each chunk. If `use_sentence_split` is set to True, this will be a maximum size.
        - use_sentence_split: Whether to split the text file into sentences rather than recursive chunking.
        - bypass_chunking: Whether to bypass chunking altogether. This is useful for calculating baselines."""
    def __init__(self, filepath, chunk_size=512, use_sentence_split=False, bypass_chunking=False):
        with open(filepath, "r",encoding='utf-8') as infile:
            self.text = infile.read()
        
        if bypass_chunking:
            self.chunks = [Document(page_content=self.text, metadata={"source": "local"})]
        else:        
            file_document = Document(page_content=self.text, metadata={"source": "local"})
            self.documents = [file_document]

            self.chunks = TextSplitter.split_file(self.documents, chunk_size, use_sentence_split)
    
class SimulatedQualifiedHealthPDFLoader:
    def __init__(self, filepath, chunk_size=512, use_sentence_split=False, bypass_chunking=False):
        documents = []
        self.chunks = []

        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()

                if len(text) > 0:
                    cleaned_text = SimulatedQualifiedHealthPDFLoader._clean_text(text)
                    metadata = {"source": os.path.basename(filepath), "page": int(i + 1)}
                    document = Document(page_content=cleaned_text, metadata=metadata)
                    documents.append(document)

        if bypass_chunking:
            text = "\n\n".join([doc.page_content for doc in documents])
            self.chunks = [Document(page_content=text, metadata={"source": os.path.basename(filepath)})]
        elif len(documents) > 0:
            self.chunks = TextSplitter.split_file(documents, 
                                                  chunk_size=chunk_size, 
                                                  use_sentence_split=use_sentence_split)

    @staticmethod
    def _clean_text(text):
        cleaned_text = re.sub(r'(\d)\1{9,}|-{10,}', '', text, flags=re.MULTILINE)
        cleaned_text = re.sub(
            r'[^\x00-\x7F\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u0E00-\u0E7F\u1E00-\u1EFF\u1F00-\u1FFF\u2000-\u206F\u2100-\u214F\u2C00-\u2C5F\u2D00-\u2D2F\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uA000-\uA48C\uA4D0-\uA4FF\uA500-\uA62B\uA640-\uA69F\uA700-\uA7FF\uA800-\uA82F\uA840-\uA87F\uA880-\uA8C5\uA8E0-\uA8FF\uA900-\uA92F\uA930-\uA95F\uAA00-\uAA5F\uAB00-\uAB2F\uFB00-\uFBFF\uFE70-\uFEFF\u0100-\u017F\u0180-\u024F\u0370-\u03FF\u1F00-\u1FFF\u10A0-\u10FF\u0530-\u058F]+',
            '', cleaned_text)

        cleaned_text = re.sub(r'^([!?"#$%&\'()\d\s*?.;\-,+=/@<>:\n^]{3,})\s*', '', cleaned_text, flags=re.MULTILINE)

        # remove spaces
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

        cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n'))

        return cleaned_text



class SimulatedQualifiedHealthTableLoader:
        def __init__(self, filepath, chunk_size=512, bypass_chunking=False, use_sentence_split=False):
            if filepath.endswith("csv"):
                df = pd.read_csv(filepath)
            elif filepath.endswith("xlsx"):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

            self.documents = [
                Document(page_content=df.to_string(), metadata={"source": os.path.basename(filepath)})
            ]
            self.chunks = []

            if bypass_chunking:
                self.chunks = self.documents
            else:
                self.chunks = TextSplitter.split_file(self.documents, 
                                                    chunk_size=chunk_size, 
                                                    use_sentence_split=use_sentence_split)



class SimulatedQualifiedHealthImageLoader:
    def __init__(self, filepath, chunk_size=512, bypass_chunking=False, use_sentence_split=False, **kwargs):
        image = Image.open(str(filepath))
        text = pytesseract.image_to_string(image)
        self.documents = [
            Document(page_content=text, metadata={"source": os.path.basename(filepath)})
        ]
        self.chunks = []

        if bypass_chunking:
            self.chunks = self.documents
        else:
            self.chunks = TextSplitter.split_file(self.documents, 
                                                  chunk_size=chunk_size, 
                                                  use_sentence_split=use_sentence_split)



class SimulatedQualifiedHealthRetriever:
    """Simulates the retrieval component of RAG. This class assumes that chunks of text are the basis for retrieval."""
    def __init__(self, chunks, embedding_model, top_k=5,use_reranking=False):
        """Initialize the retriever.
        
        Args:
            - chunks: A list of `Document` objects representing chunks.
            - embedding_model: A LangChain embedding model.
            - top_k: The number of top retrievals to return."""
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.top_k  = top_k
        self.use_reranker= use_reranking

        if type(embedding_model) == HuggingFaceBgeEmbeddings:
            """The BGE model doesn't seem to have an `embed_document` method and this breaks the normal workflow."""
            self.chunk_embeddings = np.array([embedding_model.embed_query(_.page_content) for _ in self.chunks])
        else:
            """Embed all the chunks."""
            self.chunk_embeddings = np.array(embedding_model.embed_documents([_.page_content for _ in self.chunks]))

        if self.use_reranker== True:
            self.blender = llm_blender.Blender()
            self.blender.loadranker("llm-blender/PairRM",device='cpu') 


        
    
    def retrieve(self, query):
        """Retrieve the top `top_k` chunks based on the query."""

        # Embed the query.
        query_embedding = self.embedding_model.embed_query(query)

        # Calculate the similarities.
        similarities = np.dot(self.chunk_embeddings, query_embedding)

        top_k = min(self.top_k, len(self.chunks))
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        if self.use_reranker == False:
            return [self.chunks[i] for i in top_k_indices]
        
        elif self.use_reranker==True:
        
            retrieved_top_k= [self.chunks[i] for i in top_k_indices]
            retrieved_top_k_content= [chunk.page_content for chunk in retrieved_top_k]
            ranks = self.blender.rank([query], [retrieved_top_k_content], return_scores=False, batch_size=1)
            rank=ranks[0]
            reranked_top_k= [retrieved_top_k_content[r-1] for r in rank]

            return reranked_top_k
    


class SimulatedQualifiedHealthChat:
    """Simulation of the QualifiedHealth platform chat.
    
    Args:
        - upload_file: An `UploadFile` object representing the file to be uploaded.
        - bypass_rag: Whether to bypass the RAG retrieval component. Set to True to calculate
            baselines.
        - use_azure_endpoint: Whether to use the Azure endpoint for the OpenAI model. Set to False to calculate
            baselines. Set to True for greater fidelity to real-world usage."""
    def __init__(
            self,
            upload_file=None,
            bypass_rag=False,
            chunk_tables=False,
            use_azure_endpoint=False,
            use_reranker=False,
            system_prompt=SYSTEM_PROMPT,
            use_sentence_chunking=False,
            max_chunk_size=512,
            num_retrieved_chunks=5,
            embedding_model=None,  # a shared embedding model, if not provided, a default one will be created
            **kwargs,
        ):
        self.using_file = upload_file is not None
        self.bypass_rag = bypass_rag
        self.system_prompt = system_prompt
        self.embedding_model = embedding_model
        self.reranker= use_reranker

        

        if self.using_file and upload_file.filetype == UploadFileType.TEXT:
            """Load the text file."""
            self.loader = SimulatedQualifiedHealthTextLoader(upload_file.filepath, 
                                                             chunk_size=max_chunk_size,
                                                             bypass_chunking=self.bypass_rag,
                                                             use_sentence_split=use_sentence_chunking)
            self.chunks = self.loader.chunks
        elif self.using_file and upload_file.filetype == UploadFileType.PDF:
            self.loader = SimulatedQualifiedHealthPDFLoader(upload_file.filepath, 
                                                            chunk_size=max_chunk_size,
                                                            bypass_chunking=self.bypass_rag,
                                                            use_sentence_split=use_sentence_chunking)
            self.chunks = self.loader.chunks
        elif self.using_file and upload_file.filetype == UploadFileType.TABLE:
            self.loader = SimulatedQualifiedHealthTableLoader(upload_file.filepath, 
                                                              chunk_size=max_chunk_size,
                                                              bypass_chunking=self.bypass_rag or not chunk_tables)
            self.chunks = self.loader.chunks
        elif self.using_file and upload_file.filetype == UploadFileType.IMAGE:
            self.loader = SimulatedQualifiedHealthImageLoader(
                upload_file.filepath,
                chunk_size=max_chunk_size,
                bypass_chunking=self.bypass_rag,
                use_sentence_split=use_sentence_chunking
            )
            self.chunks = self.loader.chunks
        elif self.using_file: 
            """File format not supported."""
            raise Exception(f"Invalid file type {upload_file.filetype}. Please use a text file.")

        if not self.bypass_rag:
            # Create embedding model.
            if not self.embedding_model:
                self.embedding_model = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True})

            # a lighter embedder for dev purpose.
            # self.embedding_model = HuggingFaceEmbeddings(
            #     model_name = "sentence-transformers/all-MiniLM-L6-v2",
            #     model_kwargs = {'device': 'cpu'},
            #     encode_kwargs = {'normalize_embeddings': False},
            # )

            # Create retreiver.
            self.retriever = SimulatedQualifiedHealthRetriever(self.chunks, self.embedding_model,
                                                               top_k=num_retrieved_chunks,use_reranking=self.reranker)
       
        if use_azure_endpoint:
            self.llm = AzureChatOpenAI(
                openai_api_version="2024-05-01-preview",
                openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT"])
        else:
            # Use OpenAI endpoint.
            self.llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"])

        self.query_history = []
        self.response_history = []

    def send_message(self, message):
        """Send message to the system.
        
        Args:
            - message: The message to send to the system.
        """

        human_prompt = HUMAN_CONTEXT_PROMPT if self.using_file else HUMAN_NOCONTEXT_PROMPT
        prompt_template = (ChatPromptTemplate.from_messages([self.system_prompt, human_prompt]) 
                           if self.system_prompt is not None 
                           else ChatPromptTemplate.from_messages([human_prompt]))

        if self.using_file:
            retrieved_chunks = self.chunks if self.bypass_rag else self.retriever.retrieve(message)
            if self.reranker == True:
                context = "\n\n".join([chunk for chunk in retrieved_chunks])

            elif self.reranker==False:
                context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])


            prompt = prompt_template.format_messages(**{"question": message, "context": context})
        else:
            prompt = prompt_template.format_messages(**{"question": message})

        llm_response = self.llm.invoke(prompt)

        self.response_history.append(llm_response.content)
        self.query_history.append(message)

        return llm_response.content,context
    
def convert_chat_histories_to_mapping(chats):
    """Utility method for converting a dictionary of chats to a mapping from (key, query) to response. This is
    intended to be used to be able to merge results from chats back into a DataFrame that contains queries, responses,
    and context document names.
    
    Args:
        - chats: dictionary of SimulatedQualifiedHealthChat objects.
    
    Returns:
        - mapping from pairs of chat dictionary keys and queries to responses
    """
    mapping = {}

    for key in chats:
        local_mapping = convert_chat_history_to_mapping(chats[key], filename=key)
        for local_key in local_mapping:
            mapping[local_key] = local_mapping[local_key]
    return mapping

def convert_chat_history_to_mapping(chat, filename=None):
    """Utility method for converting a single chat to a mapping from query to response. The keys of the resulting
    dictionary will be (filename, query) pairs if filename is not None and query otherwise.
    
    Args:
        - chat: A SimulatedQualifiedHealthChat object.
        - filename: The name of the file used with this chat. If None, the key will be the query.

    Returns:
        - mapping from either (filename, query) pairs to response or query to response. 
    """
    if filename is not None:
        return {(filename, query): response for query, response in zip(chat.query_history, chat.response_history)}
    return {query: response for query, response in zip(chat.query_history, chat.response_history)}




