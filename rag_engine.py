import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGEngine")


class BilingualSchemeResponse(BaseModel):
    is_agricultural_query: bool = Field(
        description="False if the query is strictly unrelated to agriculture, farming, crops, rural credit, or farmer welfare schemes."
    )
    english_response: str = Field(
        description="Structured English technical explanation detailing eligibility, benefits, and procedural steps."
    )
    tamil_response: str = Field(
        description="Accurate Tamil translation/interpretation of the response text (தமிழ் விளக்கம்)."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of exact sources and page numbers used to answer the query."
    )
    confidence_score: float = Field(
        default=0.0,
        description="Estimated RAG retrieval grounding confidence score (0.0 to 1.0)."
    )


class PDFAutoMetadata(BaseModel):
    scheme_name: str = Field(
        description="Official title or standard abbreviation of the scheme extracted from document headers (e.g., PM-KISAN, PMFBY, KCC, Kalaignar Scheme)."
    )
    jurisdiction: str = Field(
        description="Must be strictly either 'Central' or 'Tamil Nadu' depending on whether it is issued by the Union Government or State Government of Tamil Nadu."
    )
    category: str = Field(
        description="Must be strictly one of: 'income_support', 'credit', or 'insurance'."
    )


class AgriculturalRAG:
    def __init__(
        self,
        chroma_persist_dir: str = "data/vectorstore",
        collection_name: str = "agri_schemes",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini"
    ):
        self.persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        self.memory = ChatMessageHistory()
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        self.parser = PydanticOutputParser(pydantic_object=BilingualSchemeResponse)
        self.chain = self._build_execution_chain()

    def ingest_documents(self, raw_docs: List[Dict[str, Any]]) -> int:
        """Processes raw ingested records into chunked, embedded vector points."""
        documents: List[Document] = []
        for entry in raw_docs:
            chunks = self.text_splitter.split_text(entry["text"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata=entry["metadata"]
                )
                documents.append(doc)
                
        if documents:
            try:
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
                logger.info(f"Indexed {len(documents)} document chunks into ChromaDB.")
                return len(documents)
            except Exception as e:
                logger.error(f"Vector DB indexing failed: {str(e)}", exc_info=True)
                raise
        return 0

    def ingest_pdf_file(self, file_bytes: bytes, filename: str, scheme_name: str, level: str, category: str) -> int:
        """Parses an uploaded PDF circular from memory and indexes it with metadata."""
        extracted_pages = []
        temp_path = Path(f"data/pdfs/{filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(file_bytes)

        with pdfplumber.open(temp_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text()
                if raw and raw.strip():
                    extracted_pages.append({
                        "text": " ".join(raw.split()),
                        "metadata": {
                            "source": filename,
                            "page": idx,
                            "scheme_name": scheme_name,
                            "level": level,
                            "category": category,
                            "type": "uploaded_pdf"
                        }
                    })
        return self.ingest_documents(extracted_pages)

    def extract_pdf_metadata(self, file_bytes: bytes, filename: str) -> PDFAutoMetadata:
        """Reads initial pages of the PDF and uses LLM to deduce scheme name, jurisdiction, and category."""
        extracted_sample = ""
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with pdfplumber.open(tmp_path) as pdf:
                # Read first two pages for title, gazette headers, and scheme terms
                for page in pdf.pages[:2]:
                    text = page.extract_text() or ""
                    extracted_sample += text + "\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        metadata_parser = PydanticOutputParser(pydantic_object=PDFAutoMetadata)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an administrative metadata classifier for Indian government agricultural documents.\n"
                "Analyze the provided document excerpt and identify:\n"
                "1. scheme_name: Official title or standard acronym (e.g., PM-KISAN, PMFBY, KCC, Kalaignar Scheme).\n"
                "2. jurisdiction: Must be strictly either 'Central' or 'Tamil Nadu'.\n"
                "3. category: Must be strictly one of 'income_support', 'credit', or 'insurance'.\n\n"
                "{format_instructions}"
            )),
            ("human", "Filename: {filename}\n\nDocument Excerpt:\n{excerpt}")
        ])

        extraction_chain = prompt | self.llm | metadata_parser
        
        try:
            return extraction_chain.invoke({
                "filename": filename,
                "excerpt": extracted_sample[:3000],
                "format_instructions": metadata_parser.get_format_instructions()
            })
        except Exception as e:
            logger.warning(f"Metadata auto-detection failed: {e}. Falling back to default values.")
            return PDFAutoMetadata(
                scheme_name=filename.replace(".pdf", ""),
                jurisdiction="Central",
                category="income_support"
            )

    def _build_execution_chain(self):
        system_prompt = (
            "You are a Senior Agricultural Scheme Officer assisting farmers and rural stakeholders in India.\n"
            "Use the provided verified context to answer inquiries regarding Central and Tamil Nadu agricultural programs.\n\n"
            "CRITICAL DOMAIN GUARDRAILS:\n"
            "1. Domain Classification:\n"
            "   - Mark 'is_agricultural_query' as TRUE for questions regarding agriculture, crops, farming, subsidies, "
            "     loans (KCC), income transfers (PM-KISAN), insurance (PMFBY), or village schemes (Kalaignar scheme).\n"
            "   - Mark 'is_agricultural_query' as FALSE ONLY IF strictly non-agricultural (e.g., crypto, gaming, sports).\n"
            "2. Clarification on Mismatches: If a user asks about income support under a credit scheme like KCC, clearly explain "
            "   that KCC provides subsidized credit/loans, not cash grants.\n"
            "3. Bilingual Delivery: Provide a technical English explanation and an authentic Tamil translation (தமிழ் விளக்கம்).\n"
            "4. Return ONLY a valid JSON instance conforming to the instructions below.\n\n"
            "Context Information:\n{context}\n\n"
            "{format_instructions}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        return prompt | self.llm | self.parser

    def query(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, str]] = None,
        k: int = 4
    ) -> BilingualSchemeResponse:
        """Executes retrieval with dynamic metadata filters, scoring, and resilient fallback querying."""
        try:
            active_filters = {k: v for k, v in (filter_metadata or {}).items() if v}
            
            chroma_filter = None
            if len(active_filters) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in active_filters.items()]}
            elif len(active_filters) == 1:
                chroma_filter = active_filters

            docs_and_scores = []
            if chroma_filter:
                try:
                    docs_and_scores = self.vector_store.similarity_search_with_score(
                        question, k=k, filter=chroma_filter
                    )
                except Exception as filter_err:
                    logger.warning(f"Filter search failed: {filter_err}. Falling back to global search.")
                    docs_and_scores = []

            # Global fallback if filtered retrieval yields no results
            if not docs_and_scores:
                docs_and_scores = self.vector_store.similarity_search_with_score(question, k=k)

            formatted_context = ""
            citations = []
            scores = []

            for doc, score in docs_and_scores:
                src = doc.metadata.get("source", "Official Scheme Document")
                page = doc.metadata.get("page", None)
                citation = f"{src} (Page {page})" if page else src
                if citation not in citations:
                    citations.append(citation)
                scores.append(score)
                formatted_context += f"[Source: {citation}]\n{doc.page_content}\n\n"

            # Compute retrieval grounding score (L2 distance converted to 0-1 confidence)
            avg_distance = sum(scores) / len(scores) if scores else 1.0
            confidence = max(0.05, min(0.99, round(1.0 - (avg_distance / 2.0), 2)))

            if not formatted_context.strip():
                formatted_context = "No specific scheme documents were matched in the database. Provide general verified agricultural information."
                confidence = 0.20

            response: BilingualSchemeResponse = self.chain.invoke({
                "context": formatted_context,
                "question": question,
                "history": self.memory.messages,
                "format_instructions": self.parser.get_format_instructions()
            })

            response.confidence_score = confidence

            if response.is_agricultural_query:
                response.citations = citations if citations else ["Agri Scheme Portal"]
            else:
                response.citations = []
                response.confidence_score = 0.0
                response.english_response = (
                    "I am specifically designed to assist with government agricultural schemes. "
                    "This query falls outside agricultural welfare programs."
                )
                response.tamil_response = (
                    "நான் அரசு விவசாயத் திட்டங்களுக்கு மட்டுமே பதிலளிக்க வடிவமைக்கப்பட்டுள்ளேன். "
                    "இந்தக் கேள்வி விவசாய நலத் திட்டங்களின் வரம்பிற்கு வெளியே உள்ளது."
                )

            self.memory.add_user_message(question)
            self.memory.add_ai_message(f"EN: {response.english_response} | TA: {response.tamil_response}")
            return response

        except Exception as e:
            logger.critical(f"RAG Failure: {str(e)}", exc_info=True)
            return BilingualSchemeResponse(
                is_agricultural_query=False,
                english_response=f"Operational Error: {str(e)}",
                tamil_response="கோரிக்கையைச் செயலாக்குவதில் தொழில்நுட்பப் பிழை ஏற்பட்டுள்ளது.",
                citations=[],
                confidence_score=0.0
            )