from app.database import Base
from sqlalchemy import DateTime
import datetime
from pgvector.sqlalchemy import Vector #type: ignore
from sqlalchemy import Column, Integer, String, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "chatbot"}

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    source = Column(String)
    source_url = Column(String, nullable=True)
    dept = Column(String, nullable=True)
    sub_dept = Column(String, nullable=True)
    subject = Column(String, nullable=True)
    doc_date = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = {"schema": "chatbot"}

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("chatbot.documents.id"))
    page_number = Column(Integer)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))

    document = relationship("Document", back_populates="chunks")


class ChatMessage(Base):
    """
    Stores conversation history. Each user question and its assistant
    answer is saved as a separate row, grouped by conversation_id.
    This table is what makes follow-up questions "remember" earlier turns.
    """
    __tablename__ = "chat_messages"
    __table_args__ = {"schema": "chatbot"}

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MasterDataChunk(Base):
    """
    Structured master-data tables (owner master, wagon type master, org
    master, etc.) se aayi rows ka embedded text. Ek row = ek chunk —
    page-splitting ki zaroorat nahi, kyunki har row already ek chhota,
    atomic fact hai (jaise ek owner ka poora record).

    source_table + record_id ka combination unique hai, isse upsert
    (dobara sync chalane par sirf changed rows update hoti hain) possible
    hota hai.
    """
    __tablename__ = "master_data_chunks"
    __table_args__ = (
        UniqueConstraint("source_table", "record_id", name="uq_master_data_source_record"),
        {"schema": "chatbot"},
    )

    id = Column(Integer, primary_key=True, index=True)
    source_table = Column(String, nullable=False)  # e.g. "roams_owner_master"
    record_id = Column(String, nullable=False)      # source table's primary key value
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))