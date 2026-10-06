import datetime

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


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


class TableMetadataChunk(Base):
    """
    Metadata of every source-database table (table name, description,
    columns) — not row data. One row per table. At text-to-SQL time the
    question is matched against these embeddings to find which 2-3 tables
    are relevant.

    source_table is schema-qualified, e.g. "lb1adm.roams_owner_master".
    """
    __tablename__ = "table_metadata_chunks"
    __table_args__ = {"schema": "chatbot"}

    id = Column(Integer, primary_key=True, index=True)
    source_table = Column(String, unique=True, index=True, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class ChatMessage(Base):
    """
    NEW: Conversation history. Each user question and its assistant answer
    is saved as a separate row, grouped by conversation_id. This table is
    what lets follow-up questions "remember" earlier turns. The columns
    match the existing chatbot.chat_messages table in the database.
    """
    __tablename__ = "chat_messages"
    __table_args__ = {"schema": "chatbot"}

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, nullable=True)
    sources = Column(JSON, nullable=True)  # citations returned with the assistant's answer