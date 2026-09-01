from app.database import Base
from sqlalchemy import DateTime
import datetime
from pgvector.sqlalchemy import Vector #type: ignore
from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(String, default="user")


class OTP(Base):
    __tablename__ = "otps"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    otp_code = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    source = Column(String)
    dept = Column(String, nullable=True)
    sub_dept = Column(String, nullable=True)
    subject = Column(String, nullable=True)
    doc_date = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    chunks = relationship("Chunk", back_populates="document")   
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    page_number = Column(Integer)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # BGE-M3 ka dimension

    document = relationship("Document", back_populates="chunks")