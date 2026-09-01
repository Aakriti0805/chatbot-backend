from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr

class UserOut(BaseModel):
    id: int
    email: str
    role: str

    class Config:
        from_attributes = True