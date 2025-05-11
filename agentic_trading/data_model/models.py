from pydantic import BaseModel

class RagToolSchema(BaseModel):
    question:str 
class QuestionRequest(BaseModel):
    question: str