from pydantic import BaseModel
from typing import List


class Document(BaseModel):
    """
    Document model for storing document information.
    """

    text: str
    embedding: List[float]
