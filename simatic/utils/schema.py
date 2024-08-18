import pydantic
from typing import List


class Document(pydantic.BaseModel):
    """
    Document model for storing document information.
    """

    text: str
    embedding: List[float]
