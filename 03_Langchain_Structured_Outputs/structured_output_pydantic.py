from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal, List
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key, temperature=0.5)

# simple typeddict
class Data(BaseModel):
    summary : str
    sentiment : str

# annotated typeddict
class Review(BaseModel):
    key_themes : Annotated[list[str], "Give all the key themes or points discussed in the review in a list"]
    summary : Annotated[str, "Give a brief summary of the review in 1 to 2 lines"]
    sentiment : Annotated[Literal['pos', 'neg', 'neut'], "return the sentiment of the review either as Positive, Negative or Neutral"]
    pros : Annotated[Optional[list[str]] ,  "Give all the important pros in a list"]
    cons : Annotated[Optional[list[str]], "Extract review analysis. Include `cons` only if the text explicitly has a 'Cons' or 'Drawbacks' heading; otherwise return null."]

# class Review(BaseModel):
#     key_themes: List[str] = Field(
#         description="All key themes or points discussed in the review"
#     )
#     summary: str = Field(
#         description="Brief summary of the review in 1–2 lines"
#     )
#     sentiment: Literal["pos", "neg", "neut"] = Field(
#         description="Sentiment of the review: pos, neg, or neut"
#     )
#     pros: List[str] = Field(
#         description="Important pros if explicitly mentioned, otherwise null"
#     )
#     cons: List[str] = Field(
#         description="Important cons if explicitly mentioned, otherwise null"
#     )


structured_model = model.with_structured_output(Review)

response = structured_model.invoke("""I recently watched The Silent River, and it was quite an experience. 
                                   The story had deep emotional moments and explored complex themes about friendship 
                                   and personal growth. The cinematography was breathtaking, and the soundtrack perfectly
                                    complemented the mood. The lead actors delivered stellar performances. However, 
                                   the pacing felt slow in some parts, and a few scenes seemed unnecessary. Overall, 
                                   it was a heartfelt film with memorable moments, though slightly long.""")

print(response)

schema = Review.model_json_schema()  # Pydantic v2

import json

with open(r"03_Langchain_Structured_Outputs\review_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
