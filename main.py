import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Code Reviewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert code reviewer with 15 years of experience.
Review the provided code and respond ONLY in this exact JSON format:
{
  "score": number between 1-10,
  "bugs": [list of bugs found],
  "security": [list of security issues],
  "performance": [list of performance issues],
  "bestPractices": [list of best practice violations],
  "suggestions": [list of improvement suggestions],
  "correctedCode": "complete fixed version of the code"
}
Be specific and detailed. If no issues found in a category return empty array."""


class ReviewRequest(BaseModel):
    code: str
    language: str


class ReviewResponse(BaseModel):
    score: float
    bugs: list[str]
    security: list[str]
    performance: list[str]
    bestPractices: list[str]
    suggestions: list[str]
    correctedCode: str


@app.get("/")
def root():
    return {"status": "AI Code Reviewer API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    if request.language not in ["python", "javascript", "java"]:
        raise HTTPException(
            status_code=400,
            detail="Language must be one of: python, javascript, java",
        )

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    user_message = f"Review this {request.language} code:\n\n```{request.language}\n{request.code}\n```"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        raw_content = completion.choices[0].message.content.strip()

        json_match = re.search(r"\{[\s\S]*\}", raw_content)
        if not json_match:
            raise HTTPException(
                status_code=500, detail="AI returned an unexpected response format"
            )

        parsed = json.loads(json_match.group())

        return ReviewResponse(
            score=float(parsed.get("score", 5)),
            bugs=parsed.get("bugs", []),
            security=parsed.get("security", []),
            performance=parsed.get("performance", []),
            bestPractices=parsed.get("bestPractices", []),
            suggestions=parsed.get("suggestions", []),
            correctedCode=parsed.get("correctedCode", request.code),
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail="Failed to parse AI response as JSON"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI review failed: {str(e)}")
