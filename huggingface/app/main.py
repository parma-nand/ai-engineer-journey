from fastapi import FastAPI, HTTPException
from app.schemas import TextRequest, SummaryResponse
from app.hf_service import summarize_text

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Summarizer API is running"}

@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: TextRequest):

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    result = summarize_text(request.text)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    try:
        summary = result[0]["summary_text"]
    except:
        raise HTTPException(status_code=500, detail="Invalid response from model")

    return {"summary": summary}