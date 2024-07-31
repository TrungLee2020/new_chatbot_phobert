from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
from typing import List, Dict, Any, Optional
import uvicorn
from find_qs import ComplexQuestionHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import json
from datetime import datetime
import pytz
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
VALID_HASH = "827ccb0eea8a706c4c34a16891f84e7b"  # md5 hash of "12345"

# Initialize ComplexQuestionHandler
handler = ComplexQuestionHandler('data_json/db.csv')

# ThreadPoolExecutor for concurrent question processing
executor = ThreadPoolExecutor(max_workers=10)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Initialize mT5 model for text generation
llm_model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, max_length=512)


def verify_hash(provided_hash: str) -> bool:
    return provided_hash == VALID_HASH


def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data if isinstance(data, list) else [data]


def get_current_time_hcm() -> str:
    hcm_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now = datetime.now(hcm_tz)
    return now.strftime("%d-%m %H:%M:%S")


def get_weather() -> str:
    api_key = "your_api_key"
    city = "Hanoi"
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric")
    data = response.json()
    temp = data['main']['temp']
    weather_description = data['weather'][0]['description']
    return f"Thời tiết hiện tại ở {city} là {temp}°C với {weather_description}."


def handle_specific_questions(question: str) -> str:
    if "tên" in question.lower():
        return "Tôi là một trợ lý ảo, bạn hãy gọi tôi là Lý"
    elif "hiện tại mấy giờ" in question.lower() or "mấy giờ rồi" in question.lower():
        return f"Hiện tại là {get_current_time_hcm()}."
    elif "thời tiết" in question.lower():
        return get_weather()
    else:
        return None


class QuestionModel(BaseModel):
    question: str
    authen_pass: str


class QuestionResponse(BaseModel):
    question_ids: List[str]
    std_questions: List[str]
    success: bool


class QueryModel(BaseModel):
    link: Optional[str] = Field(None, description="URL to fetch JSON data")
    authen_pass: str = Field(..., description="Authentication password")
    use_local: bool = Field(False, description="Flag to use local file")
    local_file_path: Optional[str] = Field(None, description="Path to local JSON file")


class ResultItem(BaseModel):
    context_id: Optional[str] = Field(None, description="ID of the context")
    std_question: str = Field(..., description="Standard question")
    answer: str = Field(..., description="Generated answer")
    question_id: str = Field(..., description="ID of the question")


class ResponseModel(BaseModel):
    results: List[ResultItem] = Field(..., description="List of processed results")
    success: bool


class AnswerDataModel(BaseModel):
    results: List[Dict[str, Any]]
    authen_pass: str
    use_local: bool
    local_file_path: Optional[str] = None
    link: Optional[str] = None


async def process_question(question: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, handler.process_complex_question, question)
    return {
        "question_ids": [str(qid) for qid in result['question_ids']],
        "std_questions": result['std_questions'],
    }


@app.post("/recieve_question", response_model=QuestionResponse)
async def recieve_question(data: QuestionModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        result = await process_question(data.question)
        return QuestionResponse(**result, success=True)
    except Exception as e:
        logger.error(f"Error in recieve_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_json_and_run_rag(item: Dict[str, Any]) -> str:
    try:
        std_question = item.get("std_question", "")

        # First, check if it's a specific question that can be handled directly
        specific_answer = handle_specific_questions(std_question)
        if specific_answer:
            logger.info(f"Specific answer generated for question: {std_question}")
            return specific_answer

        context = item.get("context", "")
        std_answer = item.get("std_answer", "")

        logger.info(f"Processing context for question: {std_question}")

        full_context = f"{std_answer}\n{context}"
        sentences = full_context.split('.')
        sentences = [sent.strip() for sent in sentences if sent.strip()]

        sentence_embeddings = model.encode(sentences)
        question_embedding = model.encode([std_question])

        similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
        top_sentence_indices = np.argsort(similarities)[-3:][::-1]
        relevant_context = '. '.join([sentences[i] for i in top_sentence_indices])

        qa_result = qa_pipeline(question=std_question, context=relevant_context)
        answer = qa_result['answer']

        # Use mT5 for text generation
        prompt = f"Trả lời câu hỏi sau dựa trên thông tin đã cho: {std_question}\nThông tin: {answer}"
        generated_text = llm_pipeline(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

        conversational_answer = (
            f"Đối với câu hỏi '{std_question}', tôi xin được chia sẻ như sau: {generated_text}\n"
            "Tôi hy vọng thông tin này hữu ích cho bạn. Nếu bạn có bất kỳ câu hỏi nào khác, đừng ngần ngại hỏi nhé!"
        )

        logger.info(f"Successfully generated answer for question: {std_question}")
        return conversational_answer

    except Exception as e:
        logger.error(f"Error in process_json_and_run_rag: {str(e)}", exc_info=True)
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi '{std_question}'. Vui lòng thử lại sau."


@app.post("/ask", response_model=ResponseModel)
async def ask_question(query: QueryModel):
    if not verify_hash(query.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        if query.use_local:
            if not query.local_file_path:
                raise ValueError("Local file path is required when use_local is True")
            if not os.path.exists(query.local_file_path):
                raise FileNotFoundError(f"Local file not found: {query.local_file_path}")
            json_data = load_json_from_file(query.local_file_path)
            logger.info(f"Data loaded from local file: {query.local_file_path}")
        else:
            if not query.link:
                raise ValueError("URL link is required when use_local is False")

            logger.info(f"Attempting to fetch data from: {query.link}")
            async with httpx.AsyncClient() as client:
                response = await client.get(query.link)
                response.raise_for_status()
                json_data = response.json()
            logger.info("Data fetched successfully from backend API")

        if not json_data:
            raise ValueError("JSON data is empty")

        logger.info("Start processing answer")
        results = []
        for item in json_data:
            if all(key in item for key in ['std_question', 'question_id', 'context', 'std_answer']):
                try:
                    answer = process_json_and_run_rag(item)
                    logger.info(f"Answer generated for question_id: {item['question_id']}")
                    results.append(ResultItem(
                        context_id=item.get("context_id", ""),
                        std_question=item['std_question'],
                        answer=answer,
                        question_id=item['question_id']
                    ))
                except Exception as e:
                    logger.error(f"Error processing item: {e}")

        if not results:
            logger.warning("No results were generated")
            return ResponseModel(results=[], success=False)
        logger.info(f"Successfully processed answer: {results}")
        logger.info(f"Successfully processed {len(results)} items")
        return ResponseModel(results=results, success=True)

    except httpx.RequestError as e:
        logger.error(f"HTTP Request error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error making HTTP request: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/send_answers", response_model=ResponseModel)
async def send_answers(data: AnswerDataModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        if not data.results:
            raise ValueError("No results to send")

        if data.use_local:
            with open(data.local_file_path, 'w', encoding='utf-8') as f:
                json.dump(data.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to local file: {data.local_file_path}")
        else:
            if not data.link:
                raise ValueError("URL link is required when use_local is False")

            async with httpx.AsyncClient() as client:
                response = await client.post(data.link, json={"results": data.results})
                response.raise_for_status()

            logger.info("Results sent to backend API successfully")

        return ResponseModel(results=[ResultItem(**result) for result in data.results], success=True)

    except Exception as e:
        logger.error(f"Unexpected error in send_answers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def read_root():
    return {"status": "active", "message": "Server is ready to analyze questions and generate answers"}


if __name__ == "__main__":
    logger.info(f"Starting Server with {uvicorn.__version__}")
    uvicorn.run(app, host="0.0.0.0", port=8686)