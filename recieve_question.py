from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from find_qs import ComplexQuestionHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
VALID_HASH = "827ccb0eea8a706c4c34a16891f84e7b"  # md5 hash of "12345"

# Initialize ComplexQuestionHandler
handler = ComplexQuestionHandler('data_sources/db.csv')

# ThreadPoolExecutor for concurrent question processing
executor = ThreadPoolExecutor(max_workers=10)  # Adjust the number of workers as needed

def verify_hash(provided_hash: str) -> bool:
    return provided_hash == VALID_HASH

class QuestionModel(BaseModel):
    question: str
    authen_pass: str

class QuestionResponse(BaseModel):
    question_ids: List[str]
    std_questions: List[str]
    # context_id: str

async def process_question(question: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, handler.process_complex_question, question)

    # logger.info(f"Processed question {question}")
    # Ensure all fields are of the correct type
    return {
        "question_ids": [str(qid) for qid in result['question_ids']],
        "std_questions": result['std_questions'],
    }

@app.post("/recieve_question", response_model=QuestionResponse)
async def recieve_question(data: QuestionModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        logger.info(f"-"*50)
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        result = await process_question(data.question)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(e)
        logger.info(f"-" * 50)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"status": "active", "message": "Server is ready to analyze questions"}

async def test_question_id(question: str, authen_pass: str):
    data = QuestionModel(question=question, authen_pass=authen_pass)
    try:
        response = await recieve_question(data)
        # logger.info(f"Question id: {response.question_ids}")
        # logger.info(f"User Question: {response.question}")
        # logger.info(f"Standard Questions: {response.std_questions}")

        print(f"Question: {question}")
        print(f"Question IDs: {response.question_ids}")
        print(f"Standard Questions: {response.std_questions}")
        # print(f"Context ID: {response.context_id}")
    except HTTPException as e:
        print(f"Error: {e.detail}")
    print("-" * 50)

if __name__ == "__main__":
    logger.info(f"Starting Server Receive Question and Return QuestionIds with {uvicorn.__version__}")
    # Test code
    # print("Running test cases:")
    #
    # test_questions = [
    #     ("Làm thế nào để tra cứu đơn hàng?", VALID_HASH),
    #     ("Quy định về hàng cấm là gì?", VALID_HASH),
    #     ("Cách tính thuế nhập khẩu và thời gian thông quan?", VALID_HASH),
    #     ("Thủ tục hải quan điện tử là gì?", "invalid_hash"),
    # ]
    # async def run_tests():
    #     for question, authen_pass in test_questions:
    #         await test_question_id(question, authen_pass)
    #
    # asyncio.run(run_tests())

    # Start the server
    # print("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)