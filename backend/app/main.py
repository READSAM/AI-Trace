from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status #receive requests ,uploads ,  send status or error codes 
from fastapi.responses import JSONResponse #handle JSOn response/exception
import redis #caching
import json #converting pytohn objects into json format to be stored in redis 
import uuid #generating task_id uniquely
from uuid import UUID
from typing import Optional #optional fields

from app.config import settings #api prefixes ,redis connection url
from app.schemas import ( #requests, response schemens
    ModalityEnum, #input provided
    TaskStatusEnum, #task created is updated with these statuses
    TaskAcceptedResponse, #Task assigned to worker
    TaskResultResponse #Task returns a result
)
from app.services.hasher import ContentHasher #gives pHASh for image , SHA256 hash for texts
from app.tasks.celery_app import celery_app  #backgorund processing on images and texts

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
) #create an web app host

# Shared Redis Client for Hash Caching
redis_client = redis.Redis.from_url(settings.REDIS_CACHE_URL, decode_responses=True) #connection pool with the enabling of python object response


@app.post(
    f"{settings.API_V1_STR}/forensics/analyze",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED #requested received and queued
)
async def submit_analysis_job(
    modality: ModalityEnum = Form(...), #takes form inputs
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None)
):
    if modality == ModalityEnum.IMAGE:  #if image input read image bytes and get pHash
        if not file:
            raise HTTPException(status_code=400, detail="Image file required for IMAGE modality.")
        content_bytes = await file.read()
        content_hash = ContentHasher.compute_image_phash(content_bytes)
    elif modality == ModalityEnum.TEXT: #if text input , encode to utf-8 format and get SHA256 hash 
        if not text_content:
            raise HTTPException(status_code=400, detail="text_content required for TEXT modality.")
        content_bytes = text_content.encode("utf-8")
        content_hash = ContentHasher.compute_text_hash(text_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported modality.") 

    # 1. Check O(1) Perceptual Hash Cache / duplicate requests
    cache_key = f"hash:{modality.value}:{content_hash}" #for image = IMAGE:<pHash value> and for text = TEXT:<SHA256 value>
    cached_result = redis_client.get(cache_key) # get the cached result file if any
    if cached_result:
        # Instant return if cached
        result_data = json.loads(cached_result)
        return JSONResponse(status_code=200, content=result_data) #
    # 2. Dispatch Task to Celery
    task_id = str(uuid.uuid4()) #generate unique id for the scoring task

    # Save the ID to Redis with a 24-hour expiration so it doesn't bloat
    redis_client.setex(f"valid_task:{task_id}", 86400, "exists")

    if modality == ModalityEnum.IMAGE: #run image pipeline
        celery_app.send_task(
            "app.tasks.vision_tasks.run_image_pipeline",
            kwargs={"task_id": task_id, "file_bytes_hex": content_bytes.hex()}, #arguments passed to vision task function
            task_id=task_id
        )
    else: #run text pipeline
        celery_app.send_task(
            "app.tasks.text_tasks.run_text_pipeline",
            kwargs={"task_id": task_id, "text_content": text_content}, #arguments passed to text task function
            task_id=task_id
        )

    return TaskAcceptedResponse( #tells the user end that task has been received and queued for analysis in backend
        task_id=task_id,
        status=TaskStatusEnum.QUEUED,
        estimated_ms=450
    )


@app.get(
    f"{settings.API_V1_STR}/forensics/tasks/{{task_id}}", #track task,its status and its response  periodically
    response_model=TaskResultResponse 
)
async def get_task_status(task_id: UUID): #to show progress status at the user end
    
    task_id = str(task_id)
    if not redis_client.exists(f"valid_task:{task_id}"):
        raise HTTPException(status_code=404, detail="Task ID not found or expired.")
    
    async_result = celery_app.AsyncResult(task_id)
    
    if async_result.state == "PENDING":
        return TaskResultResponse(task_id=task_id, status=TaskStatusEnum.QUEUED)
    elif async_result.state in ["STARTED", "RETRY"]:
        return TaskResultResponse(task_id=task_id, status=TaskStatusEnum.PROCESSING)
    elif async_result.state == "SUCCESS":
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatusEnum.COMPLETED,
            **async_result.result #other fields are returned collectively as result
        )
    else:
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatusEnum.FAILED,
            error=str(async_result.result) #need to specify error 
        ) 