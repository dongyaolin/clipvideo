from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import pickle
from functools import wraps
from typing import List, Dict, Any, Union
import time
import json
from pathlib import Path
from loggings import logger
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import jieba
from jieba import analyse
from sqlitemodule import TaskManager
import uuid
import io
import pika
app = FastAPI()


def send(input_convert_audio):
    parameters = pika.ConnectionParameters(
                host='localhost',
                # socket_timeout=6000000,  # 增加 socket 超时时间
                heartbeat=0,       # 增加心跳间隔
                # blocked_connection_timeout=300000  # 增加连接被阻塞时的超时时间
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)
    channel.basic_publish(exchange='',  # use a default exchange identified by an empty string
    routing_key='task_queue',  # queue name needs to be specified in the routing_key parameter
    body= input_convert_audio
    )

tm = TaskManager()

# 允许的音频 MIME 类型
ALLOWED_AUDIO_TYPES = ["audio/mpeg", "audio/mp3"]
def save_var(var, filename='debug_var.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(var, f)
def load_var(filename='debug_var.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# 设置允许的源列表
# origins = [
#     "http://localhost",
#     "http://localhost:8080",
#     "http://localhost:3000",
#     "https://your-frontend-domain.com",
# ]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源列表
    allow_credentials=True,  # 允许携带 cookie
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 上传目录
UPLOAD_DIR = "/root/audio_uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)  # 确保目录存在


@app.post("/uploadAudio")
async def upload_audio(file:UploadFile):
    
    try:
        # if file.content_type not in ALLOWED_AUDIO_TYPES:
        #     raise HTTPException(status_code=400, detail="仅支持 MP3 音频文件")

        # if not file.filename.lower().endswith(".mp3"):
        #     raise HTTPException(status_code=400, detail="文件扩展名必须是 .mp3")
        audio_id = str(uuid.uuid4())
        safe_filename = f"{audio_id}.mp3"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)


        max_size = 50 * 1024 * 1024
        file_size = 0
        logger.info("开始上传")
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > max_size:
                    os.remove(file_path)
                    raise HTTPException(status_code=413, detail="文件大小超过 50MB 限制")
                f.write(chunk)
        logger.info("上传成功")
        tm = TaskManager()
        logger.info("添加任务")
        tm.add_task(audio_id, file_path)
        logger.info("添加成功")
        logger.info("开始连接消息队列")
        send(audio_id)
        logger.info(f" [x] Sent {audio_id!r}")
        return {"status_code":1, "msg":"success", "content":{"audio_id":audio_id}}
    
    except Exception as e:
        logger.error(f"上传失败{e}")
        return {"status_code": -1, "msg":"failed", "content": {}}


@app.get("/getStatus")
async def get_status(id:str):
    try:
        logger.info("准备连接数据库，获取数据状态")

        a = tm.get_audio_status(id)
        logger.info("获取成功", a)

        if len(a) ==2:
            status, result_path = a
            if status == "completed":
                return {"status_code":1, "msg":"success", "content":{"status":status}}
            elif status == "processing failed":
                return {"status_code":-1, "msg":"success", "content":{"status":status}}
            elif status == "pending":
                return {"status_code":0, "msg":"success", "content":{"status":status}}
            
    except Exception as e:
        return {"status_code":-1, "msg":"id not found", "content":{}}



@app.get("/getRes")
async def get_res(id:str):
    try:
        logger.info("准备连接数据库，获取数据状态")

        a = tm.get_audio_status(id)
        logger.info("获取成功", a)

        if len(a) ==2:
            status, result_path = a
            logger.info(f"状态{status}：结果：{result_path}")
            
            if status == "completed":
                return {"status_code":1, "msg":"success", "content":{"data":f.read()}}
            elif status == "processing failed":
                return {"status_code":-1, "msg":"processing failed", "content":{}}
            elif status == "pending":
                return {"status_code":0, "msg":"processing", "content":{}}
 
            # return FileResponse(path=result_path)
            # return StreamingResponse(io.BytesIO(open(result_path).read()))
            # return {"status_code":200, "msg":"success", "content":StreamingResponse(io.BytesIO(f.read()))}
    except Exception as e:
        return {"status_code":-1, "msg":"id not found", "content":{}}
        # raise HTTPException(status_code=500, detail=f"{str(e)}")

@app.post("/generateSearchWords")
async def generate_search_word(text:str):
    try:
        a = analyse.extract_tags(text, topK=3)  # topK为返回关键词数量
        b = jieba.cut_for_search(text)
        a+=b
        res = list(set(a))
        return {"status_code":1, "msg":"success", "content":{"res":res}}
        
    except HTTPException as e:
        logger.error(f"{e}")
        return {"status_code": -1, "msg": "failed", "content":{}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("subtitlegenapi:app", host="0.0.0.0", port=8001)