import os
import uuid

from uvicorn import run
import uvicorn
from fastapi import FastAPI, Request, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import asyncio

import api
from api.endpoints import generatePic, toCartoon
import requests

app = FastAPI()
# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/a")
def root():
    return {"message": f"Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/generatePic")
def sss(
        uploadImage: str = Body(embed=True),
        rawImage: str = Body(embed=True),
        upIsTop: bool = Body(embed=True),
        upNeedRemove: bool = Body(embed=True),
        uploadImageResizeLong: int = Body(embed=True),
        uploadImageResizeWidth: int = Body(embed=True),
        x: int = Body(embed=True),
        y: int = Body(embed=True)
):
    # generatePic.generate_pic()
    url = generatePic.generate_pic1(uploadImage, rawImage, upIsTop, upNeedRemove, uploadImageResizeLong,
                                    uploadImageResizeWidth, x, y)

    url = url.replace("D:/output", "http://192.168.1.119:9999")

    message = {
        "code": 1,
        "data": {
            "url": url
        }
    }
    return message


@app.post(path="/toCartoon")
async def toCartoon(
        uploadImage: str = Body(embed=True)
):
    path_prefix = "D:/output/toCartoon"
    url = api.endpoints.toCartoon.generate(uploadImage, path_prefix, 2)
    print(url)
    url = url.replace("D:/output", "http://192.168.1.119:9999")
    print(url)
    message = {
        "code": 1,
        "data": {
            "url": url
        }
    }
    return message


@app.post("/uploadImage")
async def upload_image(file: UploadFile = File(...)):
    path_prefix = "D:/input/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    file_suffix = file.filename.split(".")[1]
    save_path = path_prefix + uuid.uuid4().hex + "." + file_suffix
    with open(save_path, 'wb') as f:
        f.write(await file.read())
    return {
        "code": 1,
        "res": '上传成功',
        "data": {
            "url": save_path
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.119", port=8000, timeout_keep_alive=60)
