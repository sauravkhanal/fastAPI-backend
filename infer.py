import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from infer_yolo import infer, output_save_path
import cv2
import os

app = FastAPI()

origins = [
    "http://localhost:63342/",
    "http://localhost",
    "http://localhost:63342/web"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():
    return jsonable_encoder({
        "statusCode": 200,
        "message": "Kathmandu Durbar square heritage detection YOLO v8 API"
    })


@app.post("/inferyolo/")
async def infer_yolo(file: UploadFile):
    image = await preprocess_image(file)
    # image_array = infer(image)
    # return result.model_dump()
    return infer(image)


async def preprocess_image(file: UploadFile):
    # read the content of image
    contents = await file.read()

    # convert the content to numpy array

    np_array = np.frombuffer(contents, np.uint8)

    # Decode the array into an OpenCV image
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return img


@app.get("/getimage/{image_name}")
async def get_image(image_name: str):
    image_path = output_save_path + "/" + image_name + ".jpg"

    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        return jsonable_encoder({"statusCode": 404, "message": "File not found"})
