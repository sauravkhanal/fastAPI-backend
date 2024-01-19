from ultralytics import YOLO
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import cv2
import os


def check_make_dir(*directories: str) -> None:
    """
    Checks if directory exists, creates if not

    Args:
        *directories: variable number of path of directory to check
    """
    [os.makedirs(directory) for directory in directories if not os.path.exists(directory)]
    # both do same
    # for directory in directories:
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)


# path declarations
image_save_path = "./saved_images"

input_save_path = image_save_path + "/input"
output_save_path = image_save_path + "/output"

check_make_dir(input_save_path, output_save_path)

model_path = "./model/best.pt"
yolo_model = YOLO(model_path, task='detect')

api_endpoint = "https://yoloapi.khanalsaurav.com.np/getimage"


class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Detection(BaseModel):
    classNumber: int
    name: str
    confidence: float
    box: Box


# custom datatype for returning result of detection
class DetectionResult(BaseModel):
    imageURL: str
    numberOfDetection: int
    detections: list[Detection]


class ReturnDict(BaseModel):
    statusCode: int
    message: str
    data: DetectionResult


def infer(image):
    timestamp = save_input(image)
    results = yolo_model.predict(source=image)
    output_image_url = save_output(results, timestamp)
    return convert_to_json(results, output_image_url)


def convert_to_json(results, output_image_url: str):
    result = results[0]
    names = result.names

    # num of box detected ie numberOfDetections
    n = len(result.boxes)

    # all detection of one image
    detection_result = DetectionResult(
        imageURL=output_image_url,
        numberOfDetection=n,
        detections=[]
    )

    # now iterate through every box to add result(class and coordinates) of detection (of one image)
    for i in range(n):
        data = result.boxes[i]

        box_value = [round(float(x), 2) for x in list(result.boxes[i].xywh[0])]

        detection_instance = Detection(
            classNumber=int(data.cls),
            name=names[int(data.cls)],
            confidence=round(float(data.conf) * 100, 2),
            box=Box(
                x=box_value[0],
                y=box_value[1],
                width=box_value[2],
                height=box_value[3],
            )
        )

        detection_result.detections.append(detection_instance)

    return jsonable_encoder(
        ReturnDict(
            statusCode=200,
            message="Inference Successful",
            data=detection_result
        )
    )


def generate_url(timestamp) -> str:
    return api_endpoint + "/" + timestamp


def save_input(image) -> str:
    # timestamp for current date and time e.g. '2023-11-26_16-33-28'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    cv2.imwrite(f"{input_save_path}/{timestamp}.jpg", image)
    return timestamp


def save_output(results, timestamp: str) -> str:
    # result is list of detections for single prediction
    # might need modification in case of batch image / mp4 prediction.

    image_array = results[0].plot()
    cv2.imwrite(f"{output_save_path}/{timestamp}.jpg", image_array)
    # print(f"{output_save_path}/{name}.jpg")
    return generate_url(timestamp)
