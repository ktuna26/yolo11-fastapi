""" app.py
Main script, it is used to run Fast API service
for YOLO11 object detection.

Copyright 2024 ktun@

CREATED: 2024-11-12 23:50:34
MODIFIED: 2024-11-13 16:13:08
"""
# -*- coding:utf-8 -*-
# import the necessary libraries
import json
import pandas as pd
#
from PIL import Image
from model import Detector
from fastapi.middleware.cors import CORSMiddleware
from utils import ImageProcessor, Visualizer, get_logger
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi import FastAPI, status, HTTPException, File, UploadFile


#region Configuration
# Change thepaths with the desired files
MODEL_PATH = "./weights/yolo11s.pt"
SWAGGER_JSON_PATH = "./data/swagger.json"
# Define the device
DEVICE = "cpu"
# Define the allowed origins
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8001", "*"]
#endregion


#region Components
# Initialize components
logger = get_logger(__name__)  # setup logger

detector = Detector(model_path=MODEL_PATH, device=DEVICE)
image_processor = ImageProcessor()
visualizer = Visualizer()

logger.info("FastAPI components initialized. Detector model loaded from '%s' on device '%s'.", MODEL_PATH, DEVICE)
#endregion


#region FastAPI
# Set a title
app = FastAPI(
    title="Yolo11 Object Detection FastAPI Service",
    description="""This API allows you to obtain object detection values 
            from an image, returning both the image and a JSON result.""",
    version="v1.0",
)

# Middleware for handling CORS, allowing specific origins
logger.info("CORS middleware initialized with allowed origins: %s", ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json() -> None:
    """Save OpenAPI documentation data to a JSON file for offline use."""
    openapi_data = app.openapi()
    with open(SWAGGER_JSON_PATH, "w") as file:
        json.dump(openapi_data, file)
    logger.info("OpenAPI JSON documentation saved to '%s'.", SWAGGER_JSON_PATH)

@app.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Redirect root URL to API documentation."""
    logger.info("Redirecting to /docs.")
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def perform_healthcheck() -> dict:
    """Healthcheck endpoint to confirm the service is running."""
    logger.info("Healthcheck endpoint hit.")
    return {"healthcheck": "Everything OK!"}
#endregion


#region Helper Functions
def crop_image_by_predict(image: Image.Image, predict: pd.DataFrame, crop_class_name: str) -> Image.Image:
    """Crop an image based on a specific object detection.

    Args:
        image (Image.Image): The image to be cropped.
        predict (pd.DataFrame): DataFrame containing object detection predictions.
        crop_class_name (str): Object class name to crop the image by.

    Returns:
        Image.Image: Cropped image with the bounding box of the specified object class.
    
    Raises:
        HTTPException: If the specified object class is not found in the detections.
    """
    logger.info("Attempting to crop image for class '%s'.", crop_class_name)
    crop_predictions = predict[predict['name'] == crop_class_name]

    if crop_predictions.empty:
        logger.error("Crop class '%s' not found in image", crop_class_name)
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in image")

    # Select the detection with the highest confidence if multiple are present
    crop_predictions = crop_predictions.sort_values(by='confidence', ascending=False).iloc[0]
    crop_bbox = crop_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values
    img_cropped = image.crop(crop_bbox)
    
    logger.info("Image cropped successfully for class '%s'.", crop_class_name)
    return img_cropped
#endregion


#region Main Endpoints
@app.post("/img_object_detection_to_json")
async def img_object_detection_to_json(file: UploadFile) -> dict:
    """
    Perform object detection on an uploaded image and return JSON with detected objects.

    Args:
        file (UploadFile): Image file uploaded by user.

    Returns:
        dict: JSON response containing detected objects and their confidence scores.
    """
    logger.info("Received image file for object detection.")

    # Initialize result dictionary
    result = {"detect_objects": None, "detect_objects_names": None}

    # Convert image file to image object
    input_image = image_processor.get_image_from_bytes(await file.read())
    logger.info("Image file converted to image object.")

    # Perform detection
    predictions = detector.get_model_predict(
        input_image=input_image,
        save=False,
        image_size=640,
        conf=0.5,
        augment=False
    )
    logger.info("Model prediction completed for image.")

    # Process detection results
    detection_results = predictions[['name', 'confidence']]
    detected_objects = detection_results['name'].tolist()
    result["detect_objects_names"] = ', '.join(detected_objects)
    result["detect_objects"] = detection_results.to_dict(orient="records")

    # Log results
    logger.info("Detection results: %s", result)
    return result

@app.post("/img_object_detection_to_img")
async def img_object_detection_to_img(file: UploadFile) -> StreamingResponse:
    """
    Perform object detection on an image and return the image with bounding boxes.

    Args:
        file (UploadFile): Image file in bytes format.

    Returns:
        StreamingResponse: Image in bytes with bounding boxes drawn.
    """
    logger.info("Received image file for object detection with bounding boxes.")

    # Load and preprocess image
    input_image = image_processor.get_image_from_bytes(await file.read())
    logger.info("Image file converted to image object.")

    # Get predictions from the model
    predictions = detector.get_model_predict(
        input_image=input_image,
        save=False,
        image_size=640,
        conf=0.5,
        augment=False
    )
    logger.info("Model prediction completed for image.")

    # Draw bounding boxes on the image
    annotated_image = visualizer.draw_bounding_boxes(image=input_image, predictions=predictions)
    logger.info("Bounding boxes drawn on image.")

    # Stream the response
    image_stream = image_processor.get_bytes_from_image(annotated_image)
    logger.info("Returning image with bounding boxes.")
    return StreamingResponse(image_stream, media_type="image/jpeg")
#endregion