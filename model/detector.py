""" model/detector.py
Detector, it is used to run ultralytics-yolo detector

Copyright 2024 ktun@

CREATED: 2024-11-12 00:11:57
MODIFIED: 2024-11-13 17:47:19
"""
# -*- coding:utf-8 -*-
# import the necessary libraries
import io
import pandas as pd
import numpy as np
#
from PIL import Image
from typing import Optional
from ultralytics import YOLO
from utils import DataProcessor, get_logger


# Setup logger
logger = get_logger(__name__)

class Detector:
    def __init__(self, model_path, device):
        """Initialize the DataProcessor class. """
        self.model_path = model_path
        self.device = device
        self.data_processor = None
        self.model = None
        logger.debug(f"Initializing detector with model path: {self.model_path} and device: {self.device}")
        #
        self.__init_resource()
        
    def __init_resource(self):
        """Initialize the DataProcessor resources """
        try:
            logger.info("Initializing resources for the detector . . .")
            # Initializing the model
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}.")
            
            # Initializing the data processor
            self.data_processor = DataProcessor(self.device)
            logger.info(f"DataProcessor initialized for device: {self.device}.")
            
        except Exception as e:
            logger.error(f"Error initializing resources: {e}")
            raise

    def get_model_predict(self, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
        """
        Get the predictions of a model on an input image.
        
        Args:
            input_image (Image): The image on which the model will make predictions.
            save (bool, optional): Whether to save the image with the predictions. Defaults to False.
            image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
            conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
            augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
        
        Returns:
            pd.DataFrame: A DataFrame containing the predictions.
        """
        try:
            logger.info(f"Making predictions on image with size: {input_image.size}, confidence threshold: {conf}")
            
            # Make predictions
            predictions = self.model.predict(
                imgsz=image_size,
                source=input_image,
                conf=conf,
                save=save,
                augment=augment,
                flipud=0.0,
                fliplr=0.0,
                mosaic=0.0,
            )
            logger.info("Predictions made successfully.")
            
            # Transform predictions to pandas dataframe
            predictions_df = self.data_processor.transform_predict_to_df(predictions, self.model.names)
            logger.debug(f"Predictions converted to dataframe with {len(predictions_df)} entries.")
            
            return predictions_df
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise