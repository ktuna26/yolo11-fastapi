""" utils/data_processor.py
Data Processor, it is used to process
data for translation process of the application.

Copyright 2024 ktun@

CREATED: 2024-11-12 21:43:54
MODIFIED: 2024-11-13 12:37:58
"""
# -*- coding:utf-8 -*-
# import the necessary libraries
import pandas as pd
from utils import get_logger


# Setup logger
logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, device):
        """Initialize the DataProcessor class. """
        logger.info("DataProcessor initialized . . .")
        self.device = device

    def transform_predict_to_df(self, results: list, labeles_dict: dict) -> pd.DataFrame:
        """
        Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

        Args:
            results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
            labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
            
        Returns:
            predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
        """
        logger.info("Transforming predictions to DataFrame...")
        try:
            # Log the type and attributes of the results object to understand its structure
            logger.debug("Results object type: %s", type(results))
            logger.debug("Results attributes: %s", dir(results))
            
            # get the all predictions
            predictiions = results[0].to(self.device).numpy()
            
            # Transform the Tensor to numpy array
            predict_bbox = pd.DataFrame(predictiions.boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
            logger.debug("DataFrame columns after transformation: %s", predict_bbox.columns)
            
            # Add the confidence of the prediction to the DataFrame
            predict_bbox['confidence'] = predictiions.boxes.conf
            logger.debug("Added confidence to DataFrame")
            
            # Add the class of the prediction to the DataFrame
            predict_bbox['class'] = (predictiions.boxes.cls).astype(int)
            logger.debug("Added class to DataFrame")
            
            # Replace the class number with the class name from the labeles_dict
            predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
            logger.info("Transformation complete. DataFrame created with %d rows", len(predict_bbox))
            
            return predict_bbox
        except Exception as e:
            logger.error("Error while transforming predictions to DataFrame: %s", str(e))
            raise