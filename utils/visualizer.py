""" utils/visualizer.py
Visualizer, it is used to draw bboxes on the images
according to detection results by the model.

Copyright 2024 ktun@

CREATED: 2024-11-12 22:23:43
MODIFIED: 2024-11-13 15:37:38
"""
# -*- coding:utf-8 -*-
# import the necessary libraries
import random
import pandas as pd
#
from PIL import Image, ImageDraw, ImageFont
from utils import get_logger


# Setup logger
logger = get_logger(__name__)

class Visualizer:
    def __init__(self):
        """ Initialize the Visualizer class."""
        logger.info("[INFO] Visualizer initialized . . .")
        
    def _generate_class_color(self, class_name: str) -> tuple:
        """
        Generate a unique color for each class based on its name using hashing.
        
        Args:
            class_name (str): The name of the detected class.
        
        Returns:
            tuple: An (R, G, B) color tuple for the class.
        """
        logger.debug("Generating color for class: %s", class_name)
        # Use a hash of the class name to generate a unique color
        random.seed(hash(class_name) % 255)  # Ensure the same class always gets the same color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        logger.debug("Generated color for class '%s': %s", class_name, color)
        return color

    def draw_bounding_boxes(self, image: Image.Image, predictions: pd.DataFrame) -> Image.Image:
        """
        Draw bounding boxes and labels on an image, with different colors for each class.

        Args:
            image (Image.Image): The input image on which to draw.
            predictions (pd.DataFrame): DataFrame with bounding box coordinates, class names, and confidence scores.

        Returns:
            Image.Image: Image with drawn bounding boxes and labels.
        """
        logger.info("Starting to draw bounding boxes on image")
        
        # Check if predictions dataframe is empty
        if predictions.empty:
            logger.warning("No predictions to draw on the image.")
            return image

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # Default font; customize if necessary

        for _, row in predictions.iterrows():
            try:
                # Extract bounding box coordinates and class name
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                label = f"{row['name']} ({row['confidence']:.2f})"
                
                # Generate a unique color for each class
                color = self._generate_class_color(row['name'])

                # Draw the bounding box with the specified color
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
                
                # Calculate text size using textbbox
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Determine text position
                text_position = (xmin, ymin - text_height if ymin > text_height else ymin)
                
                # Draw a filled rectangle behind the text for readability
                draw.rectangle(
                    [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                    fill=color
                )
                # Draw the text label
                draw.text(text_position, label, fill="white", font=font)

            except Exception as e:
                logger.error("Error processing bounding box at row %d: %s", _, str(e))

        logger.info("Finished drawing bounding boxes on image")
        return image