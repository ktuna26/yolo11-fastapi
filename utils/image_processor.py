""" utils/image_processor.py
Image processor, it is used to process
images for application.

Copyright 2024 ktun@

CREATED: 2024-11-12 21:50:34
MODIFIED: 2024-11-13 11:17:08
"""
# -*- coding:utf-8 -*-
# import the necessary libraries
import io
#
from PIL import Image
from utils import get_logger


# Setup logger
logger = get_logger(__name__)

class ImageProcessor:
    def __init__(self):
        """ Initialize the ImageProcessor class."""
        logger.info("ImageProcessor initialized . . .")
        
    def get_image_from_bytes(self, binary_image: bytes) -> Image:
        """Convert image from bytes to PIL RGB format
        
        Args:
            binary_image (bytes): The binary representation of the image
        
        Returns:
            PIL.Image: The image in PIL RGB format
        """
        try:
            input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
            logger.info("Image successfully converted from bytes to PIL format.")
        except Exception as e:
            logger.error("Error converting image from bytes: %s", e)
            raise
        return input_image

    def get_bytes_from_image(self, image: Image) -> bytes:
        """
        Convert PIL image to Bytes
        
        Args:
        image (Image): A PIL image instance
        
        Returns:
        bytes : BytesIO object that contains the image in JPEG format with quality 85
        """
        try:
            return_image = io.BytesIO()
            image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
            return_image.seek(0)  # set the pointer to the beginning of the file
            logger.info("Image successfully converted to bytes.")
        except Exception as e:
            logger.error("Error converting image to bytes: %s", e)
            raise
        return return_image