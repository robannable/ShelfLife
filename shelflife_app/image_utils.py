"""Image processing helpers for cover uploads."""
import io

import streamlit as st
from PIL import Image

from logger import get_logger
from shelflife_app.services import _get_config

logger = get_logger(__name__)

FORMAT_MAP = {
    'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
    'gif': 'GIF', 'bmp': 'BMP', 'webp': 'WEBP'
}


def process_image(uploaded_file):
    """Process and resize uploaded images."""
    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        image_format = FORMAT_MAP.get(file_extension, 'JPEG')

        max_size = _get_config('MAX_IMAGE_SIZE', 800)
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image_format)
        return img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        st.error("Error processing image. Please ensure it's a valid image file.")
        return None
