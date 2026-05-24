"""Singleton service factories backed by Streamlit's resource cache."""
import streamlit as st

import config
from book_service import BookService
from database import Database
from logger import get_logger

logger = get_logger(__name__)


def _get_config(attr: str, default):
    """Get config attribute with fallback default."""
    return getattr(config, attr, default)


@st.cache_resource
def get_database() -> Database:
    """Get or create database instance."""
    try:
        return Database(_get_config('DB_PATH', 'data/database.db'))
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        st.error(f"Database initialization failed: {str(e)}")
        st.stop()


@st.cache_resource
def get_book_service() -> BookService:
    """Get or create book service instance."""
    try:
        return BookService()
    except Exception as e:
        logger.error(f"Failed to initialize book service: {str(e)}", exc_info=True)
        st.error(f"Book service initialization failed. Check your LLM configuration: {str(e)}")
        st.stop()
