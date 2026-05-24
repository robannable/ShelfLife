"""
ShelfLife - Intelligent Library Cataloging System
Entry point: configures the page, loads CSS, mounts navigation, and runs the selected view.
"""
from pathlib import Path

import streamlit as st

import config
from logger import ShelfLifeLogger, get_logger
from shelflife_app.sidebar import render_sidebar_extras

ShelfLifeLogger().set_level(getattr(config, 'LOG_LEVEL', 'INFO'))
logger = get_logger(__name__)


def load_css(file_name: str):
    css_path = Path('static') / file_name
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found: {css_path}")


def main():
    st.set_page_config(
        page_title="ShelfLife",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    load_css('styles.css')

    pages = [
        st.Page("shelflife_app/views/add_book.py", title="Add Book", icon="➕", default=True),
        st.Page("shelflife_app/views/view_collection.py", title="View Collection", icon="📚"),
        st.Page("shelflife_app/views/analytics.py", title="Analytics", icon="📊"),
        st.Page("shelflife_app/views/network_view.py", title="Network View", icon="🔗"),
        st.Page("shelflife_app/views/executive_summary.py", title="Executive Summary", icon="📋"),
        st.Page("shelflife_app/views/ask_library.py", title="Ask the Library", icon="💬"),
    ]
    nav = st.navigation(pages, position="sidebar")

    render_sidebar_extras()

    nav.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the logs and restart the application.")
