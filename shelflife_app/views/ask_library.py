"""Ask the Library page: single-shot Q&A over the library catalog."""
import json
import time

import streamlit as st

from logger import get_logger
from shelflife_app.library_utils import generate_library_json
from shelflife_app.services import get_book_service, get_database
from shelflife_app.ui_helpers import render_page_header

logger = get_logger(__name__)

db = get_database()
book_service = get_book_service()


render_page_header(
    "Ask the Library",
    "Have a conversation with your book collection",
    "💬"
)

st.markdown('''
    <div style="background: #F3F4F6; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <strong>Try asking:</strong>
        <ul style="margin: 0.75rem 0 0 0; padding-left: 1.25rem; color: #6B7280;">
            <li>What themes are common across my collection?</li>
            <li>Which authors do I read the most?</li>
            <li>Suggest a book for a rainy day</li>
            <li>What genres are underrepresented?</li>
            <li>Find connections between my favorite books</li>
        </ul>
    </div>
''', unsafe_allow_html=True)

query = st.text_area(
    "Your Question",
    placeholder="Ask anything about your library...",
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    ask_button = st.button("🔮 Ask the Library", use_container_width=True, type="primary")

if ask_button:
    if not query:
        st.warning("Please enter a question")
        st.stop()

    try:
        try:
            with open("data/library_catalog.json", "r") as f:
                library_data = json.load(f)
        except FileNotFoundError:
            st.warning("Library catalog not found. Generating it now...")
            json_path, library_data = generate_library_json(db)
            st.success("Catalog generated!")

        progress = st.progress(0, text="Thinking...")
        progress.progress(30, text="Analyzing your question...")
        progress.progress(60, text="Searching your library...")

        response = book_service.ask_library_question(query, library_data)

        progress.progress(100, text="Complete!")
        time.sleep(0.3)
        progress.empty()

        if response:
            st.markdown("### Response")
            st.markdown(f'''
                <div style="background: linear-gradient(135deg, #EEF2FF 0%, #F3F4F6 100%); border-radius: 12px; padding: 1.5rem; border-left: 4px solid #6366F1;">
                    {response}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.error("Failed to get a response. Please try again.")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.error("An error occurred while processing your question")
