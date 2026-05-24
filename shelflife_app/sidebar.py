"""Sidebar content shared across all pages (logo, API status, library stats)."""
import streamlit as st

from api_utils import test_api_connection
from logger import get_logger
from shelflife_app.services import _get_config, get_book_service, get_database

logger = get_logger(__name__)


def render_sidebar_extras():
    """Render logo, system status, and library stats below the navigation menu."""
    db = get_database()
    book_service = get_book_service()

    with st.sidebar:
        st.markdown('''
            <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📚</div>
                <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700;">ShelfLife</h1>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.8;">Intelligent Library Cataloging</p>
            </div>
        ''', unsafe_allow_html=True)

        st.divider()

        st.markdown(
            '<p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; '
            'opacity: 0.7; margin-bottom: 0.75rem;">System Status</p>',
            unsafe_allow_html=True
        )

        if st.button("🔍 Check API Status", use_container_width=True):
            with st.spinner("Checking connections..."):
                llm_status = book_service.test_connection()
                st.write(
                    f"{_get_config('LLM_PROVIDER', 'anthropic').title()} LLM:",
                    "✅" if llm_status["success"] else "❌"
                )

                google_status = test_api_connection("google_books")
                st.write("Google Books API:", "✅" if google_status["success"] else "❌")

                ol_status = test_api_connection("open_library")
                st.write("Open Library API:", "✅" if ol_status["success"] else "❌")

                if _get_config('DEBUG_MODE', False):
                    if not llm_status["success"]:
                        st.error(f"LLM Error: {llm_status.get('error', 'Unknown')}")
                    if not google_status["success"]:
                        st.error(f"Google Books: {google_status.get('error', 'Unknown')}")
                    if not ol_status["success"]:
                        st.error(f"Open Library: {ol_status.get('error', 'Unknown')}")

        try:
            books = db.get_all_books()
            book_count = len(books) if books else 0
            st.markdown(f'''
                <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem; margin-top: 1rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700;">{book_count}</div>
                    <div style="font-size: 0.75rem; opacity: 0.8;">Books in Library</div>
                </div>
            ''', unsafe_allow_html=True)
        except Exception:
            pass
