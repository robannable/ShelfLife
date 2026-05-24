"""Add Book page: form-driven book entry with AI metadata enhancement."""
import time
from datetime import datetime

import streamlit as st

from database import DatabaseError
from logger import get_logger
from models import Book
from shelflife_app.image_utils import process_image
from shelflife_app.services import get_book_service, get_database
from shelflife_app.ui_helpers import render_page_header
from validation import sanitize_string, validate_book_input

logger = get_logger(__name__)

db = get_database()
book_service = get_book_service()


render_page_header(
    "Add New Book",
    "Enter basic details and let AI enhance your book's metadata",
    "➕"
)

if st.session_state.get("show_success"):
    st.success(st.session_state.show_success)
    st.session_state.show_success = None

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("#### Book Details")

    with st.form("book_form", clear_on_submit=True):
        st.markdown("**Required Information**")
        title = st.text_input("Title", placeholder="Enter the book title", help="The full title of the book")
        author = st.text_input("Author", placeholder="Enter the author's name", help="The primary author of the book")

        st.markdown("---")
        st.markdown("**Optional Information**")

        col_a, col_b = st.columns(2)
        with col_a:
            year = st.number_input(
                "Publication Year",
                min_value=0,
                max_value=datetime.now().year,
                value=None,
                placeholder="e.g., 1984",
                help="Year the book was published"
            )
            year = year if year != 0 else None

            condition = st.selectbox(
                "Condition",
                ["New", "Like New", "Very Good", "Good", "Fair", "Poor"],
                help="Physical condition of your copy"
            )

        with col_b:
            isbn = st.text_input(
                "ISBN",
                placeholder="e.g., 978-0-123456-78-9",
                help="10 or 13 digit ISBN (optional)"
            )
            publisher = st.text_input(
                "Publisher",
                placeholder="e.g., Penguin Books",
                help="Publishing house (optional)"
            )

        st.markdown("---")

        cover_image = st.file_uploader(
            "Cover Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a photo of the book cover (optional)"
        )

        personal_notes = st.text_area(
            "Personal Notes",
            placeholder="Add your thoughts, reading status, annotations, or any personal notes about this book...",
            help="These notes are private and won't be used in analytics or summaries",
            height=100
        )

        submitted = st.form_submit_button(
            "✨ Add Book & Enhance Metadata",
            use_container_width=True,
            type="primary"
        )

        if submitted:
            is_valid, validation_errors = validate_book_input(
                title=title, author=author, year=year, isbn=isbn,
                publisher=publisher, condition=condition, personal_notes=personal_notes
            )

            if not is_valid:
                st.error("**Please fix the following errors:**")
                for error in validation_errors:
                    st.error(f"• {error}")
                st.stop()

            try:
                title = sanitize_string(title, 500)
                author = sanitize_string(author, 200)
                if publisher:
                    publisher = sanitize_string(publisher, 200)

                progress_bar = st.progress(0, text="Enhancing book metadata with AI...")
                progress_bar.progress(20, text="Searching book databases...")
                progress_bar.progress(40, text="Analyzing with AI...")

                enhanced_metadata = book_service.enhance_book_data(title, author, year, isbn)

                progress_bar.progress(70, text="Processing cover image...")

                if enhanced_metadata:
                    image_data = process_image(cover_image) if cover_image else None

                    progress_bar.progress(85, text="Saving to database...")

                    book = Book(
                        title=title, author=author, year=year, isbn=isbn,
                        publisher=publisher, condition=condition,
                        cover_image=image_data, metadata=enhanced_metadata,
                        personal_notes=personal_notes
                    )

                    book_id = db.add_book(book)
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress_bar.empty()

                    st.session_state.show_success = (
                        f"'{title}' by {author} has been added to your library! (ID: {book_id})"
                    )
                    logger.info(f"Added book: {title} by {author} (ID: {book_id})")
                    st.rerun()
                else:
                    progress_bar.empty()
                    st.error("Failed to fetch book information. Please try again.")

            except ValueError as e:
                st.error(f"Validation error: {str(e)}")
            except DatabaseError as e:
                st.error(f"Database error: {str(e)}")
            except Exception as e:
                logger.error(f"Error adding book: {str(e)}", exc_info=True)
                st.error("An unexpected error occurred. Please try again.")

with col2:
    st.markdown("#### How It Works")
    st.markdown('''
        <div style="background: #F3F4F6; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #6366F1; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 0.875rem;">1</span>
                <div>
                    <strong>Enter Basic Info</strong>
                    <p style="margin: 0; font-size: 0.875rem; color: #6B7280;">Just title and author are required</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #6366F1; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 0.875rem;">2</span>
                <div>
                    <strong>AI Enhancement</strong>
                    <p style="margin: 0; font-size: 0.875rem; color: #6B7280;">We fetch synopsis, themes, genres & more</p>
                </div>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background: #6366F1; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 0.875rem;">3</span>
                <div>
                    <strong>Discover Connections</strong>
                    <p style="margin: 0; font-size: 0.875rem; color: #6B7280;">Find related books & explore themes</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("#### Data Sources")
    st.markdown('''
        - **Google Books API** - Cover images, publication info
        - **Open Library** - Additional metadata
        - **AI Analysis** - Synopsis, themes, genres, historical context
    ''')
