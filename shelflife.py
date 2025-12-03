"""
ShelfLife - Intelligent Library Cataloging System
Enhanced UI/UX version with modern design patterns.
"""
import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Import our modules
import config
from database import Database, DatabaseError
from book_service import BookService, extract_and_save_themes
from models import Book, BookMetadata
from analytics import (
    generate_analytics,
    create_book_network,
    visualize_book_network,
    export_library_to_csv
)
from api_utils import test_api_connection
from logger import get_logger, ShelfLifeLogger
from constants import GENRE_CATEGORIES
from validation import validate_book_input, validate_search_term, sanitize_string

# Initialize logger
# Use getattr with default for backward compatibility with older config.py files
ShelfLifeLogger().set_level(getattr(config, 'LOG_LEVEL', 'INFO'))
logger = get_logger(__name__)

# Custom color palette for charts
CHART_COLORS = [
    '#6366F1',  # Primary indigo
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
    '#F59E0B',  # Amber
    '#10B981',  # Emerald
    '#3B82F6',  # Blue
    '#EF4444',  # Red
    '#14B8A6',  # Teal
    '#F97316',  # Orange
    '#84CC16',  # Lime
]

# Navigation items with icons
NAV_ITEMS = [
    {"icon": "plus-circle", "label": "Add Book", "emoji": "➕"},
    {"icon": "library", "label": "View Collection", "emoji": "📚"},
    {"icon": "bar-chart", "label": "Analytics", "emoji": "📊"},
    {"icon": "share-2", "label": "Network View", "emoji": "🔗"},
    {"icon": "file-text", "label": "Executive Summary", "emoji": "📋"},
    {"icon": "message-circle", "label": "Ask the Library", "emoji": "💬"},
]

# Condition badge mapping
CONDITION_BADGES = {
    "New": ("badge-new", "Mint"),
    "Like New": ("badge-new", "Like New"),
    "Very Good": ("badge-good", "Very Good"),
    "Good": ("badge-good", "Good"),
    "Fair": ("badge-fair", "Fair"),
    "Poor": ("badge-poor", "Poor"),
}


# Initialize services (singleton pattern using session state)
@st.cache_resource
def get_database():
    """Get or create database instance."""
    try:
        return Database(config.DB_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        st.error(f"Database initialization failed: {str(e)}")
        st.stop()


@st.cache_resource
def get_book_service():
    """Get or create book service instance."""
    try:
        return BookService()
    except Exception as e:
        logger.error(f"Failed to initialize book service: {str(e)}", exc_info=True)
        st.error(f"Book service initialization failed. Check your LLM configuration: {str(e)}")
        st.stop()


# ==================== UI HELPER FUNCTIONS ====================

def render_page_header(title: str, subtitle: str = None, icon: str = None):
    """Render a styled page header."""
    icon_html = f'<span style="margin-right: 12px;">{icon}</span>' if icon else ''
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ''

    st.markdown(f'''
        <div class="app-header">
            <h1>{icon_html}{title}</h1>
            {subtitle_html}
        </div>
    ''', unsafe_allow_html=True)


def render_empty_state(icon: str, title: str, message: str, show_button: bool = False, button_label: str = ""):
    """Render an empty state component."""
    st.markdown(f'''
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{message}</p>
        </div>
    ''', unsafe_allow_html=True)

    if show_button:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.button(button_label, use_container_width=True)
    return False


def render_condition_badge(condition: str) -> str:
    """Return HTML for a condition badge."""
    badge_class, label = CONDITION_BADGES.get(condition, ("badge-fair", condition))
    return f'<span class="badge {badge_class}">{label}</span>'


def render_tags(items: list, tag_class: str = "tag") -> str:
    """Render a list of items as tags."""
    if not items:
        return ""
    return " ".join([f'<span class="{tag_class}">{item}</span>' for item in items])


def simulate_progress(message: str, duration: float = 2.0):
    """Show a progress bar that simulates work being done."""
    progress_bar = st.progress(0, text=message)
    steps = 20
    for i in range(steps):
        time.sleep(duration / steps)
        progress_bar.progress((i + 1) / steps, text=message)
    progress_bar.empty()


# Image processing helper
def process_image(uploaded_file):
    """Process and resize uploaded images."""
    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file)

        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        format_map = {
            'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
            'gif': 'GIF', 'bmp': 'BMP', 'webp': 'WEBP'
        }

        image_format = format_map.get(file_extension, 'JPEG')

        # Resize if too large
        if max(image.size) > config.MAX_IMAGE_SIZE:
            ratio = config.MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image_format)
        return img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        st.error("Error processing image. Please ensure it's a valid image file.")
        return None


# Helper functions for library operations
def generate_library_json(db: Database):
    """Generate a simple JSON file with book titles and authors."""
    try:
        books = db.search_books()  # Get all books

        library_data = {
            "library": [
                {"title": book[1], "author": book[2]}
                for book in books
            ],
            "generated_at": datetime.now().isoformat()
        }

        # Save to data folder
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        json_path = data_dir / "library_catalog.json"

        with open(json_path, "w") as f:
            json.dump(library_data, f, indent=2)

        logger.info(f"Generated library catalog with {len(books)} books")
        return json_path, library_data

    except Exception as e:
        logger.error(f"Error generating library JSON: {str(e)}", exc_info=True)
        raise


def find_related_books(db: Database, current_book_id: int, metadata: dict):
    """Find up to 3 related books based on shared genres."""
    try:
        all_books = db.search_books()
        current_genres = set(metadata.get('genre', []))

        related_books = []
        for book in all_books:
            if book[0] == current_book_id:  # Skip current book
                continue

            if book[8]:  # Has metadata
                try:
                    other_metadata = json.loads(book[8])
                    other_genres = set(other_metadata.get('genre', []))
                    shared_genres = current_genres & other_genres

                    if shared_genres:
                        related_books.append({
                            'id': book[0],
                            'title': book[1],
                            'author': book[2],
                            'shared_genres': shared_genres,
                            'genre_count': len(shared_genres)
                        })
                except json.JSONDecodeError:
                    continue

        # Sort by shared genre count and return top 3
        related_books.sort(key=lambda x: x['genre_count'], reverse=True)
        return related_books[:3]

    except Exception as e:
        logger.error(f"Error finding related books: {str(e)}", exc_info=True)
        return []


def display_theme_analysis(theme_analysis: dict):
    """Display the theme analysis in an organized way."""
    if "analysis" in theme_analysis:
        st.markdown("### Thematic Overview")
        st.write(theme_analysis['analysis'].get('summary', ''))

        if theme_analysis['analysis'].get('key_insights'):
            st.markdown("#### Key Insights")
            for insight in theme_analysis['analysis']['key_insights']:
                st.markdown(f"- {insight}")

    st.markdown("### Thematic Groups")

    theme_names = [theme['name'] for theme in theme_analysis.get('uber_themes', [])]
    if not theme_names:
        st.info("No theme groups available yet.")
        return

    selected_theme = st.selectbox(
        "Select a thematic group to explore",
        theme_names
    )

    if selected_theme:
        theme_details = next(
            (theme for theme in theme_analysis['uber_themes']
             if theme['name'] == selected_theme),
            None
        )

        if theme_details:
            st.write(f"**Description:** {theme_details['description']}")

            sub_themes_data = []
            for theme in theme_details.get('sub_themes', []):
                sub_themes_data.append({
                    'Theme': theme['name'],
                    'Connection': theme['connection']
                })

            if sub_themes_data:
                st.write("**Related Themes:**")
                df = pd.DataFrame(sub_themes_data)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point."""

    # Set page configuration
    st.set_page_config(
        page_title="ShelfLife",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Load CSS
    def load_css(file_name):
        css_path = Path('static') / file_name
        if css_path.exists():
            with open(css_path) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            logger.warning(f"CSS file not found: {css_path}")

    load_css('styles.css')

    # Initialize services
    db = get_database()
    book_service = get_book_service()

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Add Book"
    if 'delete_confirm' not in st.session_state:
        st.session_state.delete_confirm = None
    if 'show_success' not in st.session_state:
        st.session_state.show_success = None

    # Sidebar
    with st.sidebar:
        # Logo and title
        st.markdown('''
            <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📚</div>
                <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700;">ShelfLife</h1>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.8;">Intelligent Library Cataloging</p>
            </div>
        ''', unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown('<p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 0.75rem;">Navigation</p>', unsafe_allow_html=True)

        for item in NAV_ITEMS:
            is_active = st.session_state.current_page == item["label"]
            button_label = f"{item['emoji']}  {item['label']}"

            if st.button(
                button_label,
                key=f"nav_{item['label']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = item["label"]
                st.rerun()

        st.divider()

        # API Status Section
        st.markdown('<p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 0.75rem;">System Status</p>', unsafe_allow_html=True)

        if st.button("🔍 Check API Status", use_container_width=True):
            with st.spinner("Checking connections..."):
                # Check LLM
                llm_status = book_service.test_connection()
                st.write(f"{config.LLM_PROVIDER.title()} LLM:",
                        "✅" if llm_status["success"] else "❌")

                # Check Google Books API
                google_status = test_api_connection("google_books")
                st.write("Google Books API:",
                        "✅" if google_status["success"] else "❌")

                # Check Open Library API
                ol_status = test_api_connection("open_library")
                st.write("Open Library API:",
                        "✅" if ol_status["success"] else "❌")

                if config.DEBUG_MODE:
                    if not llm_status["success"]:
                        st.error(f"LLM Error: {llm_status.get('error', 'Unknown')}")
                    if not google_status["success"]:
                        st.error(f"Google Books: {google_status.get('error', 'Unknown')}")
                    if not ol_status["success"]:
                        st.error(f"Open Library: {ol_status.get('error', 'Unknown')}")

        # Library stats
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

    # Main content area
    page = st.session_state.current_page

    if page == "Add Book":
        render_add_book_page(db, book_service)
    elif page == "View Collection":
        render_view_collection_page(db, book_service)
    elif page == "Analytics":
        render_analytics_page(db, book_service)
    elif page == "Network View":
        render_network_view_page(db)
    elif page == "Executive Summary":
        render_executive_summary_page(db, book_service)
    elif page == "Ask the Library":
        render_ask_library_page(db, book_service)


# ==================== PAGE RENDERERS ====================

def render_add_book_page(db: Database, book_service: BookService):
    """Render the Add Book page."""
    render_page_header(
        "Add New Book",
        "Enter basic details and let AI enhance your book's metadata",
        "➕"
    )

    # Show success message if set
    if st.session_state.show_success:
        st.success(st.session_state.show_success)
        st.session_state.show_success = None

    # Two-column layout for the form
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Book Details")

        with st.form("book_form", clear_on_submit=True):
            # Required fields
            st.markdown("**Required Information**")
            title = st.text_input(
                "Title",
                placeholder="Enter the book title",
                help="The full title of the book"
            )
            author = st.text_input(
                "Author",
                placeholder="Enter the author's name",
                help="The primary author of the book"
            )

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

            # Submit button with better styling
            submitted = st.form_submit_button(
                "✨ Add Book & Enhance Metadata",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # Validate inputs
                is_valid, validation_errors = validate_book_input(
                    title=title,
                    author=author,
                    year=year,
                    isbn=isbn,
                    publisher=publisher,
                    condition=condition,
                    personal_notes=personal_notes
                )

                if not is_valid:
                    st.error("**Please fix the following errors:**")
                    for error in validation_errors:
                        st.error(f"• {error}")
                    return

                try:
                    # Sanitize inputs
                    title = sanitize_string(title, 500)
                    author = sanitize_string(author, 200)
                    if publisher:
                        publisher = sanitize_string(publisher, 200)

                    # Show progress
                    progress_text = "Enhancing book metadata with AI..."
                    progress_bar = st.progress(0, text=progress_text)

                    # Step 1: Fetching from APIs
                    progress_bar.progress(20, text="Searching book databases...")

                    # Step 2: AI Enhancement
                    progress_bar.progress(40, text="Analyzing with AI...")
                    enhanced_metadata = book_service.enhance_book_data(title, author, year, isbn)

                    progress_bar.progress(70, text="Processing cover image...")

                    if enhanced_metadata:
                        # Process cover image
                        image_data = process_image(cover_image) if cover_image else None

                        progress_bar.progress(85, text="Saving to database...")

                        # Create book object
                        book = Book(
                            title=title,
                            author=author,
                            year=year,
                            isbn=isbn,
                            publisher=publisher,
                            condition=condition,
                            cover_image=image_data,
                            metadata=enhanced_metadata,
                            personal_notes=personal_notes
                        )

                        # Add to database
                        book_id = db.add_book(book)
                        progress_bar.progress(100, text="Complete!")
                        time.sleep(0.5)
                        progress_bar.empty()

                        st.session_state.show_success = f"'{title}' by {author} has been added to your library! (ID: {book_id})"
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


def render_view_collection_page(db: Database, book_service: BookService):
    """Render the View Collection page."""
    render_page_header(
        "Your Library",
        "Browse, search, and manage your book collection",
        "📚"
    )

    # Top action bar
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            try:
                books = db.get_all_books()
                csv_path = export_library_to_csv(books)
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name="library_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                logger.error(f"Error generating CSV: {str(e)}", exc_info=True)
                st.error("Error generating CSV file")

    with col2:
        search = st.text_input(
            "Search",
            placeholder="🔍 Search by title, author, or genre...",
            label_visibility="collapsed"
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Recent", "Title", "Author", "Year"],
            label_visibility="collapsed"
        )

    # Validate search term
    if search:
        is_valid, error = validate_search_term(search)
        if not is_valid:
            st.error(f"Invalid search term: {error}")
            return
        search = sanitize_string(search, 200)

    st.markdown("---")

    # Get books
    try:
        books = db.search_books(search, sort_by)

        if not books:
            if search:
                render_empty_state(
                    "🔍",
                    "No Results Found",
                    f"No books match '{search}'. Try a different search term."
                )
            else:
                if render_empty_state(
                    "📚",
                    "Your Library is Empty",
                    "Start building your collection by adding your first book!",
                    show_button=True,
                    button_label="➕ Add Your First Book"
                ):
                    st.session_state.current_page = "Add Book"
                    st.rerun()
            return

        # Show result count
        st.markdown(f"**{len(books)}** book{'s' if len(books) != 1 else ''} in your library")

        # Display books
        for book in books:
            book_id = book[0]

            # Check for delete confirmation
            if st.session_state.delete_confirm == book_id:
                st.warning(f"⚠️ **Delete '{book[1]}'?** This action cannot be undone.")
                col_yes, col_no, col_spacer = st.columns([1, 1, 3])
                with col_yes:
                    if st.button("🗑️ Yes, Delete", key=f"confirm_del_{book_id}", type="primary"):
                        try:
                            db.delete_book(book_id)
                            st.session_state.delete_confirm = None
                            st.success(f"'{book[1]}' has been deleted.")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting book: {str(e)}")
                with col_no:
                    if st.button("Cancel", key=f"cancel_del_{book_id}"):
                        st.session_state.delete_confirm = None
                        st.rerun()
                continue

            with st.expander(f"**{book[1]}** by {book[2]}", expanded=False):
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Display cover image
                    if book[7]:  # Local cover image
                        try:
                            st.image(book[7], use_container_width=True)
                        except Exception as e:
                            logger.warning(f"Error displaying cover image: {str(e)}")
                            st.markdown('<div style="background: #F3F4F6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #9CA3AF;">No Cover</div>', unsafe_allow_html=True)
                    elif book[8]:  # Check metadata for cover URL
                        try:
                            metadata = json.loads(book[8])
                            if 'cover_url' in metadata and metadata['cover_url']:
                                st.image(metadata['cover_url'], use_container_width=True)
                            else:
                                st.markdown('<div style="background: #F3F4F6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #9CA3AF;">No Cover</div>', unsafe_allow_html=True)
                        except Exception:
                            st.markdown('<div style="background: #F3F4F6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #9CA3AF;">No Cover</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background: #F3F4F6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #9CA3AF;">No Cover</div>', unsafe_allow_html=True)

                    # Condition badge
                    condition = book[6] if book[6] else "Unknown"
                    st.markdown(f"**Condition:** {render_condition_badge(condition)}", unsafe_allow_html=True)

                    # Basic information
                    st.markdown("---")
                    st.markdown(f"**Year:** {book[3] if book[3] else 'Unknown'}")
                    st.markdown(f"**ISBN:** {book[4] if book[4] else 'N/A'}")
                    st.markdown(f"**Publisher:** {book[5] if book[5] else 'Unknown'}")
                    st.markdown(f"**Added:** {book[9][:10] if book[9] else 'N/A'}")

                with col2:
                    if book[8]:
                        metadata = json.loads(book[8])

                        # Personal Notes (highlighted)
                        if len(book) > 11 and book[11]:
                            st.markdown("##### 📝 Personal Notes")
                            st.info(book[11])

                        # Synopsis
                        if metadata.get('synopsis'):
                            st.markdown("##### 📖 Synopsis")
                            st.write(metadata['synopsis'])

                        # Genres as tags
                        if metadata.get('genre'):
                            st.markdown("##### 🏷️ Genres")
                            genre_tags = " ".join([f'`{g}`' for g in metadata['genre']])
                            st.markdown(genre_tags)

                        # Themes as tags
                        if metadata.get('themes'):
                            st.markdown("##### 🎭 Themes")
                            theme_tags = " ".join([f'`{t}`' for t in metadata['themes']])
                            st.markdown(theme_tags)

                        # Historical Context
                        if metadata.get('historical_context'):
                            st.markdown("##### 🏛️ Historical Context")
                            st.write(metadata['historical_context'])

                        # Related works
                        if metadata.get('related_works'):
                            st.markdown("##### 📚 Related Books (AI Suggestions)")
                            for work in metadata['related_works'][:3]:
                                if isinstance(work, dict):
                                    st.markdown(f"- **{work.get('title', '')}** by {work.get('author', '')}")
                                    if work.get('reason'):
                                        st.caption(f"  _{work['reason']}_")
                                else:
                                    st.markdown(f"- {work}")

                        # Related books from collection
                        related_books = find_related_books(db, book[0], metadata)
                        if related_books:
                            st.markdown("##### 🔗 Similar in Your Collection")
                            for related in related_books:
                                st.markdown(f"- **{related['title']}** by {related['author']}")

                        # Sources
                        if "sources" in metadata:
                            st.caption(f"Data sources: {', '.join(metadata['sources'])}")

                # Action buttons
                st.markdown("---")
                col3, col4, col5 = st.columns(3)

                with col3:
                    if st.button("🗑️ Delete", key=f"del_{book[0]}", use_container_width=True):
                        st.session_state.delete_confirm = book[0]
                        st.rerun()

                with col5:
                    if st.button("🔄 Refresh Metadata", key=f"refresh_{book[0]}", use_container_width=True):
                        try:
                            progress = st.progress(0, text="Refreshing metadata...")
                            progress.progress(30, text="Fetching latest data...")

                            enhanced_metadata = book_service.enhance_book_data(
                                book[1], book[2], book[3], book[4]
                            )

                            progress.progress(70, text="Updating database...")

                            if enhanced_metadata:
                                db.update_metadata(book[0], enhanced_metadata)
                                progress.progress(100, text="Complete!")
                                time.sleep(0.3)
                                progress.empty()
                                st.success("Metadata refreshed successfully!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                progress.empty()
                                st.error("Failed to refresh metadata")
                        except Exception as e:
                            logger.error(f"Error refreshing metadata: {str(e)}", exc_info=True)
                            st.error("Failed to refresh metadata")

    except DatabaseError as e:
        st.error(f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error rendering collection: {str(e)}", exc_info=True)
        st.error("An error occurred while loading your collection")


def render_analytics_page(db: Database, book_service: BookService):
    """Render the Analytics page."""
    render_page_header(
        "Library Analytics",
        "Discover patterns and insights in your collection",
        "📊"
    )

    # Get analytics data
    try:
        books = db.get_all_books()

        if not books:
            render_empty_state(
                "📊",
                "No Data to Analyze",
                "Add some books to your library to see analytics and insights."
            )
            return

        stats, genre_counts, theme_counts = generate_analytics(books)

        # Metrics row with styled cards
        st.markdown("### Quick Stats")
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Books", stats.total_books)
        col2.metric("Authors", stats.unique_authors)
        col3.metric("Avg. Year", stats.avg_pub_year or "N/A")

        if stats.common_time_period:
            col4.metric(
                "Era",
                stats.common_time_period,
                f"{round(stats.time_period_coverage * 100)}%"
            )
        else:
            col4.metric("Era", "N/A")

        fiction_pct = round(stats.fiction_ratio * 100)
        nonfiction_pct = round(stats.nonfiction_ratio * 100)
        col5.metric("Fiction", f"{fiction_pct}%", f"{nonfiction_pct}% Non-Fiction")

        st.markdown("---")

        # Genre distribution
        if not genre_counts.empty:
            st.markdown("### Genre Distribution")

            tab1, tab2, tab3 = st.tabs(["📊 Combined View", "📖 Fiction", "📚 Non-Fiction"])

            with tab1:
                fig = px.sunburst(
                    genre_counts,
                    path=['category', 'genre'],
                    values='count',
                    color='count',
                    color_continuous_scale=['#E0E7FF', '#6366F1', '#312E81']
                )
                fig.update_layout(
                    margin=dict(t=30, l=0, r=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fiction_data = genre_counts[genre_counts['category'] == 'Fiction']
                if not fiction_data.empty:
                    fig_fiction = px.pie(
                        fiction_data,
                        values='count',
                        names='genre',
                        color_discrete_sequence=CHART_COLORS
                    )
                    fig_fiction.update_layout(
                        margin=dict(t=30, l=0, r=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter"),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                    )
                    fig_fiction.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_fiction, use_container_width=True)
                else:
                    st.info("No fiction books in your collection yet.")

            with tab3:
                nonfiction_data = genre_counts[genre_counts['category'] == 'Non-Fiction']
                if not nonfiction_data.empty:
                    fig_nonfiction = px.pie(
                        nonfiction_data,
                        values='count',
                        names='genre',
                        color_discrete_sequence=CHART_COLORS
                    )
                    fig_nonfiction.update_layout(
                        margin=dict(t=30, l=0, r=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter"),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                    )
                    fig_nonfiction.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_nonfiction, use_container_width=True)
                else:
                    st.info("No non-fiction books in your collection yet.")

        # Theme distribution
        if not theme_counts.empty:
            st.markdown("### Theme Distribution")
            fig_themes = px.treemap(
                theme_counts,
                path=['theme'],
                values='count',
                color='count',
                color_continuous_scale=['#DBEAFE', '#3B82F6', '#1E40AF']
            )
            fig_themes.update_layout(
                margin=dict(t=30, l=0, r=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_themes, use_container_width=True)

        st.markdown("---")

        # Theme Analysis section
        st.markdown("### Theme Analysis")
        st.write("Extract and analyze thematic patterns across your library.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Extract Themes", use_container_width=True):
                try:
                    progress = st.progress(0, text="Extracting themes...")
                    progress.progress(30, text="Analyzing book metadata...")

                    books_with_meta = db.get_books_with_metadata()
                    theme_data = extract_and_save_themes(books_with_meta)

                    progress.progress(100, text="Complete!")
                    time.sleep(0.3)
                    progress.empty()

                    st.success(f"Extracted {len(theme_data['themes'])} unique themes!")

                    with open("data/theme_inventory.json", "r") as f:
                        st.download_button(
                            "⬇️ Download Theme Inventory",
                            f.read(),
                            "theme_inventory.json",
                            "application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    logger.error(f"Error extracting themes: {str(e)}", exc_info=True)
                    st.error("Failed to extract themes")

        with col2:
            if st.button("🧠 Analyze Groupings", use_container_width=True):
                try:
                    with open("data/theme_inventory.json", "r") as f:
                        theme_data = json.load(f)

                    progress = st.progress(0, text="Analyzing theme relationships...")
                    progress.progress(50, text="AI is identifying patterns...")

                    theme_analysis = book_service.analyze_themes(theme_data['themes'])

                    if theme_analysis:
                        progress.progress(90, text="Saving analysis...")
                        with open("data/theme_analysis.json", "w") as f:
                            json.dump(theme_analysis, f, indent=2)
                        progress.progress(100, text="Complete!")
                        time.sleep(0.3)
                        progress.empty()
                        st.success("Theme analysis complete!")
                        st.rerun()
                    else:
                        progress.empty()
                        st.error("Failed to analyze themes")
                except FileNotFoundError:
                    st.error("Please extract themes first")
                except Exception as e:
                    logger.error(f"Error analyzing themes: {str(e)}", exc_info=True)
                    st.error("Failed to analyze themes")

        # Display existing theme analysis
        try:
            with open("data/theme_analysis.json", "r") as f:
                theme_analysis = json.load(f)
                st.markdown("---")
                display_theme_analysis(theme_analysis)
        except FileNotFoundError:
            st.info("💡 Click 'Extract Themes' to discover thematic patterns in your library.")
        except Exception as e:
            logger.error(f"Error loading theme analysis: {str(e)}")

    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}", exc_info=True)
        st.error("Error generating analytics")


def render_network_view_page(db: Database):
    """Render the Network View page."""
    render_page_header(
        "Book Relationships",
        "Visualize connections between books in your collection",
        "🔗"
    )

    # Legend
    st.markdown('''
        <div style="background: #F3F4F6; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
            <strong>Connection Types:</strong>
            <span style="margin-left: 1.5rem;">🔵 Same Author</span>
            <span style="margin-left: 1rem;">🟢 Same Decade</span>
            <span style="margin-left: 1rem;">🟠 Shared Themes</span>
        </div>
    ''', unsafe_allow_html=True)

    view_type = st.radio(
        "Filter",
        ["All Books", "Fiction Only", "Non-Fiction Only"],
        horizontal=True,
        label_visibility="collapsed"
    )

    try:
        books = db.get_all_books()

        if not books or len(books) < 2:
            render_empty_state(
                "🔗",
                "Not Enough Books",
                "Add at least 2 books to see their relationships visualized."
            )
            return

        with st.spinner("Generating network visualization..."):
            category = None
            if view_type == "Fiction Only":
                category = "Fiction"
            elif view_type == "Non-Fiction Only":
                category = "Non-Fiction"

            G = create_book_network(books, category)
            fig = visualize_book_network(G)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Network statistics
                st.markdown("### Network Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Books", len(G.nodes()))
                with col2:
                    st.metric("Connections", len(G.edges()))
                with col3:
                    if len(G.nodes()) > 0:
                        import networkx as nx
                        avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
            else:
                render_empty_state(
                    "🔗",
                    "No Connections Found",
                    "Your books don't share enough common attributes yet. Add more books or refresh metadata."
                )

    except Exception as e:
        logger.error(f"Error creating network view: {str(e)}", exc_info=True)
        st.error("Error generating network visualization")


def render_executive_summary_page(db: Database, book_service: BookService):
    """Render the Executive Summary page."""
    render_page_header(
        "Executive Summary",
        "Get a high-level overview of your library",
        "📋"
    )

    # Quick stats
    try:
        books = db.get_all_books()

        if not books:
            render_empty_state(
                "📋",
                "No Books Yet",
                "Add books to your library to generate an executive summary."
            )
            return

        stats, genre_counts, _ = generate_analytics(books)

        st.markdown("### Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Books", stats.total_books)
        with col2:
            st.metric("Unique Authors", stats.unique_authors)
        with col3:
            st.metric("Avg. Year", stats.avg_pub_year if stats.avg_pub_year else "N/A")
        with col4:
            fiction_pct = round(stats.fiction_ratio * 100)
            nonfiction_pct = round(stats.nonfiction_ratio * 100)
            st.metric("Fiction/Non-Fiction", f"{fiction_pct}% / {nonfiction_pct}%")

        st.markdown("---")
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")

    # Catalog and summary generation
    st.markdown("### Generate Reports")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
            <div style="background: #F3F4F6; border-radius: 12px; padding: 1.5rem; height: 100%;">
                <h4 style="margin-top: 0;">📄 Library Catalog</h4>
                <p style="color: #6B7280; font-size: 0.875rem;">Generate a JSON file with all your books for backup or analysis.</p>
            </div>
        ''', unsafe_allow_html=True)

        if st.button("Generate Catalog", use_container_width=True):
            try:
                progress = st.progress(0, text="Generating catalog...")
                progress.progress(50, text="Compiling book data...")

                json_path, library_data = generate_library_json(db)

                progress.progress(100, text="Complete!")
                time.sleep(0.3)
                progress.empty()

                st.success(f"Catalog generated with {len(library_data['library'])} books!")

                with open(json_path, "r") as f:
                    st.download_button(
                        "⬇️ Download Catalog",
                        f.read(),
                        "library_catalog.json",
                        "application/json",
                        use_container_width=True
                    )
            except Exception as e:
                logger.error(f"Error generating catalog: {str(e)}", exc_info=True)
                st.error("Failed to generate catalog")

    with col2:
        st.markdown('''
            <div style="background: #F3F4F6; border-radius: 12px; padding: 1.5rem; height: 100%;">
                <h4 style="margin-top: 0;">🤖 AI Summary</h4>
                <p style="color: #6B7280; font-size: 0.875rem;">Get AI-generated insights, patterns, and recommendations.</p>
            </div>
        ''', unsafe_allow_html=True)

        if st.button("Generate Summary", use_container_width=True):
            try:
                with open("data/library_catalog.json", "r") as f:
                    library_data = json.load(f)

                progress = st.progress(0, text="Analyzing library...")
                progress.progress(30, text="AI is reviewing your collection...")

                summary = book_service.generate_executive_summary(library_data)

                progress.progress(80, text="Compiling insights...")

                if summary:
                    summary_info = {
                        "last_updated": datetime.now().isoformat(),
                        "summary": summary
                    }
                    with open("data/executive_summary.json", "w") as f:
                        json.dump(summary_info, f, indent=2)

                    progress.progress(100, text="Complete!")
                    time.sleep(0.3)
                    progress.empty()
                    st.success("Summary generated!")
                    st.rerun()
                else:
                    progress.empty()
                    st.error("Failed to generate summary")
            except FileNotFoundError:
                st.error("Please generate library catalog first")
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}", exc_info=True)
                st.error("Failed to generate summary")

    st.markdown("---")

    # Display existing summary
    try:
        with open("data/executive_summary.json", "r") as f:
            summary_data = json.load(f)

        st.markdown("### Collection Summary")
        st.markdown(f'''
            <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                {summary_data["summary"]["summary"]}
            </div>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Key Patterns")
            for pattern in summary_data["summary"]["patterns"]:
                st.markdown(f"- {pattern}")

        with col2:
            st.markdown("### Recommendations")
            for rec in summary_data["summary"]["recommendations"]:
                st.markdown(f"- {rec}")

        st.caption(f"Last updated: {summary_data['last_updated'][:10]}")

    except FileNotFoundError:
        st.info("💡 Generate a catalog first, then create an AI summary to see insights about your library.")
    except Exception as e:
        logger.error(f"Error displaying summary: {str(e)}")


def render_ask_library_page(db: Database, book_service: BookService):
    """Render the Ask the Library page."""
    render_page_header(
        "Ask the Library",
        "Have a conversation with your book collection",
        "💬"
    )

    # Example questions
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
            return

        try:
            # Check if catalog exists
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the logs and restart the application.")
