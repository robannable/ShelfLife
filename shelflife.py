"""
ShelfLife - Intelligent Library Cataloging System
Refactored version with modular architecture and improved error handling.
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

# Initialize logger
ShelfLifeLogger().set_level(config.LOG_LEVEL)
logger = get_logger(__name__)

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
        st.write("### 📚 Thematic Overview")
        st.write(theme_analysis['analysis'].get('summary', ''))

        if theme_analysis['analysis'].get('key_insights'):
            st.write("#### Key Insights")
            for insight in theme_analysis['analysis']['key_insights']:
                st.markdown(f"• {insight}")

    st.write("### 🎯 Thematic Groups")

    theme_names = [theme['name'] for theme in theme_analysis.get('uber_themes', [])]
    if not theme_names:
        st.info("No theme groups available.")
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
                st.dataframe(df, use_container_width=True)

# Main application
def main():
    """Main application entry point."""

    # Set page configuration
    st.set_page_config(
        page_title="ShelfLife",
        page_icon="📚",
        layout="centered"
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

    # Sidebar
    with st.sidebar:
        st.title("ShelfLife 📚")

        # API Status Section
        st.subheader("API Status")
        if st.button("Check API Status"):
            with st.spinner("Checking API connections..."):
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

        # Navigation
        st.subheader("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Add Book", "View Collection", "Analytics", "Network View",
             "Executive Summary", "Ask the Library"]
        )

    # Main content area
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

def render_add_book_page(db: Database, book_service: BookService):
    """Render the Add Book page."""
    st.header("Add New Book")

    with st.form("book_form"):
        title = st.text_input("Title*")
        author = st.text_input("Author*")
        year = st.number_input("Year",
            min_value=0,
            max_value=datetime.now().year,
            value=None)
        year = year if year != 0 else None
        isbn = st.text_input("ISBN (optional)")
        publisher = st.text_input("Publisher (optional)")
        condition = st.selectbox("Condition",
            ["New", "Like New", "Very Good", "Good", "Fair", "Poor"])
        cover_image = st.file_uploader("Cover Image", type=['png', 'jpg', 'jpeg'])

        personal_notes = st.text_area(
            "Personal Notes",
            placeholder="Add your personal thoughts, reading status, annotations, or any other notes about this book...",
            help="These notes are private and won't be used in analytics or summaries"
        )

        submitted = st.form_submit_button("Add Book")

        if submitted and title and author:
            try:
                with st.spinner("Fetching book information..."):
                    # Enhance book data with LLM
                    enhanced_metadata = book_service.enhance_book_data(title, author, year, isbn)

                if enhanced_metadata:
                    # Process cover image
                    image_data = process_image(cover_image) if cover_image else None

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
                    st.success(f"Book added successfully! (ID: {book_id})")
                    logger.info(f"Added book: {title} by {author} (ID: {book_id})")
                else:
                    st.error("Failed to fetch book information. Please try again.")

            except ValueError as e:
                st.error(f"Validation error: {str(e)}")
            except DatabaseError as e:
                st.error(f"Database error: {str(e)}")
            except Exception as e:
                logger.error(f"Error adding book: {str(e)}", exc_info=True)
                st.error("An unexpected error occurred. Please try again.")

def render_view_collection_page(db: Database, book_service: BookService):
    """Render the View Collection page."""
    st.header("Your Library")

    # Export button
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📥 Export to CSV"):
                try:
                    books = db.get_all_books()
                    csv_path = export_library_to_csv(books)
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        csv_data = f.read()
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="library_export.csv",
                            mime="text/csv"
                        )
                    st.success("CSV file generated successfully!")
                except Exception as e:
                    logger.error(f"Error generating CSV: {str(e)}", exc_info=True)
                    st.error("Error generating CSV file")

    # Search and filter options
    search = st.text_input("Search books", "")
    sort_by = st.selectbox("Sort by", ["Title", "Author", "Year", "Recent"])

    # Get books
    try:
        books = db.search_books(search, sort_by)

        if not books:
            st.info("No books found. Add some books to get started!")
            return

        # Display books
        for book in books:
            with st.expander(f"{book[1]} by {book[2]}"):
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Display cover image
                    if book[7]:  # Local cover image
                        try:
                            st.image(book[7], caption="Cover Image")
                        except Exception as e:
                            logger.warning(f"Error displaying cover image: {str(e)}")
                            st.info("Cover image not available")
                    elif book[8]:  # Check metadata for cover URL
                        try:
                            metadata = json.loads(book[8])
                            if 'cover_url' in metadata and metadata['cover_url']:
                                st.image(metadata['cover_url'], caption="Cover Image")
                        except Exception:
                            pass

                    # Basic information
                    st.write("**Basic Information**")
                    st.write(f"**Title:** {book[1]}")
                    st.write(f"**Author:** {book[2]}")
                    st.write(f"**Year:** {book[3] if book[3] else 'Unknown'}")
                    st.write(f"**ISBN:** {book[4] if book[4] else 'N/A'}")
                    st.write(f"**Publisher:** {book[5] if book[5] else 'Unknown'}")
                    st.write(f"**Condition:** {book[6]}")
                    st.write(f"**Added:** {book[9][:10] if book[9] else 'N/A'}")

                with col2:
                    if book[8]:
                        metadata = json.loads(book[8])

                        # Personal Notes
                        if len(book) > 11 and book[11]:
                            st.write("**Personal Notes:**")
                            st.write(f"_{book[11]}_")
                            st.write("")

                        # LLM Generated Information
                        st.write("**LLM Generated Information**")

                        if metadata.get('synopsis'):
                            st.write("**Synopsis:**")
                            st.write(metadata['synopsis'])

                        if metadata.get('themes'):
                            st.write("**Themes:**")
                            for theme in metadata['themes']:
                                st.write(f"- {theme}")

                        if metadata.get('genre'):
                            st.write("**Genre:**")
                            for genre in metadata['genre']:
                                st.write(f"- {genre}")

                        if metadata.get('historical_context'):
                            st.write("**Historical Context:**")
                            st.write(metadata['historical_context'])

                        # Related works
                        if metadata.get('related_works'):
                            st.write("**LLM Suggested Related Books:**")
                            for work in metadata['related_works']:
                                if isinstance(work, dict):
                                    st.write(f"- **{work.get('title', '')}** by {work.get('author', '')}")
                                    if work.get('reason'):
                                        st.write(f"  _{work['reason']}_")
                                else:
                                    st.write(f"- {work}")

                        # Related books from collection
                        related_books = find_related_books(db, book[0], metadata)
                        if related_books:
                            st.write("**Similar Books in Your Collection:**")
                            for related in related_books:
                                st.write(f"- **{related['title']}** by {related['author']}")

                        # Sources
                        if "sources" in metadata:
                            st.write("**Data Sources:**", ", ".join(metadata["sources"]))

                # Action buttons
                st.divider()
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("🗑 Delete", key=f"del_{book[0]}"):
                        try:
                            db.delete_book(book[0])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting book: {str(e)}")

                with col5:
                    if st.button("🔄 Refresh", key=f"refresh_{book[0]}"):
                        try:
                            with st.spinner("Updating book information..."):
                                enhanced_metadata = book_service.enhance_book_data(
                                    book[1], book[2], book[3], book[4]
                                )
                                if enhanced_metadata:
                                    db.update_metadata(book[0], enhanced_metadata)
                                    st.success("Metadata refreshed!")
                                    st.rerun()
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
    st.header("Library Analytics")

    # Theme Analysis section
    st.subheader("Theme Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Extract Current Themes"):
            try:
                with st.spinner("Extracting themes from library..."):
                    books = db.get_books_with_metadata()
                    theme_data = extract_and_save_themes(books)
                    st.success(f"Extracted {len(theme_data['themes'])} unique themes!")

                    with open("data/theme_inventory.json", "r") as f:
                        st.download_button(
                            "Download Theme Inventory",
                            f.read(),
                            "theme_inventory.json",
                            "application/json"
                        )
            except Exception as e:
                logger.error(f"Error extracting themes: {str(e)}", exc_info=True)
                st.error("Failed to extract themes")

    with col2:
        if st.button("Analyze Theme Groupings"):
            try:
                with open("data/theme_inventory.json", "r") as f:
                    theme_data = json.load(f)

                with st.spinner("Analyzing themes..."):
                    theme_analysis = book_service.analyze_themes(theme_data['themes'])
                    if theme_analysis:
                        # Save analysis
                        with open("data/theme_analysis.json", "w") as f:
                            json.dump(theme_analysis, f, indent=2)
                        st.success("Theme analysis complete!")
                    else:
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
            st.divider()
            display_theme_analysis(theme_analysis)
    except FileNotFoundError:
        st.info("No theme analysis available. Start by clicking 'Extract Current Themes'.")
    except Exception as e:
        logger.error(f"Error loading theme analysis: {str(e)}")

    st.divider()

    # Get analytics data
    try:
        books = db.get_all_books()
        stats, genre_counts, theme_counts = generate_analytics(books)

        # Basic metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Books", stats.total_books)
        col2.metric("Unique Authors", stats.unique_authors)
        col3.metric("Avg. Publication Year", stats.avg_pub_year or "N/A")

        if stats.common_time_period:
            col4.metric(
                "Common Time Period",
                stats.common_time_period,
                f"{round(stats.time_period_coverage * 100)}% coverage"
            )
        else:
            col4.metric("Common Time Period", "N/A")

        fiction_pct = round(stats.fiction_ratio * 100)
        nonfiction_pct = round(stats.nonfiction_ratio * 100)
        col5.metric("Fiction/Non-Fiction", f"{fiction_pct}% / {nonfiction_pct}%")

        # Genre distribution
        if not genre_counts.empty:
            st.subheader("Genre Distribution")

            tab1, tab2, tab3 = st.tabs(["Combined View", "Fiction", "Non-Fiction"])

            with tab1:
                fig = px.sunburst(
                    genre_counts,
                    path=['category', 'genre'],
                    values='count',
                    title='All Genres'
                )
                st.plotly_chart(fig)

            with tab2:
                fiction_data = genre_counts[genre_counts['category'] == 'Fiction']
                if not fiction_data.empty:
                    fig_fiction = px.pie(
                        fiction_data,
                        values='count',
                        names='genre',
                        title='Fiction Genres'
                    )
                    st.plotly_chart(fig_fiction)
                else:
                    st.info("No fiction books in your collection yet.")

            with tab3:
                nonfiction_data = genre_counts[genre_counts['category'] == 'Non-Fiction']
                if not nonfiction_data.empty:
                    fig_nonfiction = px.pie(
                        nonfiction_data,
                        values='count',
                        names='genre',
                        title='Non-Fiction Genres'
                    )
                    st.plotly_chart(fig_nonfiction)
                else:
                    st.info("No non-fiction books in your collection yet.")

        # Theme distribution
        if not theme_counts.empty:
            st.subheader("Theme Distribution")
            fig_themes = px.treemap(
                theme_counts,
                path=['theme'],
                values='count',
                title='Themes Across Your Collection'
            )
            st.plotly_chart(fig_themes)

    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}", exc_info=True)
        st.error("Error generating analytics")

def render_network_view_page(db: Database):
    """Render the Network View page."""
    st.header("Book Relationship Network")

    st.markdown("""
    This network visualization shows relationships between books in your collection.

    **Connection Types:**
    - 👤 Author (Strong connection)
    - 📅 Decade (Medium connection)
    - 🎭 Themes (Connection strength based on number of shared themes)

    **How to Read:**
    - Larger nodes indicate books with more connections
    - Different colored lines show different types of connections
    - Hover over nodes and lines for details
    """)

    view_type = st.radio(
        "Select View",
        ["All Books", "Fiction Only", "Non-Fiction Only"],
        horizontal=True
    )

    try:
        with st.spinner("Generating network visualization..."):
            category = None
            if view_type == "Fiction Only":
                category = "Fiction"
            elif view_type == "Non-Fiction Only":
                category = "Non-Fiction"

            books = db.get_all_books()
            G = create_book_network(books, category)
            fig = visualize_book_network(G)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Network statistics
                st.subheader("Network Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Books", len(G.nodes()))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    if len(G.nodes()) > 0:
                        import networkx as nx
                        avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
            else:
                st.info("Add more books to see their relationships!")

    except Exception as e:
        logger.error(f"Error creating network view: {str(e)}", exc_info=True)
        st.error("Error generating network visualization")

def render_executive_summary_page(db: Database, book_service: BookService):
    """Render the Executive Summary page."""
    st.header("Library Executive Summary")

    # Quick stats
    try:
        books = db.get_all_books()
        stats, genre_counts, _ = generate_analytics(books)

        st.subheader("Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Books", stats.total_books)
        with col2:
            st.metric("Unique Authors", stats.unique_authors)
        with col3:
            st.metric("Average Publication Year",
                     stats.avg_pub_year if stats.avg_pub_year else "N/A")
        with col4:
            fiction_pct = round(stats.fiction_ratio * 100)
            nonfiction_pct = round(stats.nonfiction_ratio * 100)
            st.metric("Fiction/Non-Fiction",
                     f"{fiction_pct}% / {nonfiction_pct}%")

        st.divider()
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")

    # Catalog and summary generation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Library Catalog"):
            try:
                with st.spinner("Generating catalog..."):
                    json_path, library_data = generate_library_json(db)
                    st.success("Library catalog generated!")

                    with open(json_path, "r") as f:
                        st.download_button(
                            "Download Library Catalog",
                            f.read(),
                            "library_catalog.json",
                            "application/json"
                        )
            except Exception as e:
                logger.error(f"Error generating catalog: {str(e)}", exc_info=True)
                st.error("Failed to generate catalog")

    with col2:
        if st.button("Generate Summary"):
            try:
                with open("data/library_catalog.json", "r") as f:
                    library_data = json.load(f)

                with st.spinner("Generating summary..."):
                    summary = book_service.generate_executive_summary(library_data)
                    if summary:
                        summary_info = {
                            "last_updated": datetime.now().isoformat(),
                            "summary": summary
                        }
                        with open("data/executive_summary.json", "w") as f:
                            json.dump(summary_info, f, indent=2)

                        st.success("Summary generated!")
                    else:
                        st.error("Failed to generate summary")
            except FileNotFoundError:
                st.error("Please generate library catalog first")
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}", exc_info=True)
                st.error("Failed to generate summary")

    # Display existing summary
    try:
        with open("data/executive_summary.json", "r") as f:
            summary_data = json.load(f)

        st.subheader("Collection Summary")
        st.write(summary_data["summary"]["summary"])

        st.subheader("Key Patterns")
        for pattern in summary_data["summary"]["patterns"]:
            st.write(f"• {pattern}")

        st.subheader("Recommendations")
        for rec in summary_data["summary"]["recommendations"]:
            st.write(f"• {rec}")

        st.caption(f"Last updated: {summary_data['last_updated']}")

    except FileNotFoundError:
        st.info("No summary available. Generate one using the buttons above.")
    except Exception as e:
        logger.error(f"Error displaying summary: {str(e)}")

def render_ask_library_page(db: Database, book_service: BookService):
    """Render the Ask the Library page."""
    st.header("Ask the Large Library Model")

    st.markdown("""
    Ask questions about your library collection. For example:
    - What themes are common in my collection?
    - Which authors do I read most?
    - What genres are underrepresented?
    - Suggest books from my collection for a specific mood or topic
    """)

    query = st.text_area("Enter your question:", height=100)

    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question")
            return

        try:
            with open("data/library_catalog.json", "r") as f:
                library_data = json.load(f)

            with st.spinner("Analyzing your library..."):
                response = book_service.ask_library_question(query, library_data)
                if response:
                    st.markdown("### Response")
                    st.markdown(response)
                else:
                    st.error("Failed to get a response")

        except FileNotFoundError:
            st.error("Library catalog not found. Please generate it first in the Executive Summary page.")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error("An error occurred while processing your question")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the logs and restart the application.")
