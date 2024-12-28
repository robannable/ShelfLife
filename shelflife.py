import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime
import config
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import io
import networkx as nx
from functools import lru_cache
import hashlib
from typing import Optional
from api_utils import fetch_book_metadata, test_api_connection
import os
from pathlib import Path
from constants import STANDARD_GENRES, GENRE_PROMPT

# Cache for API responses
@lru_cache(maxsize=1000)
def cached_api_call(cache_key):
    return st.session_state.get(cache_key, None)

# Image processing
def process_image(uploaded_file):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Get file extension from the uploaded file name
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Map common image extensions to PIL format names
            format_map = {
                'jpg': 'JPEG',
                'jpeg': 'JPEG',
                'png': 'PNG',
                'gif': 'GIF',
                'bmp': 'BMP',
                'webp': 'WEBP'
            }
            
            # Use mapped format or default to JPEG
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
            if config.DEBUG_MODE:
                st.error(f"Error processing image: {str(e)}\nFile type: {uploaded_file.type}")
            else:
                st.error("Error processing image. Please ensure it's a valid image file (JPG, PNG, GIF, BMP, or WebP)")
            return None
    return None

# Enhanced database initialization
def init_db():
    # Create data directory if it doesn't exist
    data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database path
    db_path = data_dir / 'database.db'
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Add indices for better performance
    c.executescript('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            year INTEGER,
            isbn TEXT,
            publisher TEXT,
            condition TEXT,
            cover_image BLOB,
            metadata JSON,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_title ON books(title);
        CREATE INDEX IF NOT EXISTS idx_author ON books(author);
        CREATE INDEX IF NOT EXISTS idx_year ON books(year);
    ''')
    conn.commit()
    return conn

# Enhanced book data function
def enhance_book_data(title: str, author: str, year: Optional[int] = None, isbn: Optional[str] = None):
    """Enhanced book data function with debugging and multiple sources."""
    cache_key = hashlib.md5(f"{title}{author}{str(year)}{str(isbn)}".encode()).hexdigest()
    
    # Check cache
    cached_data = cached_api_call(cache_key)
    if cached_data:
        return cached_data
    
    try:
        # First, get metadata from Open Library and Google Books
        metadata = fetch_book_metadata(title, author, isbn)
        
        if config.DEBUG_MODE:
            st.write("Debug: Metadata from APIs:", metadata)
        
        # Then enhance with Perplexity
        headers = {
            "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Include metadata and genre constraints in the prompt
        enhanced_prompt = config.BOOK_ANALYSIS_PROMPT.format(
            title=title,
            author=author,
            year=year or metadata.get("year", "unknown")
        )
        
        # Add genre selection prompt
        genre_selection_prompt = GENRE_PROMPT.format(genres="\n".join(f"- {genre}" for genre in STANDARD_GENRES))
        
        payload = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {
                    "role": "user",
                    "content": f"{enhanced_prompt}\n\nFor genre classification:\n{genre_selection_prompt}"
                }
            ]
        }
        
        if config.DEBUG_MODE:
            st.write("Debug: Perplexity API request:", payload)
        
        response = requests.post(
            config.PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if config.DEBUG_MODE:
            st.write("Debug: Perplexity API response status:", response.status_code)
            st.write("Debug: Perplexity API response:", response.text)
        
        if response.status_code == 200:
            response_data = response.json()
            try:
                content = response_data['choices'][0]['message']['content']
                # Clean the content string and extract JSON
                content = content.split('json\n', 1)[1] if 'json\n' in content else content
                # Remove any trailing content after the JSON
                content = content.split('\n}\n')[0] + '\n}'
                book_data = json.loads(content)
                
                # Merge all data
                final_data = {
                    **metadata,
                    **book_data,
                    "sources": metadata.get("sources", []) + ["Perplexity"]
                }
                
                # Cache the result
                st.session_state[cache_key] = final_data
                return final_data
                
            except (json.JSONDecodeError, KeyError) as e:
                if config.DEBUG_MODE:
                    st.error(f"Error parsing Perplexity response: {str(e)}")
                return metadata  # Return metadata without enhancement
        else:
            if config.DEBUG_MODE:
                st.error(f"Perplexity API Error: {response.status_code}")
            return metadata
            
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error in enhance_book_data: {str(e)}")
        return None

# Analytics functions
def generate_analytics(conn):
    c = conn.cursor()
    
    # Basic statistics
    stats = {}
    stats['total_books'] = c.execute('SELECT COUNT(*) FROM books').fetchone()[0]
    stats['unique_authors'] = c.execute('SELECT COUNT(DISTINCT author) FROM books').fetchone()[0]
    
    # Calculate average year excluding NULL values
    avg_year_result = c.execute('''
        SELECT AVG(CAST(year AS FLOAT))
        FROM books 
        WHERE year IS NOT NULL
    ''').fetchone()[0]
    
    stats['avg_year'] = round(avg_year_result) if avg_year_result is not None else None
    
    # Genre distribution
    books_df = pd.read_sql_query('''
        SELECT metadata FROM books WHERE metadata IS NOT NULL
    ''', conn)
    
    genres = []
    themes = []
    for _, row in books_df.iterrows():
        metadata = json.loads(row['metadata'])
        if 'genre' in metadata:
            genres.extend(metadata['genre'])
        if 'themes' in metadata:
            themes.extend(metadata['themes'])
    
    # Create proper DataFrames for visualization
    genre_df = pd.DataFrame(genres, columns=['genre'])
    genre_counts = genre_df['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    
    theme_df = pd.DataFrame(themes, columns=['theme'])
    theme_counts = theme_df['theme'].value_counts().reset_index()
    theme_counts.columns = ['theme', 'count']
    
    return stats, genre_counts, theme_counts

def generate_executive_summary(conn):
    """Generate an executive summary with strictly formatted output."""
    c = conn.cursor()
    
    # Fetch all books with their metadata
    books = c.execute('SELECT title, author, metadata FROM books').fetchall()
    
    # Create a detailed prompt for Perplexity
    book_list = "\n".join([f"- {book[0]} by {book[1]}" for book in books])
    genres = set()
    
    # Extract genres from metadata
    for book in books:
        metadata = json.loads(book[2])
        genres.update(metadata.get('genre', []))
    
    prompt = f"""Analyze this library collection and provide a JSON response with the following structure:
{{
    "summary": "A clear paragraph describing the collection's focus and character (200 words)",
    "taxonomy": {{
        "categories": [
            {{
                "name": "Category Name",
                "items": [
                    {{
                        "title": "Item Title",
                        "description": "Brief description"
                    }}
                ]
            }}
        ]
    }},
    "patterns": [
        "Pattern 1",
        "Pattern 2",
        "Pattern 3"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ]
}}

Books in collection:
{book_list}

Primary genres: {', '.join(genres)}

Ensure the response is valid JSON and maintains this exact structure."""

    # Make Perplexity API call
    headers = {
        "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            config.PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            # Find JSON content between curly braces
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    if config.DEBUG_MODE:
                        st.error("Could not find valid JSON in response")
                    return None
            except json.JSONDecodeError as e:
                if config.DEBUG_MODE:
                    st.error(f"JSON parsing error: {str(e)}\nContent: {content}")
                return None
        else:
            if config.DEBUG_MODE:
                st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error generating executive summary: {str(e)}")
        return None

def find_related_books(conn, current_book_id, metadata):
    """Find up to 3 related books based on shared genres."""
    c = conn.cursor()
    
    # Get all other books
    other_books = c.execute('''
        SELECT id, title, author, metadata 
        FROM books 
        WHERE id != ?
    ''', (current_book_id,)).fetchall()
    
    # Extract current book's genres
    current_genres = set(metadata.get('genre', []))
    
    # Calculate genre overlap for each book
    related_books = []
    for book in other_books:
        other_metadata = json.loads(book[3])
        other_genres = set(other_metadata.get('genre', []))
        shared_genres = current_genres & other_genres
        
        if shared_genres:  # Only include if there are shared genres
            related_books.append({
                'id': book[0],
                'title': book[1],
                'author': book[2],
                'shared_genres': shared_genres,
                'genre_count': len(shared_genres)
            })
    
    # Sort by number of shared genres and return top 3
    related_books.sort(key=lambda x: x['genre_count'], reverse=True)
    return related_books[:3]

def refresh_all_metadata(conn):
    """Refresh metadata for all books in the collection."""
    c = conn.cursor()
    books = c.execute('SELECT id, title, author, year, isbn FROM books').fetchall()
    total = len(books)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, book in enumerate(books):
        try:
            # Update progress
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1} of {total}: {book[1]}")
            
            # Get fresh metadata
            enhanced_data = enhance_book_data(book[1], book[2], book[3], book[4])
            
            if enhanced_data:
                c.execute('''
                    UPDATE books 
                    SET metadata = ?,
                        updated_at = ?
                    WHERE id = ?
                ''', (
                    json.dumps(enhanced_data),
                    datetime.now().isoformat(),
                    book[0]
                ))
                conn.commit()
                
        except Exception as e:
            if config.DEBUG_MODE:
                st.error(f"Error updating {book[1]}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return True

def manage_executive_summary(conn, summary_data=None, refresh=False):
    """Manage executive summary storage and retrieval."""
    summary_path = "data/executive_summary.json"
    os.makedirs("data", exist_ok=True)
    
    if refresh or summary_data:
        # Save new summary data
        summary_info = {
            "last_updated": datetime.now().isoformat(),
            "summary": summary_data
        }
        with open(summary_path, "w") as f:
            json.dump(summary_info, f, indent=4)
        return summary_data
    
    # Try to load existing summary
    try:
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
                return data["summary"]
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error loading executive summary: {str(e)}")
    return None

def format_taxonomy_category(category):
    """Format a single taxonomy category as HTML."""
    html = f"""
    <div class="taxonomy-category" style="
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 5px;">
        <h4 style="margin: 0 0 15px 0;">{category['name']}</h4>
        <ul style="
            list-style-type: circle;
            margin: 0;
            padding-left: 20px;">
    """
    
    for item in category['items']:
        html += f"""
            <li style="margin-bottom: 10px;">
                <strong>{item['title']}</strong>
                {f": {item['description']}" if item.get('description') else ""}
            </li>
        """
    
    html += """
        </ul>
    </div>
    """
    return html

# Main application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="ShelfLife",
        page_icon="📚",
        layout="centered"
    )
    
    # Load and inject CSS
    def load_css(file_name):
        with open(os.path.join('static', file_name)) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    load_css('styles.css')
    
    # Initialize connection
    conn = init_db()
    
    with st.sidebar:
        st.title("ShelfLife 📚")
        
        # API Status Section with manual check button
        st.subheader("API Status")
        if st.button("Check API Status"):
            with st.spinner("Checking API connections..."):
                # Check Perplexity API
                perplexity_status = test_api_connection("perplexity")
                st.write("Perplexity API:", 
                        "✅" if perplexity_status["success"] else "❌")
                
                # Check Google Books API
                google_status = test_api_connection("google_books")
                st.write("Google Books API:", 
                        "✅" if google_status["success"] else "❌")
                
                # Check Open Library API
                ol_status = test_api_connection("open_library")
                st.write("Open Library API:", 
                        "✅" if ol_status["success"] else "❌")
                
                if config.DEBUG_MODE and not all(s["success"] for s in [perplexity_status, google_status, ol_status]):
                    failed_apis = []
                    if not perplexity_status["success"]:
                        failed_apis.append(f"Perplexity: {perplexity_status['error']}")
                    if not google_status["success"]:
                        failed_apis.append(f"Google Books: {google_status['error']}")
                    if not ol_status["success"]:
                        failed_apis.append(f"Open Library: {ol_status['error']}")
                    st.error("API Errors:\n" + "\n".join(failed_apis))
        
        # Navigation
        st.subheader("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Add Book", "View Collection", "Analytics", "Network View", "Executive Summary"]
        )
    
    if page == "Add Book":
        st.header("Add New Book")
        
        with st.form("book_form"):
            title = st.text_input("Title*")
            author = st.text_input("Author*")
            year = st.number_input("Year", 
                min_value=0, 
                max_value=datetime.now().year,
                value=None)  # Changed to None default
            # Convert 0 to None for database storage
            year = year if year != 0 else None
            isbn = st.text_input("ISBN (optional)")
            publisher = st.text_input("Publisher (optional)")
            condition = st.selectbox("Condition", 
                ["New", "Like New", "Very Good", "Good", "Fair", "Poor"])
            cover_image = st.file_uploader("Cover Image", type=['png', 'jpg', 'jpeg'])
            
            submitted = st.form_submit_button("Add Book")
            
            if submitted and title and author:
                with st.spinner("Fetching book information..."):
                    enhanced_data = enhance_book_data(title, author, year)
                    
                if enhanced_data:
                    image_data = process_image(cover_image) if cover_image else None
                    
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO books (
                            title, author, year, isbn, publisher, 
                            condition, cover_image, metadata, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        title, author, year, isbn, publisher,
                        condition, image_data, json.dumps(enhanced_data),
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    st.success("Book added successfully!")
    
    elif page == "View Collection":
        st.header("Your Library")
        
        # Search and filter options
        search = st.text_input("Search books", "")
        sort_by = st.selectbox("Sort by", ["Title", "Author", "Year", "Recent"])
        
        # Build query
        query = 'SELECT * FROM books'
        params = []
        if search:
            query += ' WHERE title LIKE ? OR author LIKE ?'
            params.extend([f'%{search}%', f'%{search}%'])
        
        if sort_by == "Recent":
            query += ' ORDER BY created_at DESC'
        else:
            query += f' ORDER BY {sort_by.lower()}'
            
        # Display books
        c = conn.cursor()
        books = c.execute(query, params).fetchall()
        
        for book in books:
            with st.expander(f"{book[1]} by {book[2]}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display cover image
                    if book[7]:  # Local cover image
                        st.image(book[7], caption="Cover Image")
                    elif 'cover_url' in json.loads(book[8]):  # API-fetched cover
                        st.image(json.loads(book[8])['cover_url'], caption="Cover Image")
                    
                    # Basic book information
                    st.write("**Basic Information**")
                    st.write(f"📖 **Title:** {book[1]}")
                    st.write(f"✍️ **Author:** {book[2]}")
                    st.write(f"📅 **Year:** {book[3] if book[3] else 'Unknown'}")
                    st.write(f"📚 **ISBN:** {book[4] if book[4] else 'N/A'}")
                    st.write(f"🏢 **Publisher:** {book[5] if book[5] else 'Unknown'}")
                    st.write(f"📊 **Condition:** {book[6]}")
                    st.write(f"📝 **Added:** {book[9][:10]}")  # Show only date part
                
                with col2:
                    metadata = json.loads(book[8])
                    
                    # Enhanced information from APIs
                    st.write("**Enhanced Information**")
                    
                    # Sources used
                    if "sources" in metadata:
                        st.write("🔍 **Data Sources:**", ", ".join(metadata["sources"]))
                    
                    # Synopsis
                    st.write("📝 **Synopsis:**")
                    st.write(metadata.get('synopsis', 'No synopsis available'))
                    
                    # Themes and Genre
                    if metadata.get('themes'):
                        st.write("🎯 **Themes:**")
                        for theme in metadata['themes']:
                            st.write(f"- {theme}")
                    
                    if metadata.get('genre'):
                        st.write("📚 **Genre:**")
                        for genre in metadata['genre']:
                            st.write(f"- {genre}")
                    
                    # Historical Context
                    if metadata.get('historical_context'):
                        st.write("🏺 **Historical Context:**")
                        st.write(metadata['historical_context'])
                    
                    # Reading Level and Time Period
                    col3, col4 = st.columns(2)
                    with col3:
                        if metadata.get('reading_level'):
                            st.write("📖 **Reading Level:**")
                            st.write(metadata['reading_level'])
                    with col4:
                        if metadata.get('time_period'):
                            st.write("⌛ **Time Period:**")
                            st.write(metadata['time_period'])
                    
                    # Add Related Books section after genre display
                    related_books = find_related_books(conn, book[0], metadata)
                    if related_books:
                        st.markdown("---")  # Add visual separator
                        st.write("📚 **Similar Books in Your Collection:**")
                        for related in related_books:
                            st.markdown(f"""
                            - **{related['title']}** by {related['author']}  
                              _Shared genres: {', '.join(related['shared_genres'])}_
                            """)
                        st.markdown("---")  # Add visual separator
                    
                    # Related Works from external sources
                    if metadata.get('related_works'):
                        st.write("🔗 **Related Works:**")
                        for work in metadata['related_works']:
                            st.markdown(f"""
                            **{work['title']}** by {work['author']}  
                            _{work.get('reason', 'No connection details available')}_
                            """)
                    
                    # Keywords (if present)
                    if metadata.get('keywords'):
                        st.write("🏷️ **Keywords:**")
                        st.write(", ".join(metadata['keywords']))
                
                # Add action buttons at bottom of col1
                st.divider()
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("🗑�� Delete", key=f"del_{book[0]}"):
                        c.execute('DELETE FROM books WHERE id = ?', (book[0],))
                        conn.commit()
                        st.rerun()
                with col4:
                    if st.button("✏️ Edit", key=f"edit_{book[0]}"):
                        st.session_state[f"edit_book_{book[0]}"] = True
                with col5:
                    if st.button("🔄 Refresh", key=f"refresh_{book[0]}"):
                        with st.spinner("Updating book information..."):
                            enhanced_data = enhance_book_data(book[1], book[2], book[3], book[4])
                            if enhanced_data:
                                c.execute('''
                                    UPDATE books 
                                    SET metadata = ?, updated_at = ?
                                    WHERE id = ?
                                ''', (
                                    json.dumps(enhanced_data),
                                    datetime.now().isoformat(),
                                    book[0]
                                ))
                                conn.commit()
                                st.rerun()
                
                # Show edit form if edit button was clicked
                if st.session_state.get(f"edit_book_{book[0]}", False):
                    with st.form(key=f"edit_form_{book[0]}"):
                        st.subheader("Edit Book Information")
                        new_title = st.text_input("Title", value=book[1])
                        new_author = st.text_input("Author", value=book[2])
                        new_year = st.number_input("Year", 
                            min_value=0, 
                            max_value=datetime.now().year,
                            value=book[3] if book[3] else 0)
                        new_isbn = st.text_input("ISBN", value=book[4] if book[4] else "")
                        new_publisher = st.text_input("Publisher", value=book[5] if book[5] else "")
                        new_condition = st.selectbox("Condition", 
                            ["New", "Like New", "Very Good", "Good", "Fair", "Poor"],
                            index=["New", "Like New", "Very Good", "Good", "Fair", "Poor"].index(book[6]) if book[6] else 0)
                        new_cover = st.file_uploader("New Cover Image (optional)", type=['png', 'jpg', 'jpeg'])
                        
                        col8, col9 = st.columns(2)
                        with col8:
                            if st.form_submit_button("Save Changes"):
                                # Process new cover image if provided
                                new_image_data = process_image(new_cover) if new_cover else book[7]
                                
                                # Update database
                                c.execute('''
                                    UPDATE books 
                                    SET title = ?, author = ?, year = ?, isbn = ?, 
                                        publisher = ?, condition = ?, cover_image = ?,
                                        updated_at = ?
                                    WHERE id = ?
                                ''', (
                                    new_title, new_author, new_year, new_isbn,
                                    new_publisher, new_condition, new_image_data,
                                    datetime.now().isoformat(), book[0]
                                ))
                                conn.commit()
                                
                                # Clear edit state and refresh
                                del st.session_state[f"edit_book_{book[0]}"]
                                st.rerun()
                        
                        with col9:
                            if st.form_submit_button("Cancel"):
                                del st.session_state[f"edit_book_{book[0]}"]
                                st.rerun()
    
    elif page == "Analytics":
        st.header("Library Analytics")
        
        # Existing analytics code
        stats, genre_counts, theme_counts = generate_analytics(conn)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Books", stats['total_books'])
        col2.metric("Unique Authors", stats['unique_authors'])
        col3.metric("Average Publication Year", 
                    stats['avg_year'] if stats['avg_year'] is not None else "N/A")
        
        # Genre distribution
        if not genre_counts.empty:
            fig_genre = px.pie(genre_counts, values='count', names='genre', title='Genre Distribution')
            st.plotly_chart(fig_genre)
        
        # Theme distribution
        if not theme_counts.empty:
            fig_themes = px.treemap(theme_counts, path=['theme'], values='count', title='Theme Distribution')
            st.plotly_chart(fig_themes)
    
    elif page == "Network View":
        st.header("Book Network")
        
        # Create network graph
        G = nx.Graph()
        c = conn.cursor()
        books = c.execute('SELECT id, title, metadata FROM books').fetchall()
        
        # Add nodes and edges based on shared themes/genres
        for book in books:
            G.add_node(book[1])  # Add book title as node
            metadata = json.loads(book[2])
            themes = metadata.get('themes', [])
            
            # Connect books with shared themes
            for other_book in books:
                if book[0] != other_book[0]:  # Don't connect to self
                    other_metadata = json.loads(other_book[2])
                    other_themes = other_metadata.get('themes', [])
                    shared_themes = set(themes) & set(other_themes)
                    
                    if shared_themes:
                        G.add_edge(book[1], other_book[1], 
                                 weight=len(shared_themes))
        
        # Create network visualization using plotly
        pos = nx.spring_layout(G)
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
        
        node_trace = go.Scatter(
            x=[], y=[], mode='markers+text',
            text=[],
            textposition="bottom center",
            hoverinfo='text',
            marker=dict(size=20)
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0,l=0,r=0,t=0),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        st.plotly_chart(fig)
    
    elif page == "Executive Summary":
        st.header("📊 Collection Executive Summary")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            refresh_summary = st.button("🔄 Refresh Summary")
        
        summary_data = manage_executive_summary(conn)
        
        if refresh_summary or summary_data is None:
            with st.spinner("Analyzing your collection..."):
                new_summary = generate_executive_summary(conn)
                if new_summary:
                    summary_data = manage_executive_summary(conn, new_summary, refresh=True)
                    st.success("Summary refreshed successfully!")
        
        if summary_data:
            # Display last updated time with better formatting
            if os.path.exists("data/executive_summary.json"):
                with open("data/executive_summary.json", "r") as f:
                    metadata = json.load(f)
                    last_updated = datetime.fromisoformat(metadata["last_updated"])
                    st.caption(f"_Last updated: {last_updated.strftime('%B %d, %Y at %I:%M %p')}_")
            
            # Quick Stats Section
            st.markdown("### 📈 Collection Statistics")
            st.markdown("---")
            stats, genre_counts, theme_counts = generate_analytics(conn)
            
            # Display stats in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Books", stats['total_books'])
            with col2:
                st.metric("Unique Authors", stats['unique_authors'])
            with col3:
                st.metric("Average Year", 
                         stats['avg_year'] if stats['avg_year'] is not None else "N/A")
            with col4:
                top_genre = genre_counts.iloc[0] if not genre_counts.empty else None
                st.metric("Top Genre", 
                         f"{top_genre.name}" if top_genre is not None else "N/A",
                         f"{top_genre['count']} books" if top_genre is not None else "")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Executive Summary
            st.markdown("### 📝 Executive Summary")
            st.markdown("---")
            st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;">
                {summary_data['summary']}
            </div>
            """, unsafe_allow_html=True)
            
            # Taxonomy
            st.markdown("### 🌳 Intellectual Taxonomy")
            st.markdown("---")
            taxonomy_html = "".join(
                format_taxonomy_category(category)
                for category in summary_data['taxonomy']['categories']
            )
            st.markdown(taxonomy_html, unsafe_allow_html=True)
            
            # Patterns
            st.markdown("### 🔍 Collection Patterns")
            st.markdown("---")
            patterns_html = """
            <div style="
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;">
                <ul style="
                    list-style-type: circle;
                    margin: 0;
                    padding-left: 20px;">
            """
            for pattern in summary_data['patterns']:
                patterns_html += f"<li style='margin-bottom: 10px;'>{pattern}</li>"
            patterns_html += "</ul></div>"
            st.markdown(patterns_html, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 📚 Recommended Acquisitions")
            st.markdown("---")
            recommendations_html = """
            <div style="
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;">
                <ul style="
                    list-style-type: circle;
                    margin: 0;
                    padding-left: 20px;">
            """
            for rec in summary_data['recommendations']:
                recommendations_html += f"<li style='margin-bottom: 10px;'>{rec}</li>"
            recommendations_html += "</ul></div>"
            st.markdown(recommendations_html, unsafe_allow_html=True)
            
        else:
            st.info("No summary available yet. Click 'Refresh Summary' to generate one.")

if __name__ == "__main__":
    main() 