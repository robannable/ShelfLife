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
from constants import STANDARD_GENRES, GENRE_PROMPT, GENRE_CATEGORIES
from math import sqrt

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
    
    # Create version table to track schema changes
    c.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Get current schema version
    current_version = c.execute('SELECT MAX(version) FROM schema_version').fetchone()[0] or 0
    
    # Define schema updates
    schema_updates = {
        1: '''
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
        ''',
        2: '''
            ALTER TABLE books ADD COLUMN personal_notes TEXT;
        '''
        # Add future schema versions here as needed
        # 3: 'ALTER TABLE books ADD COLUMN new_column_name TEXT;'
    }
    
    # Apply any missing updates in order
    for version, update_sql in schema_updates.items():
        if version > current_version:
            try:
                c.executescript(update_sql)
                c.execute('INSERT INTO schema_version (version) VALUES (?)', (version,))
                conn.commit()
                if config.DEBUG_MODE:
                    st.write(f"Applied database schema update version {version}")
            except sqlite3.OperationalError as e:
                if config.DEBUG_MODE:
                    st.error(f"Error applying schema update {version}: {str(e)}")
                # Continue with other updates even if one fails
                continue
    
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
    
    # Genre distribution with fiction/non-fiction categorization
    books_df = pd.read_sql_query('''
        SELECT metadata FROM books WHERE metadata IS NOT NULL
    ''', conn)
    
    # Initialize categorized genres
    fiction_genres = []
    nonfiction_genres = []
    themes = []
    
    # Fiction and Non-Fiction genre lists
    FICTION_GENRES = {
        'Fantasy', 'Science Fiction', 'Mystery', 'Romance', 'Horror', 
        'Literary Fiction', 'Historical Fiction', 'Adventure', 'Thriller',
        'Young Adult', 'Children\'s Literature', 'Contemporary Fiction'
    }
    
    NONFICTION_GENRES = {
        'Biography', 'History', 'Science', 'Philosophy', 'Self-Help',
        'Business', 'Technology', 'Politics', 'Psychology', 'Art',
        'Travel', 'Cooking', 'Religion', 'Education', 'Reference'
    }
    
    for _, row in books_df.iterrows():
        metadata = json.loads(row['metadata'])
        if 'genre' in metadata:
            for genre in metadata['genre']:
                if genre in FICTION_GENRES:
                    fiction_genres.append(genre)
                elif genre in NONFICTION_GENRES:
                    nonfiction_genres.append(genre)
        if 'themes' in metadata:
            themes.extend(metadata['themes'])
    
    # Create DataFrames for visualization
    fiction_df = pd.DataFrame(fiction_genres, columns=['genre'])
    fiction_counts = fiction_df['genre'].value_counts().reset_index()
    fiction_counts.columns = ['genre', 'count']
    fiction_counts['category'] = 'Fiction'
    
    nonfiction_df = pd.DataFrame(nonfiction_genres, columns=['genre'])
    nonfiction_counts = nonfiction_df['genre'].value_counts().reset_index()
    nonfiction_counts.columns = ['genre', 'count']
    nonfiction_counts['category'] = 'Non-Fiction'
    
    # Combine fiction and non-fiction counts
    genre_counts = pd.concat([fiction_counts, nonfiction_counts])
    
    theme_df = pd.DataFrame(themes, columns=['theme'])
    theme_counts = theme_df['theme'].value_counts().reset_index()
    theme_counts.columns = ['theme', 'count']
    
    # Calculate fiction vs non-fiction ratio
    total_categorized = len(fiction_genres) + len(nonfiction_genres)
    if total_categorized > 0:
        stats['fiction_ratio'] = len(fiction_genres) / total_categorized
        stats['nonfiction_ratio'] = len(nonfiction_genres) / total_categorized
    else:
        stats['fiction_ratio'] = 0
        stats['nonfiction_ratio'] = 0
    
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

def create_book_network(conn):
    """Create a network visualization of book relationships."""
    G = nx.Graph()
    c = conn.cursor()
    books = c.execute('SELECT id, title, author, metadata FROM books').fetchall()
    
    # Create nodes and track metadata for each book
    book_metadata = {}
    for book in books:
        book_id, title, author, metadata_json = book
        metadata = json.loads(metadata_json)
        display_name = f"{title}\n{author}"
        G.add_node(display_name)
        
        # Extract year and convert to decade
        year = metadata.get('year')
        decade = f"{str(year)[:-1]}0s" if year else None
        
        # Simplify metadata storage
        book_metadata[display_name] = {
            'genres': set(metadata.get('genre', [])),
            'is_fiction': any(g in GENRE_CATEGORIES['Fiction'] for g in metadata.get('genre', [])),
            'decade': decade,
            'author': author
        }
    
    # Calculate relationships
    for book1 in book_metadata:
        for book2 in book_metadata:
            if book1 >= book2:  # Skip duplicate pairs
                continue
                
            # Calculate relationship strength
            shared_genres = len(book_metadata[book1]['genres'] & 
                             book_metadata[book2]['genres'])
            same_fiction_type = (book_metadata[book1]['is_fiction'] == 
                               book_metadata[book2]['is_fiction'])
            same_decade = (book_metadata[book1]['decade'] == 
                         book_metadata[book2]['decade'] and 
                         book_metadata[book1]['decade'] is not None)
            
            # Only create edge if there's a meaningful relationship
            score = (shared_genres * 3 +  # Weight shared genres heavily
                    same_fiction_type * 2 +
                    same_decade * 2)
            
            if score >= 3:  # Minimum threshold for connection
                G.add_edge(book1, book2, 
                          weight=score,
                          relationship={
                              'shared_genres': shared_genres,
                              'same_type': same_fiction_type,
                              'same_decade': same_decade
                          })
    
    return G

def visualize_book_network(G):
    """Create an interactive network visualization."""
    if len(G.nodes()) == 0:
        return None
    
    # Define connection colors
    COLORS = {
        'genre': 'rgba(65, 105, 225, 0.8)',  # Royal Blue
        'decade': 'rgba(50, 205, 50, 0.8)',  # Lime Green
        'category': 'rgba(255, 140, 0, 0.8)'  # Dark Orange
    }
    
    # Calculate node sizes based on connections
    node_degrees = dict(G.degree())
    node_sizes = {node: (degree + 1) * 10 for node, degree in node_degrees.items()}
    
    # Create layout
    pos = nx.spring_layout(G, k=1/sqrt(len(G.nodes())), iterations=50)
    
    # Create separate edge traces for each connection type
    edge_traces = []
    
    for connection_type, color in COLORS.items():
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            relationship = G.edges[edge]['relationship']
            should_include = False
            
            if connection_type == 'genre' and relationship['shared_genres'] > 0:
                should_include = True
            elif connection_type == 'decade' and relationship['same_decade']:
                should_include = True
            elif connection_type == 'category' and relationship['same_type']:
                should_include = True
                
            if should_include:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                hover_text = f"""
                Connection Type: {connection_type.title()}
                Shared Genres: {relationship['shared_genres']}
                Same Type: {'Yes' if relationship['same_type'] else 'No'}
                Same Decade: {'Yes' if relationship['same_decade'] else 'No'}
                """
                edge_text.extend([hover_text, hover_text, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=color),
            hoverinfo='text',
            text=edge_text,
            mode='lines',
            name=connection_type.title(),
            showlegend=True
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(node_sizes[node])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color='rgba(200, 200, 200, 0.8)',
            line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
        ),
        name='Books'
    )
    
    # Create figure with legend
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title='Book Relationship Network',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            annotations=[
                dict(
                    text="Larger nodes = more connections",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=-0.1
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

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
                value=None)
            year = year if year != 0 else None
            isbn = st.text_input("ISBN (optional)")
            publisher = st.text_input("Publisher (optional)")
            condition = st.selectbox("Condition", 
                ["New", "Like New", "Very Good", "Good", "Fair", "Poor"])
            cover_image = st.file_uploader("Cover Image", type=['png', 'jpg', 'jpeg'])
            
            # Add personal notes section
            personal_notes = st.text_area(
                "Personal Notes",
                placeholder="Add your personal thoughts, reading status, annotations, or any other notes about this book...",
                help="These notes are private and won't be used in analytics or summaries"
            )
            
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
                            condition, cover_image, metadata, personal_notes, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        title, author, year, isbn, publisher,
                        condition, image_data, json.dumps(enhanced_data),
                        personal_notes, datetime.now().isoformat()
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
                    
                    # Add Personal Notes section if they exist
                    if book[11]:  # personal_notes column
                        st.markdown("---")
                        st.markdown("📝 **Personal Notes:**")
                        st.markdown(f"_{book[11]}_")
                        st.markdown("---")
                    
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
                        
                        # Add personal notes to edit form
                        new_personal_notes = st.text_area(
                            "Personal Notes",
                            value=book[11] if book[11] else "",
                            placeholder="Add your personal thoughts, reading status, annotations, or any other notes about this book...",
                            help="These notes are private and won't be used in analytics or summaries"
                        )
                        
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
        
        # Get analytics data
        stats, genre_counts, theme_counts = generate_analytics(conn)
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Books", stats['total_books'])
        col2.metric("Unique Authors", stats['unique_authors'])
        col3.metric("Average Year", 
                   stats['avg_year'] if stats['avg_year'] is not None else "N/A")
        
        # Fiction vs Non-Fiction ratio
        fiction_percentage = round(stats['fiction_ratio'] * 100)
        nonfiction_percentage = round(stats['nonfiction_ratio'] * 100)
        col4.metric("Fiction/Non-Fiction", 
                   f"{fiction_percentage}% / {nonfiction_percentage}%")
        
        # Genre distribution with separate Fiction and Non-Fiction sections
        if not genre_counts.empty:
            st.subheader("Genre Distribution")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Combined View", "Fiction", "Non-Fiction"])
            
            with tab1:
                # Sunburst chart for all genres
                fig = px.sunburst(
                    genre_counts,
                    path=['category', 'genre'],
                    values='count',
                    title='All Genres'
                )
                st.plotly_chart(fig)
            
            with tab2:
                # Fiction genres
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
                # Non-fiction genres
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
    
    elif page == "Network View":
        st.header("Book Relationship Network")
        
        # Add description
        st.markdown("""
        This network visualization shows relationships between books in your collection.
        
        **Relationship Factors:**
        - 🎭 Shared Genres (weighted heavily)
        - 📖 Fiction/Non-Fiction Category
        - 📅 Same Decade
        
        **How to Read:**
        - Larger nodes indicate books with more connections
        - Darker colors indicate stronger relationships
        - Hover over nodes to see book details
        - Hover over lines to see relationship details
        
        *Books are connected when they share multiple relationship factors*
        """)
        
        # Create and display network
        with st.spinner("Generating network visualization..."):
            G = create_book_network(conn)
            fig = visualize_book_network(G)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics
                st.subheader("Network Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Connections", len(G.edges()))
                with col2:
                    if len(G.nodes()) > 0:
                        density = nx.density(G)
                        st.metric("Network Density", f"{density:.2%}")
                with col3:
                    if len(G.nodes()) > 0:
                        avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
            else:
                st.info("Add more books to see their relationships!")
    
    elif page == "Executive Summary":
        st.header("📊 Collection Executive Summary")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            refresh_summary = st.button("�� Refresh Summary")
        
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