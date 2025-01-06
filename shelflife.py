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
import re
import csv

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
    
    # Calculate average publication year excluding NULL values
    avg_year_result = c.execute('''
        SELECT AVG(CAST(year AS FLOAT))
        FROM books 
        WHERE year IS NOT NULL
    ''').fetchone()[0]
    
    stats['avg_pub_year'] = round(avg_year_result) if avg_year_result is not None else None
    
    # Get time periods from metadata
    books_df = pd.read_sql_query('''
        SELECT metadata FROM books WHERE metadata IS NOT NULL
    ''', conn)
    
    # Extract and process time periods
    time_periods = []
    for _, row in books_df.iterrows():
        metadata = json.loads(row['metadata'])
        if 'time_period' in metadata and metadata['time_period']:
            time_periods.append(metadata['time_period'])
    
    # Calculate most common time period
    if time_periods:
        from collections import Counter
        period_counts = Counter(time_periods)
        stats['common_time_period'] = period_counts.most_common(1)[0][0]
        # Calculate percentage of books with time period info
        stats['time_period_coverage'] = len(time_periods) / stats['total_books']
    else:
        stats['common_time_period'] = None
        stats['time_period_coverage'] = 0
    
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

def generate_library_json(conn):
    """Generate a simple JSON file with book titles and authors."""
    c = conn.cursor()
    books = c.execute('SELECT title, author FROM books').fetchall()
    
    library_data = {
        "library": [
            {"title": book[0], "author": book[1]} 
            for book in books
        ],
        "generated_at": datetime.now().isoformat()
    }
    
    # Save to data folder
    json_path = Path("data/library_catalog.json")
    with open(json_path, "w") as f:
        json.dump(library_data, f, indent=2)
    
    return json_path, library_data

def generate_executive_summary(library_data):
    """Generate summary using the library catalog."""
    prompt = f"""You are a librarian experienced in curating eclectic book collections. Analyze this library collection and provide a JSON response with the following structure:
{{
    "summary": "A clear paragraph describing the collection's focus and character (approx 200 words). Be abductive and reflect on your chain of thought.",
    "patterns": [
        "Key Pattern 1",
        "Key Pattern 2",
        "Key Pattern 3"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ]
}}

Books in collection:
{json.dumps(library_data['library'], indent=2)}

Important: Ensure your response contains ONLY valid JSON. Do not include any additional text or formatting."""

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
            
            # Clean the response content
            # Remove any non-JSON text before and after
            content = content.strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                
                # Remove any invalid control characters
                json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    if config.DEBUG_MODE:
                        st.error(f"JSON parsing error: {str(e)}\nContent: {json_str}")
                    return None
            
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
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    error_container = st.empty()
    success_count = 0
    errors = []

    for i, book in enumerate(books):
        try:
            # Update progress
            progress = (i + 1) / total
            progress_bar.progress(progress)
            progress_text.text(f"Processing {i+1} of {total}: {book[1]} by {book[2]}")
            
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
                success_count += 1
            else:
                errors.append(f"No metadata retrieved for: {book[1]} by {book[2]}")
                
        except Exception as e:
            error_msg = f"Error updating {book[1]}: {str(e)}"
            errors.append(error_msg)
            if config.DEBUG_MODE:
                st.error(error_msg)
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Show final results
    if success_count > 0:
        st.success(f"Successfully updated metadata for {success_count} of {total} books")
    
    if errors and config.DEBUG_MODE:
        with error_container.expander("Show Errors"):
            for error in errors:
                st.error(error)
    
    return success_count > 0

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

def create_book_network(conn, category=None):
    """Create a network visualization of book relationships.
    
    Args:
        conn: Database connection
        category: Optional filter for 'Fiction' or 'Non-Fiction'
    """
    G = nx.Graph()
    c = conn.cursor()
    books = c.execute('SELECT id, title, author, metadata FROM books').fetchall()
    
    # Create nodes and track metadata for each book
    book_metadata = {}
    for book in books:
        book_id, title, author, metadata_json = book
        metadata = json.loads(metadata_json)
        
        # Skip books that don't match the category filter
        if category:
            is_fiction = any(g in GENRE_CATEGORIES['Fiction'] for g in metadata.get('genre', []))
            if (category == 'Fiction' and not is_fiction) or (category == 'Non-Fiction' and is_fiction):
                continue
        
        display_name = f"{title}\n{author}"
        G.add_node(display_name)
        
        # Extract year and convert to decade
        year = metadata.get('year')
        decade = f"{str(year)[:-1]}0s" if year else None
        
        # Extract themes
        themes = set(metadata.get('themes', []))
        
        book_metadata[display_name] = {
            'author': author,
            'decade': decade,
            'themes': themes
        }
    
    # Calculate relationships
    for book1 in book_metadata:
        for book2 in book_metadata:
            if book1 >= book2:  # Skip duplicate pairs
                continue
            
            # Calculate relationship scores
            same_author = book_metadata[book1]['author'] == book_metadata[book2]['author']
            same_decade = (book_metadata[book1]['decade'] == book_metadata[book2]['decade'] 
                         and book_metadata[book1]['decade'] is not None)
            shared_themes = len(book_metadata[book1]['themes'] & 
                             book_metadata[book2]['themes'])
            
            # Weight the relationships
            score = (
                same_author * 5 +  # Heavy weight for same author
                same_decade * 3 +  # Medium weight for same decade
                shared_themes * 2  # Weight per shared theme
            )
            
            if score > 0:  # Only create edges with relationships
                G.add_edge(book1, book2, 
                          weight=score,
                          relationship={
                              'same_author': same_author,
                              'same_decade': same_decade,
                              'shared_themes': shared_themes
                          })
    
    return G

def visualize_book_network(G):
    """Create an interactive network visualization with improved legibility."""
    if len(G.nodes()) == 0:
        return None
    
    # Define connection colors
    COLORS = {
        'author': 'rgba(65, 105, 225, 0.8)',   # Royal Blue
        'decade': 'rgba(50, 205, 50, 0.8)',    # Lime Green
        'theme': 'rgba(255, 140, 0, 0.8)'      # Dark Orange
    }
    
    # Calculate node sizes based on weighted connections
    node_weights = {node: 0 for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        node_weights[u] += weight
        node_weights[v] += weight
    
    # Scale node sizes
    max_weight = max(node_weights.values()) if node_weights else 1
    node_sizes = {node: 20 + (weight / max_weight) * 40 for node, weight in node_weights.items()}
    
    # Create layout with more spacing
    pos = nx.spring_layout(G, k=2/sqrt(len(G.nodes())), iterations=50)
    
    # Create separate edge traces for each connection type
    edge_traces = []
    
    for connection_type, color in COLORS.items():
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            relationship = G.edges[edge]['relationship']
            should_include = False
            
            if connection_type == 'author' and relationship['same_author']:
                should_include = True
            elif connection_type == 'decade' and relationship['same_decade']:
                should_include = True
            elif connection_type == 'theme' and relationship['shared_themes'] > 0:
                should_include = True
                
            if should_include:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Enhanced hover text
                hover_text = "<br>".join([
                    f"Connection: {connection_type.title()}",
                    f"Books:",
                    f"- {edge[0].split('<br>')[0]}",
                    f"- {edge[1].split('<br>')[0]}",
                    f"Shared Themes: {relationship['shared_themes']}" if relationship['shared_themes'] > 0 else ""
                ])
                edge_text.extend([hover_text, hover_text, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color=color),
            hoverinfo='text',
            text=edge_text,
            mode='lines',
            name=connection_type.title(),
            showlegend=True
        )
        edge_traces.append(edge_trace)
    
    # Create node trace with improved labels
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Format node text for better readability
        title, author = node.split('\n')
        node_text.append(f"{title}<br>by {author}")
        node_size.append(node_sizes[node])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color='rgba(200, 200, 200, 0.9)',
            line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
        ),
        name='Books'
    )
    
    # Create figure with improved layout
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title='Book Relationship Network',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            annotations=[
                dict(
                    text="Node size indicates number of connections",
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

def ask_library_llm(query, library_data):
    """Query the LLM about the library collection."""
    prompt = f"""You are a knowledgeable librarian assistant. Use the provided library catalog to answer the following query.
If the query cannot be answered using only the library information, clearly state that.

Library Catalog:
{json.dumps(library_data['library'], indent=2)}

Query: {query}

Please provide a clear, concise response based on the available library data."""

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
            return response_data['choices'][0]['message']['content']
        return None
            
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error querying LLM: {str(e)}")
        return None

def extract_and_save_themes(conn) -> dict:
    """Extract all themes from the library and save to JSON."""
    c = conn.cursor()
    books = c.execute('SELECT metadata FROM books WHERE metadata IS NOT NULL').fetchall()
    
    # Collect unique themes
    unique_themes = set()
    for book in books:
        metadata = json.loads(book[0])
        themes = metadata.get('themes', [])
        unique_themes.update(themes)
    
    # Create simplified theme data structure
    theme_data = {
        "themes": sorted(list(unique_themes)),  # Just a simple sorted list of themes
        "generated_at": datetime.now().isoformat(),
        "total_themes": len(unique_themes)
    }
    
    # Save to data folder
    os.makedirs("data", exist_ok=True)
    with open("data/theme_inventory.json", "w") as f:
        json.dump(theme_data, f, indent=2)
    
    return theme_data

def analyze_theme_groupings(theme_data: dict) -> dict:
    """Use LLM to analyze and group themes."""
    # Split themes into chunks if there are too many
    MAX_THEMES_PER_CHUNK = 20
    all_themes = theme_data['themes']
    
    if len(all_themes) > MAX_THEMES_PER_CHUNK:
        # Process themes in chunks and combine results
        combined_analysis = {
            "uber_themes": [],
            "analysis": {
                "summary": "",
                "key_insights": []
            }
        }
        
        # Process themes in chunks
        chunk_summaries = []
        chunk_insights = []
        
        # Process themes in chunks
        for i in range(0, len(all_themes), MAX_THEMES_PER_CHUNK):
            chunk = all_themes[i:i + MAX_THEMES_PER_CHUNK]
            chunk_data = {"themes": chunk}
            
            # Analyze this chunk
            chunk_analysis = analyze_theme_chunk(chunk_data)
            if chunk_analysis:
                # Collect uber-themes
                combined_analysis["uber_themes"].extend(chunk_analysis["uber_themes"])
                
                # Collect summaries and insights for later combination
                if "analysis" in chunk_analysis:
                    if "summary" in chunk_analysis["analysis"]:
                        chunk_summaries.append(chunk_analysis["analysis"]["summary"])
                    if "key_insights" in chunk_analysis["analysis"]:
                        chunk_insights.extend(chunk_analysis["analysis"]["key_insights"])
        
        # Combine summaries and insights
        if chunk_summaries:
            # Create a combined summary prompt
            summary_prompt = f"""Synthesize these theme analysis summaries into a single concise 200-word summary:

{' '.join(chunk_summaries)}"""
            
            # Use LLM to create combined summary
            combined_summary = get_combined_summary(summary_prompt)
            combined_analysis["analysis"]["summary"] = combined_summary or "Analysis summary not available."
        
        # Add unique insights
        if chunk_insights:
            # Remove duplicates while preserving order
            seen = set()
            unique_insights = []
            for insight in chunk_insights:
                if insight not in seen:
                    seen.add(insight)
                    unique_insights.append(insight)
            combined_analysis["analysis"]["key_insights"] = unique_insights[:5]  # Limit to top 5 insights
        
        # Save the combined analysis
        with open("data/theme_analysis.json", "w") as f:
            json.dump(combined_analysis, f, indent=2)
        
        return combined_analysis
    else:
        # Process all themes at once if the list is small enough
        return analyze_theme_chunk(theme_data)

def get_combined_summary(prompt: str) -> str:
    """Use LLM to combine multiple summaries into one."""
    headers = {
        "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use llama-3.1-70b-instruct model for more sophisticated theme analysis
    payload = {
        "model": "llama-3.1-70b-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a literary analyst synthesizing thematic analyses. Provide concise, insightful summaries without citations or references. Focus on patterns and relationships between themes."
            },
            {
                "role": "user",
                "content": prompt + "\n\nProvide a concise synthesis without citations or references. Focus on the relationships and patterns between themes."
            }
        ]
    }
    
    try:
        response = requests.post(
            config.PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=45  # Increased timeout for larger model
        )
        
        if response.status_code == 200:
            response_data = response.json()
            # Clean any remaining citations or references from the response
            content = response_data['choices'][0]['message']['content'].strip()
            # Remove citation patterns like [1], [2], etc.
            content = re.sub(r'\[\d+\]', '', content)
            return content
        return None
        
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error combining summaries: {str(e)}")
        return None

def analyze_theme_chunk(theme_data: dict) -> dict:
    """Analyze a subset of themes."""
    themes_list = "\n".join([
        f"- {theme}" for theme in theme_data['themes']
    ])
    
    prompt = f"""As a literary analyst, examine these themes and group them into meaningful uber-themes.
Be concise and limit your analysis to the most significant patterns.
Aim to create no more than 10 uber-themes total.

Themes to analyze:
{themes_list}

Respond with a valid JSON object using this exact structure:
{{
    "uber_themes": [
        {{
            "name": "Example Theme Group",
            "description": "Single line description",
            "sub_themes": [
                {{
                    "name": "Original Theme",
                    "connection": "Brief note"
                }}
            ]
        }}
    ],
    "analysis": {{
        "summary": "Brief overview",
        "key_insights": [
            "Key point 1",
            "Key point 2",
            "Key point 3"
        ]
    }}
}}

IMPORTANT FORMATTING RULES:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Keep descriptions concise and single-line
5. Ensure all JSON syntax is valid"""

    headers = {
        "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON generator. Always validate your JSON structure before responding."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = requests.post(
            config.PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            # Clean and parse the response
            content = content.strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                
                # Enhanced JSON cleaning
                json_str = (json_str
                    .replace('\n', ' ')  # Remove newlines
                    .replace('\\n', ' ')  # Remove escaped newlines
                    .replace('\t', ' ')   # Remove tabs
                    .replace('\\', '\\\\')  # Escape backslashes
                    .replace('""', '" "')   # Fix empty strings
                )
                
                # Remove C-style comments
                json_str = re.sub(r'/\*.*?\*/', '', json_str)
                # Remove trailing commas before closing braces/brackets
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Remove any remaining comments that might break JSON
                json_str = re.sub(r'//.*?(?=[\n\r]|$)', '', json_str)
                
                # Clean up any multiple spaces created by removals
                json_str = re.sub(r'\s+', ' ', json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    if config.DEBUG_MODE:
                        # Enhanced error reporting
                        st.error(f"""JSON parsing error: {str(e)}
                        
Position: {e.pos}
Line: {e.lineno}
Column: {e.colno}

Problematic section:
{json_str[max(0, e.pos-50):min(len(json_str), e.pos+50)]}

Full content:
{json_str}""")
                    return None
                
        return None
        
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Error analyzing themes chunk: {str(e)}")
        return None

def display_theme_analysis(theme_analysis: dict):
    """Display the theme analysis in an organized way."""
    # First display the summary analysis
    if "analysis" in theme_analysis:
        st.write("### 📚 Thematic Overview")
        st.write(theme_analysis['analysis'].get('summary', ''))
        
        if theme_analysis['analysis'].get('key_insights'):
            st.write("#### Key Insights")
            for insight in theme_analysis['analysis']['key_insights']:
                st.markdown(f"• {insight}")
    
    # Display uber-themes in a single dropdown
    st.write("### 🎯 Thematic Groups")
    
    # Create a selection box for uber-themes
    theme_names = [theme['name'] for theme in theme_analysis['uber_themes']]
    selected_theme = st.selectbox(
        "Select a thematic group to explore",
        theme_names
    )
    
    # Display the selected theme's details
    if selected_theme:
        # Find the selected theme
        theme_details = next(
            (theme for theme in theme_analysis['uber_themes'] 
             if theme['name'] == selected_theme),
            None
        )
        
        if theme_details:
            st.write(f"**Description:** {theme_details['description']}")
            
            # Create a DataFrame for the sub-themes
            sub_themes_data = []
            for theme in theme_details['sub_themes']:
                sub_themes_data.append({
                    'Theme': theme['name'],
                    'Connection': theme['connection']
                })
            
            if sub_themes_data:
                st.write("**Related Themes:**")
                df = pd.DataFrame(sub_themes_data)
                st.dataframe(df, use_container_width=True)

def export_library_to_csv(conn) -> str:
    """Export library data to CSV format."""
    import csv
    from pathlib import Path
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "library_export.csv"
    
    c = conn.cursor()
    # Get all books with their metadata
    books = c.execute('''
        SELECT 
            title, author, year, isbn, publisher, condition, 
            personal_notes, metadata, created_at, updated_at 
        FROM books
    ''').fetchall()
    
    # Define CSV headers
    headers = [
        'Title', 'Author', 'Year', 'ISBN', 'Publisher', 'Condition',
        'Personal Notes', 'Genres', 'Themes', 'Synopsis',
        'Date Added', 'Last Updated'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for book in books:
            # Parse metadata JSON
            metadata = json.loads(book[7]) if book[7] else {}
            
            # Prepare row data
            row = [
                book[0],  # Title
                book[1],  # Author
                book[2],  # Year
                book[3],  # ISBN
                book[4],  # Publisher
                book[5],  # Condition
                book[6],  # Personal Notes
                '; '.join(metadata.get('genre', [])),  # Genres
                '; '.join(metadata.get('themes', [])),  # Themes
                metadata.get('synopsis', ''),  # Synopsis
                book[8][:10] if book[8] else '',  # Date Added (date only)
                book[9][:10] if book[9] else ''   # Last Updated (date only)
            ]
            writer.writerow(row)
    
    return str(csv_path)

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
            ["Add Book", "View Collection", "Analytics", "Network View", 
             "Executive Summary", "Ask the Library"]
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
                    # Update to include ISBN in the API call
                    enhanced_data = enhance_book_data(title, author, year, isbn)
                    
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
        
        # Add export button in a container at the top
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("📥 Export to CSV"):
                    try:
                        csv_path = export_library_to_csv(conn)
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
                        if config.DEBUG_MODE:
                            st.error(f"Error generating CSV: {str(e)}")
                        else:
                            st.error("Error generating CSV file")
        
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
                        try:
                            st.image(book[7], caption="Cover Image")
                        except Exception as e:
                            if config.DEBUG_MODE:
                                st.error(f"Error displaying local cover image: {str(e)}")
                            st.info("Cover image not available")
                    elif book[8]:  # Check if metadata exists
                        try:
                            metadata = json.loads(book[8])
                            if 'cover_url' in metadata and metadata['cover_url']:
                                try:
                                    st.image(metadata['cover_url'], caption="Cover Image")
                                except Exception as e:
                                    if config.DEBUG_MODE:
                                        st.error(f"Error displaying cover from URL: {str(e)}")
                                    st.info("Cover image not available")
                        except json.JSONDecodeError as e:
                            if config.DEBUG_MODE:
                                st.error(f"Error parsing metadata: {str(e)}")
                            st.info("Cover image not available")
                    
                    # Basic book information
                    st.write("**Basic Information**")
                    st.write(f"**Title:** {book[1]}")
                    st.write(f"**Author:** {book[2]}")
                    st.write(f"**Year:** {book[3] if book[3] else 'Unknown'}")
                    st.write(f"**ISBN:** {book[4] if book[4] else 'N/A'}")
                    st.write(f"**Publisher:** {book[5] if book[5] else 'Unknown'}")
                    st.write(f"**Condition:** {book[6]}")
                    st.write(f"**Added:** {book[9][:10]}")  # Show only date part
                
                with col2:
                    metadata = json.loads(book[8])
                    
                    # Add Personal Notes section first if they exist
                    if book[11]:  # personal_notes column
                        st.write("**Personal Notes:**")
                        st.write(f"_{book[11]}_")
                        st.write("")  # Add extra space after notes
                    
                    # Enhanced information from APIs
                    st.write("**LLM Generated Information**")
                    
                    # Synopsis
                    st.write("**Synopsis:**")
                    st.write(metadata.get('synopsis', 'No synopsis available'))
                    
                    if metadata.get('themes'):
                        st.write("**Themes:**")
                        for theme in metadata['themes']:
                            st.write(f"- {theme}")
                    
                    if metadata.get('genre'):
                        st.write("**Genre:**")
                        for genre in metadata['genre']:
                            st.write(f"- {genre}")
                    
                    # Historical Context
                    if metadata.get('historical_context'):
                        st.write("**Historical Context:**")
                        st.write(metadata['historical_context'])
                    
                    # Reading Level and Time Period
                    col3, col4 = st.columns(2)
                    with col3:
                        if metadata.get('reading_level'):
                            st.write("**Reading Level:**")
                            st.write(metadata['reading_level'])
                    with col4:
                        if metadata.get('time_period'):
                            st.write("**Time Period:**")
                            st.write(metadata['time_period'])
                    
                    # LLM Related Books suggestions
                    if metadata.get('related_works'):
                        st.write("**LLM Suggested Related Books:**")
                        for work in metadata['related_works']:
                            if isinstance(work, dict):
                                # Handle structured JSON format
                                st.write(f"- **{work.get('title', '')}** by {work.get('author', '')}")
                                if work.get('reason'):
                                    st.write(f"  _{work['reason']}_")
                            else:
                                # Handle simple string format
                                st.write(f"- {work}")
                    
                    # Add Related Books from catalog
                    related_books = find_related_books(conn, book[0], metadata)
                    if related_books:
                        st.write("**Similar Books in Your Collection:**")
                        for related in related_books:
                            st.write(f"- **{related['title']}** by {related['author']}")
                    
                    # Keywords (if present)
                    if metadata.get('keywords'):
                        st.write("**Keywords:**")
                        st.write(", ".join(metadata['keywords']))
                    
                    # Sources used (at end)
                    if "sources" in metadata:
                        st.write("**Data Sources:**", ", ".join(metadata["sources"]))
                
                # Add action buttons at bottom of col1
                st.divider()
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("🗑 Delete", key=f"del_{book[0]}"):
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
                                # Update all relevant fields from the enhanced data
                                c.execute('''
                                    UPDATE books 
                                    SET metadata = ?,
                                        title = COALESCE(?, title),
                                        author = COALESCE(?, author),
                                        year = COALESCE(?, year),
                                        isbn = COALESCE(?, isbn),
                                        publisher = COALESCE(?, publisher),
                                        updated_at = ?
                                    WHERE id = ?
                                ''', (
                                    json.dumps(enhanced_data),
                                    enhanced_data.get('title'),
                                    enhanced_data.get('author'),
                                    enhanced_data.get('year'),
                                    enhanced_data.get('isbn'),
                                    enhanced_data.get('publisher'),
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
        
        # Add Theme Analysis section
        st.subheader("Theme Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Extract Current Themes"):
                with st.spinner("Extracting themes from library..."):
                    theme_data = extract_and_save_themes(conn)
                    st.success(f"Extracted {len(theme_data['themes'])} unique themes!")
                    
                    # Show download button for theme inventory
                    with open("data/theme_inventory.json", "r") as f:
                        st.download_button(
                            "Download Theme Inventory",
                            f.read(),
                            "theme_inventory.json",
                            "application/json"
                        )
        
        with col2:
            if st.button("Analyze Theme Groupings"):
                try:
                    # Load existing theme inventory
                    with open("data/theme_inventory.json", "r") as f:
                        theme_data = json.load(f)
                    
                    with st.spinner("Analyzing themes..."):
                        theme_analysis = analyze_theme_groupings(theme_data)
                        if theme_analysis:
                            st.success("Theme analysis complete!")
                        else:
                            st.error("Failed to analyze themes")
                except FileNotFoundError:
                    st.error("Please extract themes first")
        
        # Display existing theme analysis if available
        try:
            with open("data/theme_analysis.json", "r") as f:
                theme_analysis = json.load(f)
                st.divider()
                display_theme_analysis(theme_analysis)
        except FileNotFoundError:
            if os.path.exists("data/theme_inventory.json"):
                st.info("Theme inventory exists. Click 'Analyze Theme Groupings' to generate analysis.")
            else:
                st.info("No theme analysis available. Start by clicking 'Extract Current Themes'.")
        
        st.divider()
        
        # Get analytics data
        stats, genre_counts, theme_counts = generate_analytics(conn)
        
        # Basic metrics with enhanced time information
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Total Books", stats['total_books'])
        col2.metric("Unique Authors", stats['unique_authors'])
        
        # Display publication year info
        if stats['avg_pub_year']:
            col3.metric("Avg. Publication Year", stats['avg_pub_year'])
        else:
            col3.metric("Avg. Publication Year", "N/A")
        
        # Display time period info
        if stats['common_time_period']:
            col4.metric(
                "Common Time Period", 
                stats['common_time_period'],
                f"{round(stats['time_period_coverage'] * 100)}% coverage"
            )
        else:
            col4.metric("Common Time Period", "N/A")
        
        # Fiction vs Non-Fiction ratio
        fiction_percentage = round(stats['fiction_ratio'] * 100)
        nonfiction_percentage = round(stats['nonfiction_ratio'] * 100)
        col5.metric("Fiction/Non-Fiction", 
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
        
        **Connection Types:**
        - 👤 Author (Strong connection)
        - 📅 Decade (Medium connection)
        - 🎭 Themes (Connection strength based on number of shared themes)
        
        **How to Read:**
        - Larger nodes indicate books with more connections
        - Different colored lines show different types of connections
        - Hover over nodes to see book details
        - Hover over lines to see connection details
        """)
        
        # Add view selection
        view_type = st.radio(
            "Select View",
            ["All Books", "Fiction Only", "Non-Fiction Only"],
            horizontal=True
        )
        
        # Create and display network
        with st.spinner("Generating network visualization..."):
            category = None
            if view_type == "Fiction Only":
                category = "Fiction"
            elif view_type == "Non-Fiction Only":
                category = "Non-Fiction"
            
            G = create_book_network(conn, category)
            fig = visualize_book_network(G)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics
                st.subheader("Network Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Books", len(G.nodes()))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    if len(G.nodes()) > 0:
                        avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
            else:
                st.info("Add more books to see their relationships!")
    
    elif page == "Executive Summary":
        st.header("Library Executive Summary")
        
        # Add statistics section at the top
        stats, genre_counts, _ = generate_analytics(conn)
        
        # Display key metrics in columns
        st.subheader("Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Books", stats['total_books'])
        with col2:
            st.metric("Unique Authors", stats['unique_authors'])
        with col3:
            st.metric("Average Publication Year", 
                     stats['avg_pub_year'] if stats['avg_pub_year'] is not None else "N/A")
        with col4:
            fiction_percentage = round(stats['fiction_ratio'] * 100)
            nonfiction_percentage = round(stats['nonfiction_ratio'] * 100)
            st.metric("Fiction/Non-Fiction", 
                     f"{fiction_percentage}% / {nonfiction_percentage}%")
        
        st.divider()
        
        # Existing catalog and summary generation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Library Catalog"):
                with st.spinner("Generating catalog..."):
                    json_path, library_data = generate_library_json(conn)
                    st.success("Library catalog generated!")
                    
                    # Offer download
                    with open(json_path, "r") as f:
                        st.download_button(
                            "Download Library Catalog",
                            f.read(),
                            "library_catalog.json",
                            "application/json"
                        )
        
        with col2:
            if st.button("Generate Summary"):
                try:
                    # Load existing catalog
                    with open("data/library_catalog.json", "r") as f:
                        library_data = json.load(f)
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_executive_summary(library_data)
                        if summary:
                            # Store the summary
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
        
        # Display existing summary if available
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
    
    elif page == "Ask the Library":
        st.header("Ask the Large Library Model")
        
        st.markdown("""
        Ask questions about your library collection. For example:
        - What themes are common in my collection?
        - Which authors do I read most?
        - What genres are underrepresented?
        - Suggest books from my collection for a specific mood or topic
        """)
        
        # Query input
        query = st.text_area("Enter your question:", height=100)
        
        if st.button("Ask"):
            try:
                # Check for library catalog
                with open("data/library_catalog.json", "r") as f:
                    library_data = json.load(f)
                
                with st.spinner("Analyzing your library..."):
                    response = ask_library_llm(query, library_data)
                    if response:
                        st.markdown("### Response")
                        st.markdown(response, unsafe_allow_html=True)
                    else:
                        st.error("Failed to get a response")
                        
            except FileNotFoundError:
                st.error("Library catalog not found. Please generate it first in the Executive Summary page.")

if __name__ == "__main__":
    main() 