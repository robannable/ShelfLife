"""
Analytics and visualization functions for ShelfLife.
"""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from math import sqrt
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from models import LibraryStats
from constants import GENRE_CATEGORIES
from logger import get_logger

logger = get_logger(__name__)


def generate_analytics(books_data: List[Tuple]) -> Tuple[LibraryStats, pd.DataFrame, pd.DataFrame]:
    """
    Generate comprehensive analytics from book data.

    Args:
        books_data: List of book tuples from database

    Returns:
        Tuple of (LibraryStats, genre_counts_df, theme_counts_df)
    """
    try:
        stats = LibraryStats()
        stats.total_books = len(books_data)

        # Calculate unique authors
        authors = set()
        years = []
        time_periods = []
        fiction_genres = []
        nonfiction_genres = []
        themes = []

        for book in books_data:
            authors.add(book[2])  # author column

            # Year
            if book[3]:  # year column
                years.append(book[3])

            # Parse metadata
            if book[8]:  # metadata column
                try:
                    metadata = json.loads(book[8])

                    # Time periods
                    if 'time_period' in metadata and metadata['time_period']:
                        time_periods.append(metadata['time_period'])

                    # Genres
                    if 'genre' in metadata:
                        for genre in metadata['genre']:
                            if genre in GENRE_CATEGORIES.get('Fiction', []):
                                fiction_genres.append(genre)
                            elif genre in GENRE_CATEGORIES.get('Non-Fiction', []):
                                nonfiction_genres.append(genre)

                    # Themes
                    if 'themes' in metadata:
                        themes.extend(metadata['themes'])

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in metadata for book ID {book[0]}")
                    continue

        stats.unique_authors = len(authors)
        stats.avg_pub_year = round(sum(years) / len(years)) if years else None

        # Time period statistics
        if time_periods:
            period_counts = Counter(time_periods)
            stats.common_time_period = period_counts.most_common(1)[0][0]
            stats.time_period_coverage = len(time_periods) / stats.total_books
        else:
            stats.common_time_period = None
            stats.time_period_coverage = 0

        # Fiction vs non-fiction ratio
        total_categorized = len(fiction_genres) + len(nonfiction_genres)
        if total_categorized > 0:
            stats.fiction_ratio = len(fiction_genres) / total_categorized
            stats.nonfiction_ratio = len(nonfiction_genres) / total_categorized
        else:
            stats.fiction_ratio = 0
            stats.nonfiction_ratio = 0

        # Create genre counts DataFrame
        fiction_df = pd.DataFrame(fiction_genres, columns=['genre'])
        fiction_counts = fiction_df['genre'].value_counts().reset_index()
        fiction_counts.columns = ['genre', 'count']
        fiction_counts['category'] = 'Fiction'

        nonfiction_df = pd.DataFrame(nonfiction_genres, columns=['genre'])
        nonfiction_counts = nonfiction_df['genre'].value_counts().reset_index()
        nonfiction_counts.columns = ['genre', 'count']
        nonfiction_counts['category'] = 'Non-Fiction'

        genre_counts = pd.concat([fiction_counts, nonfiction_counts])

        # Create theme counts DataFrame
        theme_df = pd.DataFrame(themes, columns=['theme'])
        theme_counts = theme_df['theme'].value_counts().reset_index()
        theme_counts.columns = ['theme', 'count']

        logger.info(f"Generated analytics for {stats.total_books} books")
        return stats, genre_counts, theme_counts

    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}", exc_info=True)
        # Return empty data structures on error
        return LibraryStats(), pd.DataFrame(), pd.DataFrame()


def create_book_network(books_data: List[Tuple], category: Optional[str] = None) -> nx.Graph:
    """
    Create a network graph of book relationships.

    Args:
        books_data: List of book tuples from database
        category: Optional filter for 'Fiction' or 'Non-Fiction'

    Returns:
        NetworkX graph of book relationships
    """
    try:
        G = nx.Graph()
        book_metadata = {}

        for book in books_data:
            book_id, title, author, year = book[0], book[1], book[2], book[3]

            try:
                metadata_json = book[8]
                if not metadata_json:
                    continue

                metadata = json.loads(metadata_json)

                # Filter by category if specified
                if category:
                    book_genres = metadata.get('genre', [])
                    is_fiction = any(g in GENRE_CATEGORIES.get('Fiction', []) for g in book_genres)
                    if (category == 'Fiction' and not is_fiction) or \
                       (category == 'Non-Fiction' and is_fiction):
                        continue

                display_name = f"{title}\n{author}"
                G.add_node(display_name)

                # Extract and process year to decade
                decade = f"{str(year)[:-1]}0s" if year else None

                # Extract themes
                themes = set(metadata.get('themes', []))

                book_metadata[display_name] = {
                    'author': author,
                    'decade': decade,
                    'themes': themes
                }

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing metadata for book {title}: {str(e)}")
                continue

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

        logger.info(f"Created book network with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G

    except Exception as e:
        logger.error(f"Error creating book network: {str(e)}", exc_info=True)
        return nx.Graph()


def visualize_book_network(G: nx.Graph) -> Optional[go.Figure]:
    """Create an interactive network visualization."""
    if len(G.nodes()) == 0:
        logger.warning("Empty graph provided for visualization")
        return None

    try:
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

        # Create layout with spacing
        pos = nx.spring_layout(G, k=2/sqrt(len(G.nodes())), iterations=50)

        # Create edge traces for each connection type
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

                    hover_text = "<br>".join([
                        f"Connection: {connection_type.title()}",
                        f"Books:",
                        f"- {edge[0].split(chr(10))[0]}",
                        f"- {edge[1].split(chr(10))[0]}",
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

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

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

        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title='Book Relationship Network',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
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

        logger.info("Book network visualization created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error visualizing book network: {str(e)}", exc_info=True)
        return None


def export_library_to_csv(books_data: List[Tuple], output_path: Optional[str] = None) -> str:
    """Export library data to CSV format."""
    import csv

    try:
        if output_path is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            output_path = str(data_dir / "library_export.csv")

        headers = [
            'Title', 'Author', 'Year', 'ISBN', 'Publisher', 'Condition',
            'Personal Notes', 'Genres', 'Themes', 'Synopsis',
            'Date Added', 'Last Updated'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for book in books_data:
                # Parse metadata JSON
                metadata = json.loads(book[8]) if book[8] else {}

                row = [
                    book[1],  # Title
                    book[2],  # Author
                    book[3],  # Year
                    book[4],  # ISBN
                    book[5],  # Publisher
                    book[6],  # Condition
                    book[11] if len(book) > 11 else '',  # Personal Notes
                    '; '.join(metadata.get('genre', [])),
                    '; '.join(metadata.get('themes', [])),
                    metadata.get('synopsis', ''),
                    book[9][:10] if book[9] else '',  # Date Added
                    book[10][:10] if book[10] else ''  # Last Updated
                ]
                writer.writerow(row)

        logger.info(f"Exported {len(books_data)} books to CSV: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}", exc_info=True)
        raise
