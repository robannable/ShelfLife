"""Library-wide helpers: catalog export, related-book lookups, theme display."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from database import Database
from logger import get_logger

logger = get_logger(__name__)


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
            if book[0] == current_book_id:
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
