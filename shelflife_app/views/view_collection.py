"""View Collection page: browse, search, edit, and delete books."""
import json
import time

import streamlit as st

from analytics import export_library_to_csv
from database import DatabaseError
from logger import get_logger
from shelflife_app.library_utils import find_related_books
from shelflife_app.services import get_book_service, get_database
from shelflife_app.ui_helpers import render_condition_badge, render_empty_state, render_page_header
from validation import sanitize_string, validate_search_term

logger = get_logger(__name__)

db = get_database()
book_service = get_book_service()

if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None


render_page_header(
    "Your Library",
    "Browse, search, and manage your book collection",
    "📚"
)

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

if search:
    is_valid, error = validate_search_term(search)
    if not is_valid:
        st.error(f"Invalid search term: {error}")
        st.stop()
    search = sanitize_string(search, 200)

st.markdown("---")

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
                st.switch_page("shelflife_app/views/add_book.py")
        st.stop()

    st.markdown(f"**{len(books)}** book{'s' if len(books) != 1 else ''} in your library")

    for book in books:
        book_id = book[0]

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
                if book[7]:
                    try:
                        st.image(book[7], use_container_width=True)
                    except Exception as e:
                        logger.warning(f"Error displaying cover image: {str(e)}")
                        st.markdown('<div style="background: #F3F4F6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #9CA3AF;">No Cover</div>', unsafe_allow_html=True)
                elif book[8]:
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

                condition = book[6] if book[6] else "Unknown"
                st.markdown(f"**Condition:** {render_condition_badge(condition)}", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"**Year:** {book[3] if book[3] else 'Unknown'}")
                st.markdown(f"**ISBN:** {book[4] if book[4] else 'N/A'}")
                st.markdown(f"**Publisher:** {book[5] if book[5] else 'Unknown'}")
                st.markdown(f"**Added:** {book[9][:10] if book[9] else 'N/A'}")

            with col2:
                if book[8]:
                    metadata = json.loads(book[8])

                    if len(book) > 11 and book[11]:
                        st.markdown("##### 📝 Personal Notes")
                        st.info(book[11])

                    if metadata.get('synopsis'):
                        st.markdown("##### 📖 Synopsis")
                        st.write(metadata['synopsis'])

                    if metadata.get('genre'):
                        st.markdown("##### 🏷️ Genres")
                        genre_tags = " ".join([f'`{g}`' for g in metadata['genre']])
                        st.markdown(genre_tags)

                    if metadata.get('themes'):
                        st.markdown("##### 🎭 Themes")
                        theme_tags = " ".join([f'`{t}`' for t in metadata['themes']])
                        st.markdown(theme_tags)

                    if metadata.get('historical_context'):
                        st.markdown("##### 🏛️ Historical Context")
                        st.write(metadata['historical_context'])

                    if metadata.get('related_works'):
                        st.markdown("##### 📚 Related Books (AI Suggestions)")
                        for work in metadata['related_works'][:3]:
                            if isinstance(work, dict):
                                st.markdown(f"- **{work.get('title', '')}** by {work.get('author', '')}")
                                if work.get('reason'):
                                    st.caption(f"  _{work['reason']}_")
                            else:
                                st.markdown(f"- {work}")

                    related_books = find_related_books(db, book[0], metadata)
                    if related_books:
                        st.markdown("##### 🔗 Similar in Your Collection")
                        for related in related_books:
                            st.markdown(f"- **{related['title']}** by {related['author']}")

                    if "sources" in metadata:
                        st.caption(f"Data sources: {', '.join(metadata['sources'])}")

            st.markdown("---")
            col3, col4, col5 = st.columns(3)

            with col3:
                if st.button("🗑️ Delete", key=f"del_{book[0]}", use_container_width=True):
                    st.session_state.delete_confirm = book[0]
                    st.rerun()

            with col5:
                if st.button("🔄 Refresh Metadata", key=f"refresh_{book[0]}", use_container_width=True):
                    try:
                        with st.status("Refreshing metadata...", expanded=False) as status:
                            status.update(label="Fetching latest data from AI...")
                            enhanced_metadata = book_service.enhance_book_data(
                                book[1], book[2], book[3], book[4]
                            )

                            if not enhanced_metadata:
                                status.update(label="Failed to refresh metadata.", state="error")
                                st.error("Failed to refresh metadata")
                                st.stop()

                            status.update(label="Updating database...")
                            db.update_metadata(book[0], enhanced_metadata)
                            status.update(label="Metadata refreshed.", state="complete")

                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error refreshing metadata: {str(e)}", exc_info=True)
                        st.error("Failed to refresh metadata")

except DatabaseError as e:
    st.error(f"Database error: {str(e)}")
except Exception as e:
    logger.error(f"Error rendering collection: {str(e)}", exc_info=True)
    st.error("An error occurred while loading your collection")
