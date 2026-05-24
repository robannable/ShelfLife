"""View Collection page: browse, search, edit, and delete books.

Each book card is wrapped in @st.fragment so refresh-metadata and
delete-confirm interactions rerun only the affected card rather than the
whole page (which would re-query all books and re-render every expander).
"""
import json

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

NO_COVER_HTML = (
    '<div style="background: #F3F4F6; height: 200px; display: flex; '
    'align-items: center; justify-content: center; border-radius: 8px; '
    'color: #9CA3AF;">No Cover</div>'
)


def _render_cover(book):
    """Render the cover image or a placeholder."""
    if book[7]:
        try:
            st.image(book[7], use_container_width=True)
            return
        except Exception as e:
            logger.warning(f"Error displaying cover image: {str(e)}")
    elif book[8]:
        try:
            metadata = json.loads(book[8])
            cover_url = metadata.get('cover_url')
            if cover_url:
                st.image(cover_url, use_container_width=True)
                return
        except Exception:
            pass
    st.markdown(NO_COVER_HTML, unsafe_allow_html=True)


@st.fragment
def render_book_card(book_id: int):
    """Render a single book's card. Reruns scoped to this card on most actions."""
    try:
        book = db.get_book(book_id)
    except DatabaseError as e:
        st.error(f"Failed to load book: {str(e)}")
        return

    if not book:
        # Book was deleted elsewhere; nothing to render.
        return

    confirm_key = f"confirm_delete_{book_id}"

    if st.session_state.get(confirm_key):
        st.warning(f"⚠️ **Delete '{book[1]}'?** This action cannot be undone.")
        col_yes, col_no, _ = st.columns([1, 1, 3])
        with col_yes:
            if st.button("🗑️ Yes, Delete", key=f"confirm_del_{book_id}", type="primary"):
                try:
                    db.delete_book(book_id)
                    st.session_state[confirm_key] = False
                    st.toast(f"'{book[1]}' deleted.", icon="🗑️")
                    st.rerun(scope="app")  # book must vanish from the outer list
                except Exception as e:
                    st.error(f"Error deleting book: {str(e)}")
        with col_no:
            if st.button("Cancel", key=f"cancel_del_{book_id}"):
                st.session_state[confirm_key] = False
                st.rerun()
        return

    with st.expander(f"**{book[1]}** by {book[2]}", expanded=False):
        col_a, col_b = st.columns([1, 2])

        with col_a:
            _render_cover(book)

            condition = book[6] if book[6] else "Unknown"
            st.markdown(f"**Condition:** {render_condition_badge(condition)}", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Year:** {book[3] if book[3] else 'Unknown'}")
            st.markdown(f"**ISBN:** {book[4] if book[4] else 'N/A'}")
            st.markdown(f"**Publisher:** {book[5] if book[5] else 'Unknown'}")
            st.markdown(f"**Added:** {book[9][:10] if book[9] else 'N/A'}")

        with col_b:
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
                    st.markdown(" ".join(f'`{g}`' for g in metadata['genre']))

                if metadata.get('themes'):
                    st.markdown("##### 🎭 Themes")
                    st.markdown(" ".join(f'`{t}`' for t in metadata['themes']))

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

                related = find_related_books(db, book_id, metadata)
                if related:
                    st.markdown("##### 🔗 Similar in Your Collection")
                    for r in related:
                        st.markdown(f"- **{r['title']}** by {r['author']}")

                if "sources" in metadata:
                    st.caption(f"Data sources: {', '.join(metadata['sources'])}")

        st.markdown("---")
        col_del, _, col_refresh = st.columns(3)

        with col_del:
            if st.button("🗑️ Delete", key=f"del_{book_id}", use_container_width=True):
                st.session_state[confirm_key] = True
                st.rerun()

        with col_refresh:
            if st.button("🔄 Refresh Metadata", key=f"refresh_{book_id}", use_container_width=True):
                try:
                    with st.status("Refreshing metadata...", expanded=False) as status:
                        status.update(label="Fetching latest data from AI...")
                        enhanced = book_service.enhance_book_data(
                            book[1], book[2], book[3], book[4]
                        )

                        if not enhanced:
                            status.update(label="Failed to refresh metadata.", state="error")
                            st.error("Failed to refresh metadata")
                            st.stop()

                        status.update(label="Updating database...")
                        db.update_metadata(book_id, enhanced)
                        status.update(label="Metadata refreshed.", state="complete")

                    st.toast("Metadata refreshed.", icon="✅")
                    st.rerun()  # fragment-scoped: re-fetches book with updated metadata
                except Exception as e:
                    logger.error(f"Error refreshing metadata: {str(e)}", exc_info=True)
                    st.error("Failed to refresh metadata")


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
        render_book_card(book[0])

except DatabaseError as e:
    st.error(f"Database error: {str(e)}")
except Exception as e:
    logger.error(f"Error rendering collection: {str(e)}", exc_info=True)
    st.error("An error occurred while loading your collection")
