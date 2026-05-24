"""Network View page: book-relationship graph."""
import streamlit as st

from analytics import create_book_network, visualize_book_network
from logger import get_logger
from shelflife_app.services import get_database
from shelflife_app.ui_helpers import render_empty_state, render_page_header

logger = get_logger(__name__)

db = get_database()


render_page_header(
    "Book Relationships",
    "Visualize connections between books in your collection",
    "🔗"
)

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
        st.stop()

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

            st.markdown("### Network Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Books", len(G.nodes()))
            with col2:
                st.metric("Connections", len(G.edges()))
            with col3:
                if len(G.nodes()) > 0:
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
