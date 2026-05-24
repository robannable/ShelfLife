"""Executive Summary page: catalog export + AI-generated library overview."""
import json
from datetime import datetime

import streamlit as st

from analytics import generate_analytics
from logger import get_logger
from shelflife_app.library_utils import generate_library_json
from shelflife_app.services import get_book_service, get_database
from shelflife_app.ui_helpers import render_empty_state, render_page_header

logger = get_logger(__name__)

db = get_database()
book_service = get_book_service()


render_page_header(
    "Executive Summary",
    "Get a high-level overview of your library",
    "📋"
)

try:
    books = db.get_all_books()

    if not books:
        render_empty_state(
            "📋",
            "No Books Yet",
            "Add books to your library to generate an executive summary."
        )
        st.stop()

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
            with st.spinner("Compiling book data..."):
                json_path, library_data = generate_library_json(db)

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

            with st.status("AI is reviewing your collection...", expanded=False) as status:
                summary = book_service.generate_executive_summary(library_data)

                if not summary:
                    status.update(label="Failed to generate summary.", state="error")
                    st.error("Failed to generate summary")
                    st.stop()

                status.update(label="Saving insights...")
                summary_info = {
                    "last_updated": datetime.now().isoformat(),
                    "summary": summary
                }
                with open("data/executive_summary.json", "w") as f:
                    json.dump(summary_info, f, indent=2)

                status.update(label="Summary generated.", state="complete")

            st.rerun()
        except FileNotFoundError:
            st.error("Please generate library catalog first")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            st.error("Failed to generate summary")

st.markdown("---")

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
