"""Analytics page: charts and AI-assisted theme analysis."""
import json

import plotly.express as px
import streamlit as st

from analytics import generate_analytics
from book_service import extract_and_save_themes
from logger import get_logger
from shelflife_app.library_utils import display_theme_analysis
from shelflife_app.services import get_book_service, get_database
from shelflife_app.ui_helpers import CHART_COLORS, render_empty_state, render_page_header

logger = get_logger(__name__)

db = get_database()
book_service = get_book_service()


render_page_header(
    "Library Analytics",
    "Discover patterns and insights in your collection",
    "📊"
)

try:
    books = db.get_all_books()

    if not books:
        render_empty_state(
            "📊",
            "No Data to Analyze",
            "Add some books to your library to see analytics and insights."
        )
        st.stop()

    stats, genre_counts, theme_counts = generate_analytics(books)

    st.markdown("### Quick Stats")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Books", stats.total_books)
    col2.metric("Authors", stats.unique_authors)
    col3.metric("Avg. Year", stats.avg_pub_year or "N/A")

    if stats.common_time_period:
        col4.metric(
            "Era",
            stats.common_time_period,
            f"{round(stats.time_period_coverage * 100)}%"
        )
    else:
        col4.metric("Era", "N/A")

    fiction_pct = round(stats.fiction_ratio * 100)
    nonfiction_pct = round(stats.nonfiction_ratio * 100)
    col5.metric("Fiction", f"{fiction_pct}%", f"{nonfiction_pct}% Non-Fiction")

    st.markdown("---")

    if not genre_counts.empty:
        st.markdown("### Genre Distribution")

        tab1, tab2, tab3 = st.tabs(["📊 Combined View", "📖 Fiction", "📚 Non-Fiction"])

        with tab1:
            fig = px.sunburst(
                genre_counts,
                path=['category', 'genre'],
                values='count',
                color='count',
                color_continuous_scale=['#E0E7FF', '#6366F1', '#312E81']
            )
            fig.update_layout(
                margin=dict(t=30, l=0, r=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter")
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fiction_data = genre_counts[genre_counts['category'] == 'Fiction']
            if not fiction_data.empty:
                fig_fiction = px.pie(
                    fiction_data,
                    values='count',
                    names='genre',
                    color_discrete_sequence=CHART_COLORS
                )
                fig_fiction.update_layout(
                    margin=dict(t=30, l=0, r=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                )
                fig_fiction.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_fiction, use_container_width=True)
            else:
                st.info("No fiction books in your collection yet.")

        with tab3:
            nonfiction_data = genre_counts[genre_counts['category'] == 'Non-Fiction']
            if not nonfiction_data.empty:
                fig_nonfiction = px.pie(
                    nonfiction_data,
                    values='count',
                    names='genre',
                    color_discrete_sequence=CHART_COLORS
                )
                fig_nonfiction.update_layout(
                    margin=dict(t=30, l=0, r=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                )
                fig_nonfiction.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_nonfiction, use_container_width=True)
            else:
                st.info("No non-fiction books in your collection yet.")

    if not theme_counts.empty:
        st.markdown("### Theme Distribution")
        fig_themes = px.treemap(
            theme_counts,
            path=['theme'],
            values='count',
            color='count',
            color_continuous_scale=['#DBEAFE', '#3B82F6', '#1E40AF']
        )
        fig_themes.update_layout(
            margin=dict(t=30, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_themes, use_container_width=True)

    st.markdown("---")

    st.markdown("### Theme Analysis")
    st.write("Extract and analyze thematic patterns across your library.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Extract Themes", use_container_width=True):
            try:
                with st.spinner("Analyzing book metadata..."):
                    books_with_meta = db.get_books_with_metadata()
                    theme_data = extract_and_save_themes(books_with_meta)

                st.success(f"Extracted {len(theme_data['themes'])} unique themes!")

                with open("data/theme_inventory.json", "r") as f:
                    st.download_button(
                        "⬇️ Download Theme Inventory",
                        f.read(),
                        "theme_inventory.json",
                        "application/json",
                        use_container_width=True
                    )
            except Exception as e:
                logger.error(f"Error extracting themes: {str(e)}", exc_info=True)
                st.error("Failed to extract themes")

    with col2:
        if st.button("🧠 Analyze Groupings", use_container_width=True):
            try:
                with open("data/theme_inventory.json", "r") as f:
                    theme_data = json.load(f)

                with st.status("AI is identifying patterns...", expanded=False) as status:
                    theme_analysis = book_service.analyze_themes(theme_data['themes'])

                    if not theme_analysis:
                        status.update(label="Failed to analyze themes.", state="error")
                        st.error("Failed to analyze themes")
                        st.stop()

                    status.update(label="Saving analysis...")
                    with open("data/theme_analysis.json", "w") as f:
                        json.dump(theme_analysis, f, indent=2)
                    status.update(label="Theme analysis complete.", state="complete")

                st.rerun()
            except FileNotFoundError:
                st.error("Please extract themes first")
            except Exception as e:
                logger.error(f"Error analyzing themes: {str(e)}", exc_info=True)
                st.error("Failed to analyze themes")

    try:
        with open("data/theme_analysis.json", "r") as f:
            theme_analysis = json.load(f)
            st.markdown("---")
            display_theme_analysis(theme_analysis)
    except FileNotFoundError:
        st.info("💡 Click 'Extract Themes' to discover thematic patterns in your library.")
    except Exception as e:
        logger.error(f"Error loading theme analysis: {str(e)}")

except Exception as e:
    logger.error(f"Error generating analytics: {str(e)}", exc_info=True)
    st.error("Error generating analytics")
