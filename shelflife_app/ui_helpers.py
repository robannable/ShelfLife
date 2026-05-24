"""Reusable Streamlit UI components and visual constants."""
import streamlit as st


CHART_COLORS = [
    '#6366F1',  # Primary indigo
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
    '#F59E0B',  # Amber
    '#10B981',  # Emerald
    '#3B82F6',  # Blue
    '#EF4444',  # Red
    '#14B8A6',  # Teal
    '#F97316',  # Orange
    '#84CC16',  # Lime
]

CONDITION_BADGES = {
    "New": ("badge-new", "Mint"),
    "Like New": ("badge-new", "Like New"),
    "Very Good": ("badge-good", "Very Good"),
    "Good": ("badge-good", "Good"),
    "Fair": ("badge-fair", "Fair"),
    "Poor": ("badge-poor", "Poor"),
}


def render_page_header(title: str, subtitle: str = None, icon: str = None):
    """Render a styled page header."""
    icon_html = f'<span style="margin-right: 12px;">{icon}</span>' if icon else ''
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ''

    st.markdown(f'''
        <div class="app-header">
            <h1>{icon_html}{title}</h1>
            {subtitle_html}
        </div>
    ''', unsafe_allow_html=True)


def render_empty_state(icon: str, title: str, message: str, show_button: bool = False, button_label: str = ""):
    """Render an empty state component. Returns True if the optional button was clicked."""
    st.markdown(f'''
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{message}</p>
        </div>
    ''', unsafe_allow_html=True)

    if show_button:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.button(button_label, use_container_width=True)
    return False


def render_condition_badge(condition: str) -> str:
    """Return HTML for a condition badge."""
    badge_class, label = CONDITION_BADGES.get(condition, ("badge-fair", condition))
    return f'<span class="badge {badge_class}">{label}</span>'


def render_tags(items: list, tag_class: str = "tag") -> str:
    """Render a list of items as tags."""
    if not items:
        return ""
    return " ".join([f'<span class="{tag_class}">{item}</span>' for item in items])
