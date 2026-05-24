"""End-to-end tests for view pages using streamlit.testing.v1.AppTest.

These tests render each view in an isolated environment (temp database,
mocked LLM responses) and assert on the rendered Streamlit elements.
They are intentionally coarse-grained: they verify that pages render
without exception and that key user-visible behaviors (empty states,
form validation, mock invocation) work as expected. Fine-grained logic
is covered by the existing unit tests.
"""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from unittest.mock import MagicMock, patch


VIEWS = [
    "shelflife_app/views/add_book.py",
    "shelflife_app/views/view_collection.py",
    "shelflife_app/views/analytics.py",
    "shelflife_app/views/network_view.py",
    "shelflife_app/views/executive_summary.py",
    "shelflife_app/views/ask_library.py",
]


@pytest.fixture(autouse=True)
def isolated_app(tmp_path, monkeypatch):
    """Give each test its own DB and a clean st.cache_resource."""
    import config
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def mock_book_service():
    """Replace BookService construction so views don't hit the real LLM."""
    with patch("shelflife_app.services.BookService") as cls:
        instance = MagicMock()
        instance.test_connection.return_value = {"success": True}
        instance.enhance_book_data.return_value = {
            "synopsis": "Mock synopsis",
            "genre": ["Fiction"],
            "themes": ["mock"],
            "sources": ["mock"],
        }
        instance.ask_library_question.return_value = "Mock answer."
        instance.ask_library_question_stream.return_value = iter(["Mock ", "answer."])
        instance.generate_executive_summary.return_value = {
            "summary": "Mock summary",
            "patterns": ["Pattern A"],
            "recommendations": ["Read more"],
        }
        instance.analyze_themes.return_value = {"uber_themes": []}
        cls.return_value = instance
        yield instance


def _run(path, timeout=15):
    at = AppTest.from_file(path, default_timeout=timeout)
    at.run()
    return at


# ---------------------------------------------------------------------------
# Smoke tests: each view renders without exception on an empty database
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("view_path", VIEWS)
def test_view_renders_without_exception(view_path, mock_book_service):
    at = _run(view_path)
    assert not at.exception, f"{view_path} raised: {at.exception}"


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

def test_add_book_validation_rejects_empty_form(mock_book_service):
    """Submitting the form with no title/author shows validation errors
    and does not invoke the LLM enrichment path."""
    at = _run("shelflife_app/views/add_book.py")

    submit = next(b for b in at.button if "Add Book" in b.label)
    submit.click().run()

    error_text = " ".join(e.value for e in at.error)
    assert "fix the following errors" in error_text.lower() or "title" in error_text.lower(), (
        f"Expected a validation error to be displayed, got: {error_text!r}"
    )
    mock_book_service.enhance_book_data.assert_not_called()


def test_view_collection_empty_state_shows_add_button(mock_book_service):
    """With no books, the empty state offers a CTA to add the first book."""
    at = _run("shelflife_app/views/view_collection.py")
    assert any("Add Your First Book" in b.label for b in at.button), (
        f"Expected 'Add Your First Book' button. Buttons: {[b.label for b in at.button]}"
    )


def test_analytics_shows_empty_state_when_no_books(mock_book_service):
    at = _run("shelflife_app/views/analytics.py")
    rendered = " ".join(m.value for m in at.markdown)
    assert "No Data to Analyze" in rendered, "Expected analytics empty state"


def test_network_view_shows_empty_state_when_under_two_books(mock_book_service):
    at = _run("shelflife_app/views/network_view.py")
    rendered = " ".join(m.value for m in at.markdown)
    assert "Not Enough Books" in rendered, "Expected network view empty state"


def test_executive_summary_shows_empty_state_when_no_books(mock_book_service):
    at = _run("shelflife_app/views/executive_summary.py")
    rendered = " ".join(m.value for m in at.markdown)
    assert "No Books Yet" in rendered, "Expected executive summary empty state"


def test_ask_library_warns_on_empty_query(mock_book_service):
    """Clicking 'Ask the Library' with no query shows a warning rather than
    invoking the streaming generator."""
    at = _run("shelflife_app/views/ask_library.py")

    ask = next(b for b in at.button if "Ask the Library" in b.label)
    ask.click().run()

    warning_text = " ".join(w.value for w in at.warning)
    assert "Please enter a question" in warning_text, (
        f"Expected empty-query warning, got: {warning_text!r}"
    )
    mock_book_service.ask_library_question_stream.assert_not_called()
