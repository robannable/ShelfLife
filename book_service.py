"""
Book service for handling LLM-enhanced metadata operations.
Includes caching for LLM responses and parallel processing for performance.
"""
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from api_utils import fetch_book_metadata
from llm_client import LLMClientFactory, parse_json_from_response
from models import BookMetadata
from constants import STANDARD_GENRES, GENRE_PROMPT
from logger import get_logger
from cache_utils import get_memory_cache, generate_cache_key

logger = get_logger(__name__)

# In-memory cache for LLM responses (1 hour TTL)
llm_cache = get_memory_cache()


class BookService:
    """Service for book metadata enhancement using LLM."""

    def __init__(self):
        """Initialize the book service with configured LLM client."""
        try:
            # Initialize LLM client based on config
            provider = config.LLM_PROVIDER.lower()

            if provider == "anthropic":
                self.llm_client = LLMClientFactory.create_client(
                    provider="anthropic",
                    api_key=config.ANTHROPIC_API_KEY,
                    model=config.ANTHROPIC_MODEL
                )
                logger.info(f"BookService initialized with Anthropic ({config.ANTHROPIC_MODEL})")

            elif provider == "ollama":
                self.llm_client = LLMClientFactory.create_client(
                    provider="ollama",
                    base_url=config.OLLAMA_BASE_URL,
                    model=config.OLLAMA_MODEL
                )
                logger.info(f"BookService initialized with Ollama ({config.OLLAMA_MODEL})")

            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to initialize BookService: {str(e)}", exc_info=True)
            raise

    def enhance_book_data(
        self,
        title: str,
        author: str,
        year: Optional[int] = None,
        isbn: Optional[str] = None
    ) -> Optional[BookMetadata]:
        """
        Enhance book data with metadata from APIs and LLM analysis.
        Uses caching to avoid redundant LLM calls.

        Args:
            title: Book title
            author: Book author
            year: Publication year (optional)
            isbn: ISBN (optional)

        Returns:
            BookMetadata object with enhanced information, or None if enhancement fails
        """
        try:
            logger.info(f"Enhancing data for: {title} by {author}")

            # Generate cache key for LLM response
            cache_key = f"llm_enhance:{generate_cache_key(title, author, year or 0, isbn or '')}"

            # Step 1: Fetch basic metadata from Open Library and Google Books
            api_metadata = fetch_book_metadata(title, author, isbn)

            # Step 2: Use year from API if not provided
            if not year and api_metadata.get('year'):
                try:
                    year = int(api_metadata['year'])
                except (ValueError, TypeError):
                    year = None

            # Step 3: Check cache for LLM analysis
            cached_llm_data = llm_cache.get(cache_key)

            if cached_llm_data:
                logger.debug(f"Using cached LLM analysis for: {title} by {author}")
                # Merge cached LLM data with fresh API metadata
                merged_data = {**api_metadata, **cached_llm_data}
                sources = api_metadata.get('sources', [])
                sources.append(f"LLM ({config.LLM_PROVIDER}) - Cached")
                merged_data['sources'] = sources
                return BookMetadata.from_dict(merged_data)

            # Step 4: Build enhanced prompt with genre constraints
            book_prompt = config.BOOK_ANALYSIS_PROMPT.format(
                title=title,
                author=author,
                year=year or api_metadata.get("year", "unknown")
            )

            genre_selection_prompt = GENRE_PROMPT.format(
                genres="\n".join(f"- {genre}" for genre in STANDARD_GENRES)
            )

            full_prompt = f"{book_prompt}\n\nFor genre classification:\n{genre_selection_prompt}"

            # Step 5: Get LLM analysis
            logger.debug("Requesting LLM analysis...")
            llm_response = self.llm_client.generate(full_prompt)

            if not llm_response:
                logger.warning("No response from LLM, returning API metadata only")
                return BookMetadata.from_dict(api_metadata)

            # Step 6: Parse LLM response
            llm_data = parse_json_from_response(llm_response)

            if not llm_data:
                logger.warning("Failed to parse LLM response, returning API metadata only")
                return BookMetadata.from_dict(api_metadata)

            # Step 7: Cache LLM analysis
            llm_cache.set(cache_key, llm_data, ttl=3600)  # Cache for 1 hour

            # Step 8: Merge all data sources
            merged_data = {**api_metadata, **llm_data}

            # Add sources
            sources = api_metadata.get('sources', [])
            sources.append(f"LLM ({config.LLM_PROVIDER})")
            merged_data['sources'] = sources

            logger.info(f"Successfully enhanced metadata for: {title}")
            return BookMetadata.from_dict(merged_data)

        except Exception as e:
            logger.error(f"Error enhancing book data for '{title}': {str(e)}", exc_info=True)
            # Return basic metadata if available
            if api_metadata:
                return BookMetadata.from_dict(api_metadata)
            return None

    def generate_executive_summary(self, library_catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an executive summary of the library collection.

        Args:
            library_catalog: Dictionary with 'library' list of books

        Returns:
            Dictionary with summary, patterns, and recommendations
        """
        try:
            logger.info("Generating executive summary...")

            prompt = config.EXECUTIVE_SUMMARY_PROMPT.format(
                library_catalog=json.dumps(library_catalog['library'], indent=2)
            )

            response = self.llm_client.generate(prompt)

            if not response:
                logger.error("No response from LLM for executive summary")
                return None

            summary_data = parse_json_from_response(response)

            if summary_data:
                logger.info("Executive summary generated successfully")
                return summary_data
            else:
                logger.error("Failed to parse executive summary response")
                return None

        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}", exc_info=True)
            return None

    def analyze_themes(self, themes: List[str], max_themes_per_chunk: int = 20) -> Optional[Dict[str, Any]]:
        """
        Analyze and group themes into uber-themes.

        Args:
            themes: List of theme strings
            max_themes_per_chunk: Maximum themes to process in one LLM call

        Returns:
            Dictionary with uber_themes and analysis
        """
        try:
            logger.info(f"Analyzing {len(themes)} themes...")

            if len(themes) > max_themes_per_chunk:
                # Process in chunks
                return self._analyze_themes_chunked(themes, max_themes_per_chunk)
            else:
                # Process all at once
                return self._analyze_theme_chunk(themes)

        except Exception as e:
            logger.error(f"Error analyzing themes: {str(e)}", exc_info=True)
            return None

    def _analyze_theme_chunk(self, themes: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze a single chunk of themes."""
        try:
            themes_list = "\n".join([f"- {theme}" for theme in themes])

            prompt = config.THEME_ANALYSIS_PROMPT.format(themes_list=themes_list)

            response = self.llm_client.generate(prompt)

            if not response:
                logger.warning("No response from LLM for theme analysis")
                return None

            analysis = parse_json_from_response(response)

            if analysis:
                logger.debug(f"Successfully analyzed {len(themes)} themes")
                return analysis
            else:
                logger.warning("Failed to parse theme analysis response")
                return None

        except Exception as e:
            logger.error(f"Error analyzing theme chunk: {str(e)}", exc_info=True)
            return None

    def _analyze_themes_chunked(self, themes: List[str], chunk_size: int, max_workers: int = 3) -> Dict[str, Any]:
        """
        Analyze themes in multiple chunks using parallel processing for better performance.

        Args:
            themes: List of themes to analyze
            chunk_size: Size of each chunk
            max_workers: Maximum number of parallel workers (default: 3 to avoid API rate limits)

        Returns:
            Combined analysis from all chunks
        """
        combined_analysis = {
            "uber_themes": [],
            "analysis": {
                "summary": "",
                "key_insights": []
            }
        }

        chunk_summaries = []
        chunk_insights = []

        # Create theme chunks
        theme_chunks = [themes[i:i + chunk_size] for i in range(0, len(themes), chunk_size)]
        logger.info(f"Processing {len(theme_chunks)} theme chunks in parallel (max {max_workers} workers)")

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk analysis tasks
            future_to_chunk = {
                executor.submit(self._analyze_theme_chunk, chunk): idx
                for idx, chunk in enumerate(theme_chunks)
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_analysis = future.result()

                    if chunk_analysis:
                        logger.debug(f"Completed chunk {chunk_idx + 1}/{len(theme_chunks)}")

                        # Collect uber-themes
                        combined_analysis["uber_themes"].extend(chunk_analysis.get("uber_themes", []))

                        # Collect summaries and insights
                        if "analysis" in chunk_analysis:
                            if "summary" in chunk_analysis["analysis"]:
                                chunk_summaries.append(chunk_analysis["analysis"]["summary"])
                            if "key_insights" in chunk_analysis["analysis"]:
                                chunk_insights.extend(chunk_analysis["analysis"]["key_insights"])
                    else:
                        logger.warning(f"Chunk {chunk_idx + 1} returned no analysis")

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}", exc_info=True)
                    # Continue with other chunks

        logger.info(f"Completed parallel processing of {len(theme_chunks)} chunks")

        # Combine summaries
        if chunk_summaries:
            combine_prompt = f"""Synthesize these theme analysis summaries into a single concise 200-word summary:

{' '.join(chunk_summaries)}

Provide a concise synthesis without citations or references. Focus on the relationships and patterns between themes."""

            combined_summary = self.llm_client.generate(combine_prompt)
            if combined_summary:
                combined_analysis["analysis"]["summary"] = combined_summary
            else:
                combined_analysis["analysis"]["summary"] = " ".join(chunk_summaries)

        # Add unique insights
        if chunk_insights:
            seen = set()
            unique_insights = []
            for insight in chunk_insights:
                if insight not in seen:
                    seen.add(insight)
                    unique_insights.append(insight)
            combined_analysis["analysis"]["key_insights"] = unique_insights[:5]

        logger.info(f"Combined analysis from {len(chunk_summaries)} chunks")
        return combined_analysis

    def ask_library_question(self, query: str, library_catalog: Dict[str, Any]) -> Optional[str]:
        """
        Ask a question about the library collection.

        Args:
            query: User's question
            library_catalog: Dictionary with library data

        Returns:
            LLM's response as string
        """
        try:
            logger.info(f"Processing library query: {query[:50]}...")

            system_prompt = """You are a knowledgeable librarian assistant. Use the provided library catalog to answer queries.
If the query cannot be answered using only the library information, clearly state that."""

            prompt = f"""Library Catalog:
{json.dumps(library_catalog['library'], indent=2)}

Query: {query}

Please provide a clear, concise response based on the available library data."""

            response = self.llm_client.generate(prompt, system_prompt=system_prompt)

            if response:
                logger.info("Library query answered successfully")
                return response
            else:
                logger.error("No response from LLM for library query")
                return None

        except Exception as e:
            logger.error(f"Error processing library query: {str(e)}", exc_info=True)
            return None

    def test_connection(self) -> Dict[str, Any]:
        """Test the LLM connection."""
        try:
            return self.llm_client.test_connection()
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


# Helper functions for backward compatibility with existing code

def extract_and_save_themes(books_with_metadata: List[tuple]) -> Dict[str, Any]:
    """Extract all unique themes from books and save to JSON."""
    try:
        unique_themes = set()

        for book in books_with_metadata:
            metadata_json = book[8] if len(book) > 8 else None
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    themes = metadata.get('themes', [])
                    unique_themes.update(themes)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in book metadata")
                    continue

        theme_data = {
            "themes": sorted(list(unique_themes)),
            "generated_at": datetime.now().isoformat(),
            "total_themes": len(unique_themes)
        }

        # Save to data folder
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "theme_inventory.json", "w") as f:
            json.dump(theme_data, f, indent=2)

        logger.info(f"Extracted {len(unique_themes)} unique themes")
        return theme_data

    except Exception as e:
        logger.error(f"Error extracting themes: {str(e)}", exc_info=True)
        return {"themes": [], "total_themes": 0}
