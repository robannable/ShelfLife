"""
Tests for LLM client abstraction.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_client import (
    LLMClient,
    AnthropicClient,
    OllamaClient,
    LLMClientFactory,
    parse_json_from_response
)


class TestLLMClientFactory:
    """Test LLM client factory."""

    def test_create_anthropic_client(self):
        """Test creating Anthropic client."""
        client = LLMClientFactory.create_client(
            provider="anthropic",
            api_key="test-key",
            model="claude-3-5-sonnet-20241022"
        )

        assert isinstance(client, AnthropicClient)
        assert client.api_key == "test-key"
        assert client.model == "claude-3-5-sonnet-20241022"

    def test_create_anthropic_without_key(self):
        """Test creating Anthropic client without API key raises error."""
        with pytest.raises(ValueError, match="api_key is required"):
            LLMClientFactory.create_client(provider="anthropic")

    def test_create_ollama_client(self):
        """Test creating Ollama client."""
        client = LLMClientFactory.create_client(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.1"
        )

        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3.1"

    def test_create_ollama_with_defaults(self):
        """Test creating Ollama client with default values."""
        client = LLMClientFactory.create_client(provider="ollama")

        assert isinstance(client, OllamaClient)
        assert "localhost" in client.base_url

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClientFactory.create_client(provider="invalid")

    def test_case_insensitive_provider(self):
        """Test provider name is case insensitive."""
        client = LLMClientFactory.create_client(
            provider="ANTHROPIC",
            api_key="test-key"
        )
        assert isinstance(client, AnthropicClient)


class TestAnthropicClient:
    """Test Anthropic client."""

    def test_initialization(self):
        """Test Anthropic client initialization."""
        client = AnthropicClient(api_key="test-key", model="claude-3")
        assert client.api_key == "test-key"
        assert client.model == "claude-3"
        assert "anthropic.com" in client.api_url

    @patch('llm_client.requests.post')
    def test_generate_success(self, mock_post):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Generated response'}]
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")
        response = client.generate("Test prompt")

        assert response == "Generated response"
        mock_post.assert_called_once()

    @patch('llm_client.requests.post')
    def test_generate_with_system_prompt(self, mock_post):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Response'}]
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")
        response = client.generate(
            "User prompt",
            system_prompt="System instructions"
        )

        # Verify system prompt was included in request
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert 'system' in payload
        assert payload['system'] == "System instructions"

    @patch('llm_client.requests.post')
    def test_generate_api_error(self, mock_post):
        """Test handling API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")
        response = client.generate("Test prompt")

        assert response is None

    @patch('llm_client.requests.post')
    def test_generate_timeout(self, mock_post):
        """Test handling timeout."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        client = AnthropicClient(api_key="test-key")
        response = client.generate("Test prompt")

        assert response is None

    @patch('llm_client.requests.post')
    def test_custom_max_tokens(self, mock_post):
        """Test custom max_tokens parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Response'}]
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")
        client.generate("Test", max_tokens=1000)

        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['max_tokens'] == 1000


class TestOllamaClient:
    """Test Ollama client."""

    def test_initialization(self):
        """Test Ollama client initialization."""
        client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama3.1"
        )
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3.1"

    def test_initialization_strips_trailing_slash(self):
        """Test base URL trailing slash is removed."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    @patch('llm_client.requests.post')
    def test_generate_success(self, mock_post):
        """Test successful generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Generated text'
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.generate("Test prompt")

        assert response == "Generated text"

    @patch('llm_client.requests.post')
    def test_generate_with_system_prompt(self, mock_post):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Response'
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.generate("User prompt", system_prompt="System instructions")

        # Verify prompts were combined
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert "System instructions" in payload['prompt']
        assert "User prompt" in payload['prompt']

    @patch('llm_client.requests.post')
    def test_generate_connection_error(self, mock_post):
        """Test handling connection error."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()

        client = OllamaClient()
        response = client.generate("Test")

        assert response is None

    @patch('llm_client.requests.get')
    def test_connection_test_success(self, mock_get):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'llama3.1'}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.1")
        result = client.test_connection()

        assert result['success'] is True

    @patch('llm_client.requests.get')
    def test_connection_test_model_not_found(self, mock_get):
        """Test connection test when model not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'other-model'}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.1")
        result = client.test_connection()

        assert result['success'] is False
        assert "not found" in result['error']


class TestParseJsonFromResponse:
    """Test JSON parsing from LLM responses."""

    def test_parse_direct_json(self):
        """Test parsing direct JSON response."""
        response = '{"key": "value", "number": 123}'
        parsed = parse_json_from_response(response)

        assert parsed is not None
        assert parsed['key'] == 'value'
        assert parsed['number'] == 123

    def test_parse_json_with_markdown(self):
        """Test parsing JSON in markdown code block."""
        response = '''Here is the JSON:
```json
{
  "key": "value",
  "list": [1, 2, 3]
}
```
Hope this helps!'''

        parsed = parse_json_from_response(response)

        assert parsed is not None
        assert parsed['key'] == 'value'
        assert parsed['list'] == [1, 2, 3]

    def test_parse_json_embedded_in_text(self):
        """Test parsing JSON embedded in text."""
        response = '''
The analysis shows that {"result": "positive", "confidence": 0.95} based on the data.
'''

        parsed = parse_json_from_response(response)

        assert parsed is not None
        assert parsed['result'] == 'positive'
        assert parsed['confidence'] == 0.95

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing commas."""
        response = '''
{
  "key": "value",
  "list": [1, 2, 3,]
}
'''

        parsed = parse_json_from_response(response)

        # Parser should handle trailing commas
        assert parsed is not None
        assert parsed['key'] == 'value'

    def test_parse_complex_nested_json(self):
        """Test parsing complex nested JSON."""
        response = '''
{
  "themes": [
    {"name": "Theme 1", "description": "Desc 1"},
    {"name": "Theme 2", "description": "Desc 2"}
  ],
  "metadata": {
    "count": 2,
    "generated_at": "2024-01-01"
  }
}
'''

        parsed = parse_json_from_response(response)

        assert parsed is not None
        assert len(parsed['themes']) == 2
        assert parsed['metadata']['count'] == 2

    def test_parse_invalid_json(self):
        """Test handling invalid JSON."""
        response = "This is not JSON at all"
        parsed = parse_json_from_response(response)

        assert parsed is None

    def test_parse_empty_response(self):
        """Test handling empty response."""
        parsed = parse_json_from_response("")
        assert parsed is None

    def test_parse_json_with_newlines(self):
        """Test parsing JSON with newlines in strings."""
        response = '{"text": "Line 1\\nLine 2\\nLine 3"}'
        parsed = parse_json_from_response(response)

        assert parsed is not None
        assert "Line 1" in parsed['text']
