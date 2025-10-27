"""
LLM client abstraction supporting Anthropic and Ollama.
"""
import json
import re
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import requests
from logger import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the LLM service."""
        pass


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        logger.info(f"Initialized Anthropic client with model: {model}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 4096) -> Optional[str]:
        """Generate a response using Claude."""
        try:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }

            if system_prompt:
                payload["system"] = system_prompt

            logger.debug(f"Sending request to Anthropic API with model {self.model}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                content = data['content'][0]['text']
                logger.debug(f"Received response from Anthropic API ({len(content)} chars)")
                return content
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Anthropic API request timed out")
            return None
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
            return None

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Anthropic API."""
        try:
            response = self.generate("Hello", max_tokens=10)
            if response:
                logger.info("Anthropic API connection test successful")
                return {"success": True}
            return {"success": False, "error": "No response received"}
        except Exception as e:
            logger.error(f"Anthropic API connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class OllamaClient(LLMClient):
    """Client for local Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        logger.info(f"Initialized Ollama client with model: {model} at {base_url}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate a response using Ollama."""
        try:
            # Combine system and user prompts if system prompt provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }

            logger.debug(f"Sending request to Ollama API with model {self.model}")
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Longer timeout for local models
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('response', '')
                logger.debug(f"Received response from Ollama API ({len(content)} chars)")
                return content
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
            return None
        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
            return None

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama API."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model in model_names or any(self.model in name for name in model_names):
                    logger.info(f"Ollama API connection test successful. Model '{self.model}' available.")
                    return {"success": True}
                else:
                    logger.warning(f"Ollama is running but model '{self.model}' not found. Available: {model_names}")
                    return {"success": False, "error": f"Model '{self.model}' not found. Available: {model_names}"}
            return {"success": False, "error": f"Unexpected status code: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            return {"success": False, "error": f"Cannot connect to Ollama at {self.base_url}"}
        except Exception as e:
            logger.error(f"Ollama API connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(provider: str, **kwargs) -> LLMClient:
        """
        Create an LLM client based on provider.

        Args:
            provider: 'anthropic' or 'ollama'
            **kwargs: Provider-specific configuration
                For Anthropic: api_key (required), model (optional)
                For Ollama: base_url (optional), model (optional)
        """
        provider = provider.lower()

        if provider == "anthropic":
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("api_key is required for Anthropic provider")
            model = kwargs.get('model', 'claude-3-5-sonnet-20241022')
            return AnthropicClient(api_key=api_key, model=model)

        elif provider == "ollama":
            base_url = kwargs.get('base_url', 'http://localhost:11434')
            model = kwargs.get('model', 'llama3.1')
            return OllamaClient(base_url=base_url, model=model)

        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'ollama'")


def parse_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from LLM response.
    Handles various formats including markdown code blocks.
    """
    try:
        # Try direct parsing first
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if '```json' in response:
        try:
            json_str = response.split('```json')[1].split('```')[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass

    # Try finding JSON object in response
    start_idx = response.find('{')
    end_idx = response.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx + 1]
            # Clean up the string
            json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
            # Remove trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.debug(f"Problematic JSON: {json_str[:500]}")

    logger.error("Could not extract valid JSON from LLM response")
    return None
