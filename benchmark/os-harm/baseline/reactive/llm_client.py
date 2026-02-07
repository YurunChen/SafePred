"""
Unified LLM Client Module for Reactive Safety System.

Provides a unified interface for calling different LLM providers
(OpenAI, Qwen, custom) using either official SDKs or requests library.
Based on SafePred_v9/models/llm_client.py
"""

from typing import Optional, Dict, List
import requests
import time
import logging

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Exception raised when LLM API connection fails after maximum retries."""
    
    def __init__(self, message: str, max_retries: int, last_error: Exception):
        """
        Initialize connection error.
        
        Args:
            message: Error message
            max_retries: Maximum number of retry attempts
            last_error: The last exception that occurred
        """
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(message)


class LLMClient:
    """
    Unified LLM client for calling different providers.
    
    Supports:
    - OpenAI: Uses OpenAI SDK or requests fallback
    - Qwen: Uses Qwen SDK or requests fallback
    - Custom: Uses requests library
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: str = "gpt-4",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: LLM API key
            api_url: LLM API URL
            model_name: Model name/identifier
            provider: Provider type ('openai', 'qwen', 'custom')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            retry_backoff: Multiplier for exponential backoff (default: 2.0)
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self._last_token_usage = None
    
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using LLM with automatic retry on transient errors.
        
        Args:
            prompt: Text prompt (used if messages is None). If both prompt and messages are None, raises ValueError.
            messages: List of message dicts with 'role' and 'content' keys (used if provided, overrides prompt)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        
        Raises:
            ValueError: If both prompt and messages are None
            LLMConnectionError: If all retry attempts fail
        """
        # Validate that at least one input is provided
        if prompt is None and messages is None:
            raise ValueError("LLMClient.generate() requires either 'prompt' or 'messages' argument")
        
        # If prompt is None but messages is provided, set prompt to empty string
        if prompt is None:
            prompt = ""
        
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        
        # Execute with retry mechanism based on provider
        if self.provider == "openai":
            return self._execute_with_retry(
                lambda: self._generate_openai(prompt, messages, temp, max_toks, **kwargs)
            )
        elif self.provider == "qwen":
            return self._execute_with_retry(
                lambda: self._generate_qwen(prompt, messages, temp, max_toks, **kwargs)
            )
        else:  # custom
            return self._execute_with_retry(
                lambda: self._generate_custom(prompt, messages, temp, max_toks, **kwargs)
            )
    
    def _generate_openai(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI SDK or requests fallback."""
        # Prepare messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Check if model is qwen (dashscope) - use requests directly instead of SDK
        model_lower = self.model_name.lower()
        api_url_lower = (self.api_url or "").lower()
        is_qwen_model = "qwen" in model_lower or "dashscope" in api_url_lower or "compatible-mode" in api_url_lower
        
        if is_qwen_model:
            # Use requests for Qwen models (even with provider="openai")
            return self._generate_custom(prompt, messages, temperature, max_tokens, **kwargs)
        
        # Try OpenAI SDK first
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=self.timeout
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            content = response.choices[0].message.content
            # Store token usage if available
            if hasattr(response, "usage") and response.usage:
                self._last_token_usage = {
                    "model": self.model_name,
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                }
            else:
                self._last_token_usage = None
            return content
        except ImportError:
            # Fallback to requests if OpenAI SDK not available
            logger.warning("OpenAI SDK not available, using requests fallback")
            return self._generate_custom(prompt, messages, temperature, max_tokens, **kwargs)
    
    def _generate_qwen(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Qwen SDK or requests fallback."""
        # Try Qwen SDK first
        try:
            from dashscope import Generation
            
            # Prepare messages for Qwen
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            
            # Convert messages format for Qwen
            qwen_messages = []
            for msg in messages:
                qwen_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            response = Generation.call(
                model=self.model_name,
                messages=qwen_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                **kwargs
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # Store token usage if available (Qwen may have usage in response)
                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "model": self.model_name,
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    }
                else:
                    self._last_token_usage = None
                return content
            else:
                raise Exception(f"Qwen API error: {response.message}")
        except ImportError:
            # Fallback to requests if Qwen SDK not available
            logger.warning("Qwen SDK not available, using requests fallback")
            return self._generate_custom(prompt, messages, temperature, max_tokens, **kwargs)
    
    def _generate_custom(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using requests library (works for any OpenAI-compatible API)."""
        # Prepare messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Determine endpoint
        if self.api_url:
            if self.api_url.endswith("/v1"):
                endpoint = f"{self.api_url}/chat/completions"
            elif self.api_url.endswith("/"):
                endpoint = f"{self.api_url}v1/chat/completions"
            else:
                endpoint = f"{self.api_url}/v1/chat/completions"
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
        
        logger.debug(f"[LLMClient] Using requests to call endpoint: {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            logger.debug(f"[LLMClient] API response status: {response.status_code}")
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = response.text[:500] if response.text else "No error details"
                logger.error(f"[LLMClient] API returned non-200 status: {response.status_code}, error: {error_detail}")
                response.raise_for_status()
            
            result = response.json()
            
            # Check for API-level errors in response
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown API error")
                logger.error(f"[LLMClient] API returned error: {error_msg}")
                raise Exception(f"API error: {error_msg}")
            
            content = result["choices"][0]["message"]["content"]
            # Store token usage if available
            if "usage" in result:
                usage = result["usage"]
                self._last_token_usage = {
                    "model": self.model_name,
                    "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                    "completion_tokens": usage.get("completion_tokens", 0) or 0,
                    "total_tokens": usage.get("total_tokens", 0) or 0,
                }
            else:
                self._last_token_usage = None
            return content
        except requests.exceptions.Timeout as e:
            logger.error(f"[LLMClient] Request timeout after {self.timeout}s: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[LLMClient] Connection error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"[LLMClient] Request exception: {e}")
            raise
    
    def _execute_with_retry(self, generate_func):
        """Execute generation function with retry mechanism."""
        last_error = None
        
        # Log API configuration (without exposing full API key)
        api_key_preview = f"{self.api_key[:8]}..." if self.api_key and len(self.api_key) > 8 else "not set"
        logger.info(f"[LLMClient] Attempting API call: provider={self.provider}, model={self.model_name}, api_url={self.api_url}, api_key={api_key_preview}")
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"[LLMClient] API call attempt {attempt + 1}/{self.max_retries}")
                result = generate_func()
                logger.debug(f"[LLMClient] API call succeeded on attempt {attempt + 1}")
                return result
            except (requests.exceptions.RequestException, 
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(
                        f"[LLMClient] LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"[LLMClient] LLM API call failed after {self.max_retries} attempts: {type(e).__name__}: {str(e)}")
            except Exception as e:
                # Non-retryable errors (e.g., authentication, invalid request)
                logger.error(f"[LLMClient] Non-retryable LLM API error: {type(e).__name__}: {str(e)}", exc_info=True)
                raise
        
        # All retries exhausted - provide detailed error information
        error_details = f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}"
        if last_error:
            error_details += f", Last Error: {type(last_error).__name__}: {str(last_error)}"
        logger.error(f"[LLMClient] All retry attempts exhausted. {error_details}")
        
        raise LLMConnectionError(
            f"Failed to connect to LLM API after {self.max_retries} retries. {error_details}",
            self.max_retries,
            last_error
        )
