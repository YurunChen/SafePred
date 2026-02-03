"""
Unified LLM Client Module.

Supports three providers only:
- openai: OpenAI SDK (openai library)
- gemini: Google Gemini SDK (google-generativeai library)
- custom: HTTP via requests (OpenAI-compatible or other APIs)
"""

from typing import Optional, Dict, Any, List, Type
import requests
import json
import time
from .logger import get_logger

logger = get_logger("SafePred.LLMClient")

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_CUSTOM = "custom"


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
    
    - openai: Uses openai library (OpenAI API)
    - gemini: Uses google-generativeai library (Gemini API)
    - custom: Uses requests library (OpenAI-compatible or other HTTP APIs)
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
            provider: One of 'openai', 'gemini', 'custom'
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
        self.provider = (provider or "openai").strip().lower()
        if self.provider not in (PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_CUSTOM):
            self.provider = PROVIDER_CUSTOM
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self._last_token_usage = None  # Store token usage from last API call
    
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
            LLMConnectionError: If all retry attempts fail and connection cannot be established.
                This exception will terminate the program execution.
            Exception: For non-retryable errors (raised immediately without retry)
        """
        # Validate that at least one input is provided
        if prompt is None and messages is None:
            raise ValueError("LLMClient.generate() requires either 'prompt' or 'messages' argument")
        
        # If prompt is None but messages is provided, set prompt to empty string
        # This maintains backward compatibility with internal implementations
        if prompt is None:
            prompt = ""
        
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        
        # Dispatch by provider: openai (SDK), gemini (SDK), custom (requests)
        if self.provider == PROVIDER_OPENAI:
            generate_func = lambda: self._generate_openai(prompt, messages, temp, max_toks, **kwargs)
        elif self.provider == PROVIDER_GEMINI:
            generate_func = lambda: self._generate_gemini(prompt, messages, temp, max_toks, **kwargs)
        else:
            generate_func = lambda: self._generate_custom(prompt, messages, temp, max_toks, **kwargs)
        
        # Execute with retry mechanism
        return self._execute_with_retry(generate_func)
    
    def _generate_openai(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI SDK (openai library)."""
        from openai import OpenAI

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        client_kwargs = {"api_key": self.api_key}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url.rstrip("/")
        client = OpenAI(**client_kwargs)

        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        model_lower = self.model_name.lower()
        if "gpt-5" in model_lower or "o3" in model_lower or "o4" in model_lower:
            create_kwargs.pop("temperature", None)
            create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens", max_tokens)

        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned None content")
        if response.usage:
            self._last_token_usage = {
                "model": self.model_name,
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        return str(content)

    def _generate_gemini(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Gemini SDK (google-generativeai library)."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        contents = prompt
        if messages:
            contents = self._messages_to_prompt(messages)

        try:
            gen_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except Exception:
            gen_config = None  # use SDK defaults
        response = model.generate_content(contents, generation_config=gen_config)
        text = getattr(response, "text", None) if response else None
        if not text:
            raise ValueError("Gemini returned empty or None content")
        return text

    def _generate_requests(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using requests for OpenAI-compatible API (used by provider=custom)."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Check if model requires special parameters
        model_lower = self.model_name.lower()
        if "gpt-5" in model_lower or "o3" in model_lower or "o4" in model_lower:
            # Use max_completion_tokens for newer models
            # Don't pass temperature - these models only support default temperature=1
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                **kwargs
            }
            # Remove temperature from kwargs if present (these models don't support it)
            payload.pop("temperature", None)
        elif "qwen3" in model_lower:
            # Qwen3 models require enable_thinking parameter (default to False)
            # If enable_thinking is explicitly set in kwargs, it will override the default
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": False,  # Default for qwen3 models
                **kwargs  # kwargs can override enable_thinking if explicitly set
            }
        else:
            # Use max_tokens for standard models
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
        
        # For requests, we need the full endpoint URL: https://api.openai-proxy.org/v1/chat/completions
        # Input URL is base URL: https://api.openai-proxy.org/v1
        # IMPORTANT: Always use self.api_url from config.yaml, never use a cached or modified URL
        base_api_url = self.api_url or "https://api.openai.com/v1"
        if base_api_url != self.api_url:
            logger.warning(f"[LLM Client] Using default URL instead of config: {base_api_url}")
        
        api_url = base_api_url.rstrip("/")
        # If URL doesn't end with /v1, add it
        if not api_url.endswith("/v1"):
            api_url = api_url + "/v1"
        # Add /chat/completions for requests
        api_url = api_url + "/chat/completions"
        
        # Log the URL being used (for debugging retry issues)
        logger.debug(f"[LLM Client] Using API URL from config: {self.api_url} -> {api_url}")
        
        # Explicitly disable proxy to avoid using system proxy (e.g., 127.0.0.1:7897)
        # which could cause URL to change during retries
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            proxies={'http': None, 'https': None},  # Disable proxy - ensures we use config URL, not proxy
        )
        
        # Check if error is due to max_tokens not being supported
        if response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', '') or str(error_data)
                # Log detailed error information
                logger.error("=" * 80)
                logger.error(f"[LLM Client] 400 Bad Request - Model: {self.model_name}")
                logger.error(f"API URL: {api_url}")
                logger.error(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                logger.error(f"Error response: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                logger.error(f"Error message: {error_msg}")
                logger.error("=" * 80)
                
                if 'max_tokens' in error_msg.lower() and ('max_completion_tokens' in error_msg.lower() or 'unsupported parameter' in error_msg.lower()):
                    # Retry with max_completion_tokens instead
                    logger.warning(f"[LLM Client] Model {self.model_name} doesn't support max_tokens, retrying with max_completion_tokens")
                    payload.pop('max_tokens', None)
                    payload['max_completion_tokens'] = max_tokens
                    # Remove temperature if present (some models don't support it)
                    payload.pop('temperature', None)
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                        proxies={'http': None, 'https': None},  # Disable proxy
                    )
                    # If retry also fails, log the error again
                    if response.status_code == 400:
                        retry_error_msg = error_msg  # Default to original error message
                        try:
                            retry_error_data = response.json()
                            retry_error_msg = retry_error_data.get('error', {}).get('message', '') or str(retry_error_data)
                            logger.error("=" * 80)
                            logger.error(f"[LLM Client] Retry also failed with 400 Bad Request")
                            logger.error(f"Retry payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                            logger.error(f"Retry error response: {json.dumps(retry_error_data, indent=2, ensure_ascii=False)}")
                            logger.error(f"Retry error message: {retry_error_msg}")
                            logger.error("=" * 80)
                        except Exception:
                            logger.error(f"[LLM Client] Retry error response text: {response.text[:500]}")
                        # Raise exception with detailed error information
                        raise requests.exceptions.HTTPError(
                            f"400 Bad Request: {retry_error_msg}. "
                            f"API URL: {api_url}, Model: {self.model_name}. "
                            f"Retry with max_completion_tokens also failed.",
                            response=response
                        )
                else:
                    # Not a max_tokens error, raise immediately with detailed error information
                    raise requests.exceptions.HTTPError(
                        f"400 Bad Request: {error_msg}. "
                        f"API URL: {api_url}, Model: {self.model_name}. "
                        f"This is not a max_tokens issue, cannot auto-retry.",
                        response=response
                    )
            except requests.exceptions.HTTPError:
                # Re-raise HTTPError (already formatted with detailed info)
                raise
            except (ValueError, KeyError) as e:
                # If we can't parse the error, log the raw response and raise with details
                logger.error("=" * 80)
                logger.error(f"[LLM Client] Failed to parse 400 error response")
                logger.error(f"Raw response text: {response.text[:1000]}")
                logger.error(f"Parse error: {e}")
                logger.error("=" * 80)
                raise requests.exceptions.HTTPError(
                    f"400 Bad Request: Failed to parse error response. "
                    f"API URL: {api_url}, Model: {self.model_name}. "
                    f"Raw response: {response.text[:500]}",
                    response=response
                ) from e
        
        response.raise_for_status()
        result = response.json()
        
        # Debug: Log response structure
        logger.debug(f"[LLM Client] API Response (requests) - Model: {self.model_name}")
        logger.debug(f"[LLM Client] Response status: {response.status_code}")
        logger.debug(f"[LLM Client] Response keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("=" * 80)
            logger.error("[LLM Client] No choices in response (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Response JSON: {result}")
            logger.error("=" * 80)
            raise ValueError("LLM response has no choices")
        
        content = result["choices"][0]["message"]["content"]
        if content is None:
            # Log full response for debugging
            logger.error("=" * 80)
            logger.error("[LLM Client] LLM returned None content (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Full response JSON: {result}")
            logger.error("=" * 80)
            raise ValueError("LLM returned None content in response")
        content_str = str(content)
        # Log and store token usage if available
        if "usage" in result:
            usage = result["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            logger.info(
                f"[LLM Client] Token usage - Model: {self.model_name}, "
                f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
            )
            # Store token usage for World Model to retrieve
            self._last_token_usage = {
                'model': self.model_name,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            }
        # Log if content is empty
        if not content_str or len(content_str.strip()) == 0:
            logger.error("=" * 80)
            logger.error("[LLM Client] LLM returned empty content (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Response JSON: {result}")
            logger.error(f"Content (repr): {repr(content)}")
            logger.error("=" * 80)
        return content_str

    def _generate_custom(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using requests for custom provider."""
        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
            "Content-Type": "application/json",
        }
        # Remove None values
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Prepare messages if not provided
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Check if API URL looks like OpenAI-compatible format (contains /v1)
        # If so, use OpenAI-format requests instead
        api_url_lower = (self.api_url or "").lower()
        if "/v1" in api_url_lower or "openai" in api_url_lower or "chat/completions" in api_url_lower:
            # Use OpenAI-compatible format
            return self._generate_requests(prompt, messages, temperature, max_tokens, **kwargs)
        
        # Custom providers may use different formats
        # Try prompt-based format first
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # If messages provided, try messages format
        if messages:
            payload_messages = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
        else:
            payload_messages = payload
        
        # IMPORTANT: Always use self.api_url from config.yaml, never use a cached or modified URL
        # Log the URL being used (for debugging retry issues)
        logger.debug(f"[LLM Client] Using API URL from config: {self.api_url}")
        
        # Try messages format first, then prompt format
        last_error = None
        for idx, payload_to_try in enumerate([payload_messages, payload]):
            format_name = "messages" if idx == 0 else "prompt"
            try:
                # Explicitly disable proxy to avoid using system proxy (e.g., 127.0.0.1:7897)
                # which could cause URL to change during retries
                response = requests.post(
                    self.api_url,  # Always use self.api_url from config.yaml
                    headers=headers,
                    json=payload_to_try,
                    timeout=self.timeout,
                    proxies={'http': None, 'https': None},  # Disable proxy - ensures we use config URL, not proxy
                )
                response.raise_for_status()
                result = response.json()
                
                # Log and store token usage if available
                if "usage" in result:
                    usage = result["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    logger.info(
                        f"[LLM Client] Token usage (Custom) - Model: {self.model_name}, "
                        f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
                    )
                    # Store token usage for World Model to retrieve
                    self._last_token_usage = {
                        'model': self.model_name,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                    }
                
                # Extract text from various response formats
                return self._extract_text_from_response(result)
                
            except Exception as e:
                last_error = e
                logger.debug(f"[LLM Client] Custom API call failed with {format_name} format: {e}")
                # Log response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_result = e.response.json()
                        logger.debug(f"[LLM Client] Error response JSON: {error_result}")
                    except Exception:
                        logger.debug(f"[LLM Client] Error response text: {e.response.text[:500]}")
                continue
        
        # If all attempts failed, raise with diagnostic information
        error_msg = f"Failed to extract text from custom API response"
        if last_error:
            error_msg += f": {last_error}"
        logger.error("=" * 80)
        logger.error(f"[LLM Client] {error_msg}")
        logger.error("=" * 80)
        logger.error(f"API URL: {self.api_url}")
        logger.error(f"Model: {self.model_name}")
        logger.error(f"Provider: {self.provider}")
        logger.error(f"Last error: {last_error}")
        logger.error("=" * 80)
        raise Exception(error_msg)
    
    def _extract_text_from_response(self, result: Dict[str, Any]) -> str:
        """
        Extract text from various API response formats.
        
        Args:
            result: API response dictionary
        
        Returns:
            Extracted text content
        
        Raises:
            ValueError: If text cannot be extracted
        """
        # Try different response formats
        if "text" in result:
            return result["text"]
        elif "content" in result:
            return result["content"]
        elif "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "text" in choice:
                return choice["text"]
        elif "output" in result:
            if isinstance(result["output"], str):
                return result["output"]
            elif "text" in result["output"]:
                return result["output"]["text"]
        
        # If no standard format found, return first string value
        for value in result.values():
            if isinstance(value, str) and value:
                return value
        
        raise ValueError("Failed to extract text from response")
    
    def _execute_with_retry(self, func, max_retries: Optional[int] = None) -> Any:
        """
        Execute a function with retry mechanism for transient errors.
        
        Args:
            func: Function to execute (callable that takes no arguments)
            max_retries: Override default max_retries (optional)
        
        Returns:
            Function result
        
        Raises:
            LLMConnectionError: If all retry attempts fail and connection cannot be established.
                This exception will terminate the program execution.
            Exception: For non-retryable errors (raised immediately without retry).
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        retries = 0
        current_delay = self.retry_delay
        
        # Define retryable exceptions
        retryable_exceptions = self._get_retryable_exceptions()
        
        last_exception = None
        
        while retries < max_retries:
            try:
                return func()
            except requests.exceptions.HTTPError as e:
                # Special handling for HTTP errors - check if retryable
                if self._is_retryable_http_error(e):
                    retries += 1
                    last_exception = e
                    
                    # Check if we've exhausted retries
                    if retries >= max_retries:
                        error_msg = (
                            f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                            f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                            f"Last error: {type(e).__name__}: {e}"
                        )
                        logger.error("=" * 80)
                        logger.error(error_msg)
                        logger.error("=" * 80)
                        raise LLMConnectionError(error_msg, max_retries, e) from e
                    
                    # Log retry attempt with URL from config
                    status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'unknown'
                    logger.warning(
                        f"[LLM Client] Retryable HTTP error (status {status_code}): {e} "
                        f"(attempt {retries}/{max_retries}, waiting {current_delay:.2f}s) "
                        f"Using API URL from config: {self.api_url}"
                    )
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff
                else:
                    # Non-retryable HTTP error (e.g., 4xx except 429), raise immediately
                    logger.error(f"[LLM Client] Non-retryable HTTP error: {e}")
                    raise
            except tuple(retryable_exceptions) as e:
                retries += 1
                last_exception = e
                
                # Check if we've exhausted retries
                if retries >= max_retries:
                    error_type = type(e).__name__
                    error_msg = (
                        f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                        f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                        f"Last error ({error_type}): {e}"
                    )
                    logger.error("=" * 80)
                    logger.error(error_msg)
                    logger.error("=" * 80)
                    raise LLMConnectionError(error_msg, max_retries, e) from e
                
                # Log retry attempt with URL from config
                error_type = type(e).__name__
                logger.warning(
                    f"[LLM Client] Retryable error ({error_type}): {e} "
                    f"(attempt {retries}/{max_retries}, waiting {current_delay:.2f}s) "
                    f"Using API URL from config: {self.api_url}"
                )
                
                # Wait before retrying (exponential backoff)
                time.sleep(current_delay)
                current_delay *= self.retry_backoff
                
            except Exception as e:
                # Non-retryable exception, raise immediately
                logger.error(f"[LLM Client] Non-retryable error: {e}")
                raise
        
        # Should not reach here, but just in case (fallback)
        if last_exception:
            error_msg = (
                f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                f"Last error: {type(last_exception).__name__}: {last_exception}"
            )
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise LLMConnectionError(error_msg, max_retries, last_exception) from last_exception
        else:
            error_msg = (
                f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}."
            )
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise LLMConnectionError(error_msg, max_retries, Exception("Unknown error"))
    
    def _get_retryable_exceptions(self) -> List[Type[Exception]]:
        """
        Get list of retryable exception types based on provider.
        
        Returns:
            List of exception types that should trigger retry
        """
        retryable = []
        
        # Common network/HTTP errors (from requests library)
        retryable.extend([
            requests.exceptions.ConnectionError,  # Connection failed
            requests.exceptions.Timeout,  # Request timeout
            requests.exceptions.HTTPError,  # HTTP error (will check status code)
            requests.exceptions.SSLError,  # SSL/TLS connection error
            requests.exceptions.RequestException,  # Base exception for all requests errors
        ])
        
        # OpenAI SDK exceptions (provider=openai)
        if self.provider == PROVIDER_OPENAI:
            try:
                import openai
                retryable.extend([
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ])
            except ImportError:
                pass
        # Gemini/Google API exceptions (provider=gemini)
        if self.provider == PROVIDER_GEMINI:
            try:
                from google.api_core.exceptions import (
                    ResourceExhausted,
                    ServiceUnavailable,
                    DeadlineExceeded,
                )
                retryable.extend([ResourceExhausted, ServiceUnavailable, DeadlineExceeded])
            except ImportError:
                pass
        return retryable
    
    def _is_retryable_http_error(self, error: requests.exceptions.HTTPError) -> bool:
        """
        Check if an HTTP error should trigger retry.
        
        Args:
            error: HTTPError exception
        
        Returns:
            True if error is retryable (5xx, 429, or 404 for custom providers), False otherwise
        """
        if not hasattr(error, 'response') or error.response is None:
            return True  # If we can't check, assume retryable
        
        status_code = error.response.status_code
        
        # Retry on server errors (5xx) and rate limits (429)
        # Also retry on 404 for custom providers (vLLM may return 404 temporarily during model loading)
        if status_code >= 500 or status_code == 429:
            return True
        
        # For custom providers (like vLLM), 404 might be temporary (model loading, server restart)
        # Check if this is a custom provider by checking if api_url contains non-standard domains
        if status_code == 404 and self.provider == PROVIDER_CUSTOM:
            # Check if error message suggests model not found (permanent) vs server issue (temporary)
            error_msg = ""
            try:
                if hasattr(error, 'response') and error.response:
                    error_data = error.response.json()
                    error_msg = str(error_data.get('error', {}).get('message', '')).lower()
            except:
                pass
            
            # If error message indicates model doesn't exist, don't retry
            if 'model' in error_msg and ('does not exist' in error_msg or 'not found' in error_msg):
                return False
            
            # Otherwise, treat 404 as retryable for custom providers (might be temporary)
            return True
        
        return False
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts)
    
    @classmethod
    def from_config(cls, llm_config: Dict[str, Any]) -> "LLMClient":
        """
        Create LLMClient from configuration dictionary.
        
        Args:
            llm_config: Configuration dict from SafetyConfig.get_llm_config()
        
        Returns:
            LLMClient instance
        """
        return cls(
            api_key=llm_config.get("api_key"),
            api_url=llm_config.get("api_url"),
            model_name=llm_config.get("model_name"),
            provider=llm_config.get("provider", PROVIDER_OPENAI),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 512),
            timeout=llm_config.get("timeout", 30),
            max_retries=llm_config.get("max_retries", 3),
            retry_delay=llm_config.get("retry_delay", 1.0),
            retry_backoff=llm_config.get("retry_backoff", 2.0),
        )

