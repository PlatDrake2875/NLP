# backend/routers/chat_router.py
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Optional

import httpx
from fastapi import APIRouter
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

# --- Direct Imports ---

# Attempt to import configurations
try:
    from config import (
        NEMO_GUARDRAILS_SERVER_URL,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL_FOR_RAG,
        RAG_ENABLED,
        USE_GUARDRAILS,
        logger,  # Use the centrally configured logger
    )

    logger.info("Successfully imported settings from config.py in chat_router.")
except ImportError as e:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    logger = logging.getLogger("chat_router_fallback_logger")
    logger.error(
        f"CRITICAL: Could not import all settings from config.py: {e}. Using environment variables or defaults. Ensure all expected variables (NEMO_GUARDRAILS_SERVER_URL, USE_GUARDRAILS, etc.) are in config.py or environment."
    )

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_FOR_RAG = os.getenv("OLLAMA_MODEL_FOR_RAG", "llama3")
    RAG_ENABLED = os.getenv("RAG_ENABLED", "False").lower() == "true"
    NEMO_GUARDRAILS_SERVER_URL = os.getenv(
        "NEMO_GUARDRAILS_SERVER_URL", "http://nemo-guardrails:8001"
    )
    USE_GUARDRAILS = os.getenv("USE_GUARDRAILS", "false").lower() == "true"

try:
    from rag_components import get_llm_for_automation, get_rag_context_prefix

    logger.info(
        "Successfully imported get_rag_context_prefix and get_llm_for_automation from rag_components."
    )
except ImportError as e:
    logger.critical(
        f"CRITICAL IMPORT ERROR in chat_router: Failed to import from rag_components: {e}. RAG/Automation features might fail severely."
    )

    async def get_rag_context_prefix(query: str) -> Optional[str]:  # Dummy function
        logger.error(
            "CRITICAL DUMMY: get_rag_context_prefix is not available due to import error."
        )
        return None

    def get_llm_for_automation():  # Dummy function
        logger.error(
            "CRITICAL DUMMY: get_llm_for_automation is not available due to import error."
        )
        return None


# Import local NeMo Guardrails integration
try:
    from nemo_guardrails_local import get_local_nemo_instance

    logger.info("Successfully imported local NeMo Guardrails integration.")
    LOCAL_NEMO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Local NeMo Guardrails not available: {e}")
    LOCAL_NEMO_AVAILABLE = False


# --- Router Setup ---
router = APIRouter(
    tags=["chat"],
)

# --- Helper Async Generators for Streaming ---


async def _direct_ollama_stream(
    model_name: str, messages_payload: list[dict[str, str]], stream_id: str
) -> AsyncGenerator[str, None]:
    logger.info(
        f"[{stream_id}] ENTERING _direct_ollama_stream for model '{model_name}'. Target: {OLLAMA_BASE_URL}/api/chat"
    )
    logger.debug(
        f"[{stream_id}] Direct Ollama: Sending messages payload (count: {len(messages_payload)}): {json.dumps(messages_payload, indent=2)}"
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages_payload,
                    "stream": True,
                },
            ) as response:
                logger.info(
                    f"[{stream_id}] Direct Ollama: Response status code: {response.status_code}"
                )
                if response.status_code != 200:
                    error_content_bytes = await response.aread()
                    error_content_str = error_content_bytes.decode(errors="replace")
                    err_msg = (
                        f"Ollama API error {response.status_code}: {error_content_str}"
                    )
                    logger.error(f"[{stream_id}] Direct Ollama: {err_msg}")
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return

                logger.info(
                    f"[{stream_id}] Direct Ollama: Successfully connected to stream."
                )
                chunk_index = 0
                async for line in response.aiter_lines():
                    logger.debug(f"[{stream_id}] Direct Ollama: Received line: {line}")
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            token = chunk_data.get("message", {}).get("content", "")
                            is_done = chunk_data.get("done", False)

                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                                chunk_index += 1
                            if is_done:
                                logger.info(
                                    f"[{stream_id}] Direct Ollama: stream reported done=true after {chunk_index} token chunks."
                                )
                                break
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{stream_id}] Direct Ollama: Failed to decode JSON line: {line}"
                            )
                        except Exception as e_parse:
                            logger.error(
                                f"[{stream_id}] Direct Ollama: Error processing chunk: {line}. Error: {e_parse}",
                                exc_info=True,
                            )

        logger.info(f"[{stream_id}] Direct Ollama: Yielding final done status.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"
    except httpx.RequestError as e_req:
        logger.error(
            f"[{stream_id}] Direct Ollama: HTTP request error: {e_req}", exc_info=True
        )
        yield f"data: {json.dumps({'error': f'Could not connect to Ollama service: {str(e_req)}'})}\n\n"
    except Exception as e_direct:
        logger.error(
            f"[{stream_id}] Direct Ollama: Unhandled exception: {e_direct}",
            exc_info=True,
        )
        yield f"data: {json.dumps({'error': f'Server error during direct Ollama call: {str(e_direct)}'})}\n\n"
    finally:
        logger.info(f"[{stream_id}] EXITING _direct_ollama_stream.")


async def _local_nemo_guardrails_stream(
    messages_for_llm: list[dict], stream_id: str, model_name: str
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion using local NeMo Guardrails integration.

    Args:
        messages_for_llm: List of message dictionaries
        stream_id: Stream identifier for logging
        model_name: Model name to use

    Yields:
        str: SSE formatted chunks
    """
    logger.info(f"[{stream_id}] ENTERING _local_nemo_guardrails_stream with local integration.")

    try:
        if not LOCAL_NEMO_AVAILABLE:
            error_msg = "Local NeMo Guardrails integration not available"
            logger.error(f"[{stream_id}] {error_msg}")
            yield f'data: {{"error": "{error_msg}"}}\n\n'
            return

        # Get the local NeMo instance
        logger.info(f"[{stream_id}] Getting local NeMo Guardrails instance...")
        nemo_instance = await get_local_nemo_instance()

        if not nemo_instance.is_available():
            error_msg = "Local NeMo Guardrails instance not properly initialized"
            logger.error(f"[{stream_id}] {error_msg}")
            yield f'data: {{"error": "{error_msg}"}}\n\n'
            return

        logger.info(f"[{stream_id}] Local NeMo instance ready, processing chat completion...")

        # Use the local NeMo integration for chat completion
        response = await nemo_instance.chat_completion(
            messages=messages_for_llm,
            model=model_name,
            stream=False  # We'll handle streaming ourselves
        )

        if response and "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
            logger.info(f"[{stream_id}] Local NeMo generated response ({len(content)} chars)")

            # Stream the response in SSE format
            chunk_data = {
                "id": response.get("id", f"local-{int(time.time())}"),
                "object": "chat.completion.chunk",
                "created": response.get("created", int(time.time())),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

            # Final chunk
            final_chunk = {
                "id": response.get("id", f"local-{int(time.time())}"),
                "object": "chat.completion.chunk",
                "created": response.get("created", int(time.time())),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        else:
            error_msg = "No valid response from local NeMo Guardrails"
            logger.error(f"[{stream_id}] {error_msg}")
            yield f'data: {{"error": "{error_msg}"}}\n\n'

    except Exception as e:
        error_msg = f"Local NeMo Guardrails error: {str(e)}"
        logger.error(f"[{stream_id}] {error_msg}", exc_info=True)
        yield f'data: {{"error": "{error_msg}"}}\n\n'

    finally:
        logger.info(f"[{stream_id}] EXITING _local_nemo_guardrails_stream.")


async def _guardrails_ollama_stream(
    model_name_for_guardrails: str,
    messages_payload: list[dict[str, str]],
    stream_id: str,
) -> AsyncGenerator[str, None]:
    guardrails_endpoint = f"{NEMO_GUARDRAILS_SERVER_URL}/v1/chat/completions"
    logger.info(
        f"[{stream_id}] ENTERING _guardrails_ollama_stream. Target: '{guardrails_endpoint}', Model for Guardrails: '{model_name_for_guardrails}'."
    )

    guardrails_payload = {
        "model": model_name_for_guardrails,
        "messages": messages_payload,
        "config_id": "mybot",
        "stream": True,
    }

    logger.info(
        f"[{stream_id}] Guardrails: Sending payload: {json.dumps(guardrails_payload, indent=2)}"
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            logger.debug(
                f"[{stream_id}] Guardrails: AsyncClient created. Attempting to stream POST request to {guardrails_endpoint}"
            )
            async with client.stream(
                "POST", guardrails_endpoint, json=guardrails_payload
            ) as response:
                logger.info(
                    f"[{stream_id}] Guardrails: Response status code: {response.status_code}"
                )
                logger.debug(
                    f"[{stream_id}] Guardrails: Response headers: {json.dumps(dict(response.headers))}"
                )

                # It's possible to get 200 OK but still have an error in the stream content from Guardrails
                # So we will process the stream and look for errors there too.

                # Read the entire stream content for debugging if not 200, or if streaming fails later
                raw_response_content_for_debug = ""
                if response.status_code != 200:
                    try:
                        error_content_bytes = await response.aread()
                        raw_response_content_for_debug = error_content_bytes.decode(
                            errors="replace"
                        )
                        err_msg = f"NeMo Guardrails API error {response.status_code}: {raw_response_content_for_debug}"
                        logger.error(f"[{stream_id}] Guardrails: {err_msg}")
                        yield f"data: {json.dumps({'error': err_msg})}\n\n"
                        return
                    except Exception as e_read_err:
                        logger.error(
                            f"[{stream_id}] Guardrails: Could not read error response body: {e_read_err}"
                        )
                        yield f"data: {json.dumps({'error': f'NeMo Guardrails API error {response.status_code} (body unreadable)'})}\n\n"
                        return

                logger.info(
                    f"[{stream_id}] Guardrails: Successfully initiated stream connection (HTTP Status: {response.status_code})."
                )
                chunk_index = 0
                full_response_lines_debug = []  # Collect all lines for debugging if stream is empty

                async for line in response.aiter_lines():
                    logger.debug(
                        f"[{stream_id}] Guardrails: Received line from stream: '{line}'"
                    )
                    full_response_lines_debug.append(line)

                    if line.startswith("data: "):
                        line_content = line[len("data: ") :].strip()
                        if line_content == "[DONE]":
                            logger.info(
                                f"[{stream_id}] Guardrails: Stream reported [DONE] after {chunk_index} token chunks."
                            )
                            break
                        if not line_content:  # Empty data message, skip
                            logger.debug(
                                f"[{stream_id}] Guardrails: Received empty data message, skipping."
                            )
                            continue
                        try:
                            chunk_data = json.loads(line_content)
                            logger.debug(
                                f"[{stream_id}] Guardrails: Parsed chunk_data: {json.dumps(chunk_data)}"
                            )

                            # Check for explicit error message from Guardrails within the stream
                            # This structure might vary based on Guardrails version and error type
                            if (
                                "object" in chunk_data
                                and chunk_data["object"] == "error"
                            ):
                                err_msg = chunk_data.get(
                                    "message", "Unknown error from Guardrails stream"
                                )
                                logger.error(
                                    f"[{stream_id}] Guardrails: Error in stream chunk: {err_msg} - Full chunk: {chunk_data}"
                                )
                                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                                continue  # Or break, depending on desired behavior

                            if "detail" in chunk_data:  # Standard FastAPI error format
                                err_detail = chunk_data["detail"]
                                if (
                                    isinstance(err_detail, list)
                                    and err_detail
                                    and "msg" in err_detail[0]
                                ):
                                    err_msg = f"Error from Guardrails stream: {err_detail[0]['msg']}"
                                elif isinstance(err_detail, str):
                                    err_msg = (
                                        f"Error from Guardrails stream: {err_detail}"
                                    )
                                else:
                                    err_msg = f"Unknown error structure from Guardrails stream: {err_detail}"
                                logger.error(
                                    f"[{stream_id}] Guardrails: {err_msg} - Full chunk: {chunk_data}"
                                )
                                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                                continue

                            token = (
                                chunk_data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            finish_reason = chunk_data.get("choices", [{}])[0].get(
                                "finish_reason"
                            )

                            if token:
                                logger.debug(
                                    f"[{stream_id}] Guardrails: Yielding token: '{token}'"
                                )
                                yield f"data: {json.dumps({'token': token})}\n\n"
                                chunk_index += 1
                            if finish_reason:
                                logger.info(
                                    f"[{stream_id}] Guardrails: Stream reported finish_reason: {finish_reason} after {chunk_index} token chunks."
                                )
                                break
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{stream_id}] Guardrails: Failed to decode JSON line: '{line_content}'"
                            )
                        except (IndexError, KeyError, TypeError) as e_struct:
                            logger.warning(
                                f"[{stream_id}] Guardrails: Unexpected JSON structure in line: '{line_content}'. Error: {e_struct}"
                            )
                        except Exception as e_parse:
                            logger.error(
                                f"[{stream_id}] Guardrails: Error processing chunk: '{line_content}'. Error: {e_parse}",
                                exc_info=True,
                            )
                    elif line.strip():  # Log non-empty, non-SSE lines for debugging
                        logger.debug(
                            f"[{stream_id}] Guardrails: Received non-SSE (and non-empty) line: '{line}'"
                        )

                if chunk_index == 0 and not any(
                    "[DONE]" in s for s in full_response_lines_debug
                ):
                    logger.warning(
                        f"[{stream_id}] Guardrails: Stream ended but no valid token chunks or [DONE] message received. Full raw response lines for debugging: {full_response_lines_debug}"
                    )
                    # If the raw response was already captured due to non-200, don't overwrite
                    if (
                        response.status_code == 200
                        and not raw_response_content_for_debug
                    ):
                        try:
                            # Attempt to read the full body if stream was empty but status was 200
                            # This might already be consumed by aiter_lines, but worth a try for debugging.
                            # response.content is not available on a streaming response after iteration.
                            # We rely on full_response_lines_debug here.
                            logger.warning(
                                f"[{stream_id}] Guardrails: Attempting to log full response as stream was empty. Lines: {''.join(full_response_lines_debug)}"
                            )
                        except Exception as e_read_full:
                            logger.error(
                                f"[{stream_id}] Guardrails: Could not read full response body after empty stream: {e_read_full}"
                            )

        logger.info(f"[{stream_id}] Guardrails: Yielding final done status.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    except httpx.ConnectError as e_conn:
        logger.error(
            f"[{stream_id}] Guardrails: HTTP ConnectError: {e_conn}", exc_info=True
        )
        yield f"data: {json.dumps({'error': f'Could not connect to NeMo Guardrails service (ConnectError): {str(e_conn)}'})}\n\n"
    except httpx.RequestError as e_req:
        logger.error(
            f"[{stream_id}] Guardrails: HTTP RequestError: {e_req}", exc_info=True
        )
        yield f"data: {json.dumps({'error': f'NeMo Guardrails service request failed: {str(e_req)}'})}\n\n"
    except Exception as e_guard:
        logger.error(
            f"[{stream_id}] Guardrails: Unhandled exception: {e_guard}", exc_info=True
        )
        yield f"data: {json.dumps({'error': f'Server error during Guardrails call: {str(e_guard)}'})}\n\n"
    finally:
        logger.info(f"[{stream_id}] EXITING _guardrails_ollama_stream.")


# --- API Endpoints ---
@router.post("/chat")
async def chat_endpoint(fastapi_req: FastAPIRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    try:
        actual_body = await fastapi_req.json()
    except json.JSONDecodeError:
        logger.error(f"[{request_id}] /chat: Invalid JSON in request body.")
        return JSONResponse(
            status_code=400, content={"detail": "Invalid JSON in request body."}
        )

    logger.info(
        f"[{request_id}] ENTERING /chat endpoint. RAG_ENABLED={RAG_ENABLED}, USE_GUARDRAILS={USE_GUARDRAILS}. Raw body: {json.dumps(actual_body)}"
    )

    query = actual_body.get("query")
    model_name_from_request = actual_body.get("model")
    history = actual_body.get("history", [])
    use_rag_for_this_request = actual_body.get("use_rag", RAG_ENABLED)

    if not query:
        logger.error(f"[{request_id}] /chat: 'query' field is missing.")
        return JSONResponse(
            status_code=400, content={"detail": "'query' field is required."}
        )
    if not isinstance(history, list):
        logger.warning(
            f"[{request_id}] /chat: Invalid or missing 'history', defaulting to empty list."
        )
        history = []

    effective_model_name = model_name_from_request or OLLAMA_MODEL_FOR_RAG
    logger.info(f"[{request_id}] Effective LLM for generation: {effective_model_name}")

    messages_for_llm: list[dict[str, str]] = []
    rag_prefix_generated = False

    for i, msg in enumerate(history):
        role = msg.get("sender", "user").lower()
        content = msg.get("text", "")
        logger.debug(
            f"[{request_id}] Processing history message {i}: Role='{role}', Content snippet='{content[:50]}...'"
        )
        if role == "bot":
            messages_for_llm.append({"role": "assistant", "content": content})
        else:
            messages_for_llm.append({"role": "user", "content": content})

    current_user_query_content = query
    logger.debug(f"[{request_id}] Current user query: '{current_user_query_content}'")

    if RAG_ENABLED and use_rag_for_this_request:
        logger.info(
            f"[{request_id}] Attempting to generate RAG context prefix for query: '{current_user_query_content}'"
        )
        try:
            rag_enhanced_prompt = await get_rag_context_prefix(
                current_user_query_content
            )
        except Exception as e_rag:
            logger.error(
                f"[{request_id}] Error during get_rag_context_prefix: {e_rag}",
                exc_info=True,
            )
            rag_enhanced_prompt = None

        if rag_enhanced_prompt:
            messages_for_llm.append({"role": "user", "content": rag_enhanced_prompt})
            rag_prefix_generated = True
            logger.info(
                f"[{request_id}] Using RAG-enhanced prompt as final user message (length: {len(rag_enhanced_prompt)})."
            )
            logger.debug(
                f"[{request_id}] RAG-enhanced prompt snippet: {rag_enhanced_prompt[:100]}..."
            )
        else:
            logger.info(
                f"[{request_id}] Failed to generate RAG prefix or no documents found. Using original query for final user message."
            )
            messages_for_llm.append(
                {"role": "user", "content": current_user_query_content}
            )
    else:
        logger.info(
            f"[{request_id}] RAG not used for this request. Using original query for final user message."
        )
        messages_for_llm.append({"role": "user", "content": current_user_query_content})

    stream_id = f"stream_{request_id}"

    if USE_GUARDRAILS:
        logger.info(
            f"[{stream_id}] Routing to NeMo Guardrails. RAG prefix used: {rag_prefix_generated}. Final messages count: {len(messages_for_llm)}"
        )

        # Try local NeMo integration first, fallback to HTTP service
        if LOCAL_NEMO_AVAILABLE:
            logger.info(f"[{stream_id}] Using local NeMo Guardrails integration")
            generator = _local_nemo_guardrails_stream(
                messages_for_llm=messages_for_llm,
                stream_id=stream_id,
                model_name=effective_model_name,
            )
        else:
            logger.info(f"[{stream_id}] Using HTTP NeMo Guardrails service")
            generator = _guardrails_ollama_stream(
                model_name_for_guardrails=effective_model_name,
                messages_payload=messages_for_llm,
                stream_id=stream_id,
            )
    else:
        logger.info(
            f"[{stream_id}] Routing directly to Ollama. RAG prefix used: {rag_prefix_generated}. Final messages count: {len(messages_for_llm)}"
        )
        generator = _direct_ollama_stream(
            model_name=effective_model_name,
            messages_payload=messages_for_llm,
            stream_id=stream_id,
        )
    logger.info(f"[{request_id}] EXITING /chat endpoint, returning stream.")
    return StreamingResponse(generator, media_type="text/event-stream")
