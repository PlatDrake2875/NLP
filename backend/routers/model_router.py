# backend/routers/models_router.py

import httpx
import ollama
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import OLLAMA_BASE_URL, logger

# Import schemas and config using direct absolute paths
from schemas import OllamaModelInfo

router = APIRouter(
    prefix="/models",  # Prefix relative to the /api added in main.py
    tags=["models"],
)


@router.get(
    "", response_model=list[OllamaModelInfo]
)  # Path is relative to prefix -> /api/models
async def list_ollama_models_endpoint():
    # This is the exact code from your selected section in the Canvas
    logger.info("/api/models endpoint called.")
    logger.info(
        f"Attempting to connect to Ollama at OLLAMA_BASE_URL: {OLLAMA_BASE_URL}"
    )

    try:
        client = ollama.AsyncClient(host=OLLAMA_BASE_URL)

        logger.info(
            f"Ollama AsyncClient created for host {OLLAMA_BASE_URL}. Calling client.list()..."
        )
        ollama_list_response = await client.list()
        logger.debug(
            f"Received raw response object from Ollama client.list(): {ollama_list_response}"
        )

        if not hasattr(ollama_list_response, "models"):
            logger.error(
                f"Ollama ListResponse does not contain 'models' attribute. Full response object: {ollama_list_response}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Invalid response structure from Ollama (missing 'models' attribute in ListResponse)."
                },
            )

        models_from_ollama_lib = ollama_list_response.models

        if not isinstance(models_from_ollama_lib, list):
            logger.error(
                f"'models' attribute in Ollama ListResponse is not a list. Got {type(models_from_ollama_lib)}. Full response: {ollama_list_response}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Invalid data type for 'models' in Ollama ListResponse."
                },
            )

        logger.info(
            f"List of 'ollama.Model' objects from Ollama library contains {len(models_from_ollama_lib)} items."
        )

        parsed_models = []
        for ollama_model_obj in models_from_ollama_lib:
            try:
                if hasattr(ollama_model_obj, "model_dump"):
                    model_data_dict = ollama_model_obj.model_dump()
                elif hasattr(ollama_model_obj, "dict"):
                    model_data_dict = ollama_model_obj.dict()
                else:
                    logger.warning(
                        f"Skipping Ollama model object as it cannot be converted to dict: {ollama_model_obj}"
                    )
                    continue
            except Exception as e_dump:
                logger.error(
                    f"Failed to convert ollama.Model object to dictionary: {ollama_model_obj}. Error: {e_dump}",
                    exc_info=True,
                )
                continue

            if not isinstance(model_data_dict, dict):
                logger.warning(
                    f"Skipping item as it did not convert to a dictionary: {model_data_dict}"
                )
                continue

            try:
                model_info_instance = OllamaModelInfo.from_ollama(model_data_dict)
                if model_info_instance:
                    parsed_models.append(model_info_instance)
            except Exception as parse_exc:
                logger.error(
                    f"Error parsing individual model data derived from {ollama_model_obj}. Dict was: {model_data_dict}. Exception: {parse_exc}",
                    exc_info=True,
                )

        logger.info(
            f"Successfully parsed {len(parsed_models)} models for API response."
        )

        if not parsed_models and models_from_ollama_lib:
            logger.warning(
                "/api/models is returning an empty list because no models could be successfully parsed, though Ollama returned model data."
            )
        elif not models_from_ollama_lib:
            logger.info(
                "/api/models received no models from Ollama (Ollama's 'models' list was empty in ListResponse)."
            )

        return parsed_models

    except httpx.TimeoutException as e:
        logger.error(
            f"Timeout connecting to Ollama at {OLLAMA_BASE_URL} for /api/models: {e}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=504,
            content={"detail": f"Timeout connecting to Ollama: {str(e)}"},
        )
    except httpx.RequestError as e:
        logger.error(
            f"httpx.RequestError connecting to Ollama at {OLLAMA_BASE_URL} for /api/models: {e}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": f"Could not connect to Ollama service: {str(e)}"},
        )
    except ollama.ResponseError as e:
        logger.error(
            f"Ollama API responded with an error for /api/models: {getattr(e, 'error', 'Unknown Ollama error')} (Status: {getattr(e, 'status_code', 'N/A')})",
            exc_info=True,
        )
        status_code_to_return = getattr(e, "status_code", 500)
        if (
            not isinstance(status_code_to_return, int)
            or status_code_to_return < 100
            or status_code_to_return > 599
        ):
            status_code_to_return = 500
        return JSONResponse(
            status_code=status_code_to_return,
            content={
                "detail": f"Ollama API error: {getattr(e, 'error', 'Failed to communicate with Ollama')}"
            },
        )
    except ollama.OllamaError as e:
        logger.error(
            f"Ollama library error occurred in /api/models: {e}", exc_info=True
        )
        return JSONResponse(
            status_code=500, content={"detail": f"Ollama library error: {str(e)}"}
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in /api/models endpoint: {e}", exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"An unexpected server error occurred while fetching models: {str(e)}"
            },
        )
