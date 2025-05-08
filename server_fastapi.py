from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from your_model import predict_is_ai
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Classifier")

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/check-ai")
async def check_ai(
        image_data: str = Body(..., embed=True, alias="imageData")
) -> JSONResponse:
    """
    Обработка изображения и классификация с помощью ML модели

    - **image_data**: Data URL изображения (format: data:image/<type>;base64,<data>)
    """
    try:
        # Валидация данных
        if not image_data.startswith("data:image/"):
            raise HTTPException(
                status_code=422,
                detail="Invalid image format. Expected Data URL"
            )

        # Декодирование base64
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Логирование информации о размере изображения
        logger.info(f"Processing image size: {len(image_bytes) // 1024} KB")

        # Получение предсказания
        result = predict_is_ai(image_bytes)

        return JSONResponse(content={"result": result})

    except HTTPException as he:
        logger.error(f"Validation error: {he.detail}")
        raise

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Используем настройки логирования из logging
    )