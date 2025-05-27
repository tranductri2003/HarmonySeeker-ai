import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from VoicesSeparator_Pipeline.inferrence import separate_audio

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ai/separate-voice")
async def separate_voice(file: UploadFile = File(...)):
    """
    Process voice separation on an audio file
    """
    MODEL_PATH = "VoicesSeparator_Pipeline/model_jax.keras"

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    result = separate_audio(file, MODEL_PATH)

    return StreamingResponse(
        content=result,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=separated_audio.zip"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
