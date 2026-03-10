"""Qwen3 TTS API - Text-to-speech with voice cloning on Modal."""

import modal

R2_BUCKET_NAME = "resonance-app"
R2_ACCOUNT_ID = "d62ff5aad44573d8544880797e50e5ee"
R2_MOUNT_PATH = "/r2"

r2_bucket = modal.CloudBucketMount(
    R2_BUCKET_NAME,
    bucket_endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    secret=modal.Secret.from_name("cloudflare-r2"),
    read_only=True,
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "qwen-tts",
        "fastapi[standard]",
        "torchaudio",
    )
)

app = modal.App("qwen3-tts", image=image)

with image.imports():
    import io
    import os
    from pathlib import Path

    import torch
    import torchaudio as ta

    from fastapi import FastAPI, HTTPException, Depends, Security
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field

    from qwen_tts import Qwen3TTSModel

    api_key_scheme = APIKeyHeader(
        name="x-api-key",
        scheme_name="ApiKeyAuth",
        auto_error=False,
    )

    def verify_api_key(x_api_key: str | None = Security(api_key_scheme)):
        expected = os.environ.get("CHATTERBOX_API_KEY", "")
        if not expected or x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return x_api_key

    class TTSRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=5000)
        voice_key: str = Field(..., min_length=1, max_length=300)


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("chatterbox-api-key"),
        modal.Secret.from_name("cloudflare-r2"),
    ],
    volumes={R2_MOUNT_PATH: r2_bucket},
)
@modal.concurrent(max_inputs=10)
class QwenTTS:

    @modal.enter()
    def load_model(self):

        # Use smaller model for latency
        model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        self.sample_rate = 24000

    @modal.asgi_app()
    def serve(self):

        web_app = FastAPI(
            title="Qwen3 TTS API",
            docs_url="/docs",
            dependencies=[Depends(verify_api_key)],
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/generate", responses={200: {"content": {"audio/wav": {}}}})
        def generate_speech(request: TTSRequest):

            voice_path = Path(R2_MOUNT_PATH) / request.voice_key

            if not voice_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice not found at '{request.voice_key}'",
                )

            try:
                audio_bytes = self.generate.local(
                    request.prompt,
                    str(voice_path),
                )

                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/wav",
                )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"TTS generation failed: {e}",
                )

        return web_app

    @modal.method()
    def generate(self, prompt: str, ref_audio: str):

        wav = self.model.generate(
            text=prompt,
            ref_audio=ref_audio,   # voice cloning reference
        )

        buffer = io.BytesIO()

        ta.save(
            buffer,
            torch.tensor(wav).unsqueeze(0),
            self.sample_rate,
            format="wav",
        )

        buffer.seek(0)

        return buffer.read()


@app.local_entrypoint()
def test(
    prompt: str = "Hello from Qwen3 TTS on Modal.",
    voice_key: str = "voices/system/default.wav",
    output_path: str = "/tmp/qwen3/output.wav",
):

    import pathlib

    tts = QwenTTS()

    audio_bytes = tts.generate.remote(
        prompt,
        f"{R2_MOUNT_PATH}/{voice_key}",
    )

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)

    print(f"Audio saved to {output_file}")