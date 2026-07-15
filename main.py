
import base64
import io
import os
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image

APP_NAME = "Car Damage Lab API"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description=(
        "API sperimentale per modificare gravità e superficie di un danno "
        "automotive usando una fotografia e una maschera."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clamp_percentage(value: int) -> int:
    if value < -100 or value > 100:
        raise HTTPException(
            status_code=422,
            detail="I valori percentuali devono essere compresi tra -100 e +100.",
        )
    return value


async def read_image(upload: UploadFile, mode: str) -> Image.Image:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"File vuoto: {upload.filename}")
    try:
        return Image.open(io.BytesIO(raw)).convert(mode)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Immagine non valida: {upload.filename}"
        ) from exc


def resize_mask(mask: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return mask


def alter_damage_area(mask: Image.Image, area_percent: int) -> Image.Image:
    """
    La maschera fornita dall'interfaccia è:
      bianco = zona modificabile
      nero   = zona protetta

    L'API OpenAI usa una maschera RGBA:
      alpha 0   = zona da rigenerare
      alpha 255 = zona protetta
    """
    gray = np.array(mask.convert("L"))
    binary = np.where(gray >= 128, 255, 0).astype(np.uint8)

    if area_percent != 0:
        height, width = binary.shape
        reference = max(3, round(min(width, height) * 0.10))
        radius = max(1, round(reference * abs(area_percent) / 100))
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        if area_percent > 0:
            binary = cv2.dilate(binary, kernel, iterations=1)
        else:
            binary = cv2.erode(binary, kernel, iterations=1)

    alpha = 255 - binary
    rgba = np.zeros((binary.shape[0], binary.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = 0
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def severity_instruction(severity: int) -> str:
    if severity == 0:
        return (
            "Mantieni sostanzialmente invariata la gravità visiva del danno, "
            "limitandoti ad armonizzare la zona modificata."
        )
    magnitude = abs(severity)
    if severity > 0:
        return (
            f"Aumenta la gravità visiva della deformazione di circa {magnitude}% "
            "rispetto alla fotografia originale. Rendi più evidenti profondità, "
            "pieghe, tensioni della lamiera, distacchi o rotture solo quando "
            "fisicamente compatibili, senza trasformare il danno in un incidente "
            "estremo se il valore è basso."
        )
    return (
        f"Riduci la gravità visiva della deformazione di circa {magnitude}% "
        "rispetto alla fotografia originale. Raddrizza progressivamente la parte, "
        "riduci pieghe, profondità, distacchi, graffi e rotture. A -100% mostra "
        "una riparazione professionale e realistica della sola zona selezionata."
    )


def area_instruction(area: int) -> str:
    if area == 0:
        return "Mantieni invariata l'estensione apparente del danno."
    direction = "estesa" if area > 0 else "ridotta"
    return (
        f"La superficie visibilmente danneggiata deve risultare {direction} di "
        f"circa {abs(area)}% rispetto all'originale, restando entro la zona "
        "consentita dalla maschera e rispettando i confini dei componenti."
    )


def build_prompt(severity: int, area: int) -> str:
    return f"""
Modifica fotografica automotive realistica e documentale.

Obiettivo:
- {severity_instruction(severity)}
- {area_instruction(area)}

Vincoli obbligatori:
- conserva la stessa automobile, modello, colore, prospettiva e inquadratura;
- conserva targa, logo, ruote, vetri, fanali non coinvolti e dettagli identificativi;
- conserva integralmente sfondo, strada, persone, edifici e illuminazione;
- intervieni esclusivamente nella zona indicata dalla maschera;
- mantieni riflessi, ombre, materiali e geometrie fisicamente plausibili;
- non aggiungere testo, watermark, veicoli, oggetti o componenti inesistenti;
- non cambiare la risoluzione logica o il rapporto d'aspetto;
- il risultato deve sembrare una fotografia reale, non un rendering.

Restituisci l'intera fotografia finale, con tutto ciò che non è interessato dal
danno visivamente identico all'originale.
""".strip()


def pil_to_named_bytes(image: Image.Image, name: str) -> tuple[str, io.BytesIO, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return (name, buffer, "image/png")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "mode": "mock" if os.getenv("MOCK_MODE", "false").lower() == "true" else "ai",
    }


@app.post("/v1/damage/edit")
async def edit_damage(
    image: UploadFile = File(..., description="Fotografia originale"),
    mask: UploadFile = File(..., description="Maschera: bianco modificabile, nero protetto"),
    severity_percent: int = Form(..., ge=-100, le=100),
    area_percent: int = Form(..., ge=-100, le=100),
    output_quality: Literal["low", "medium", "high", "auto"] = Form("medium"),
):
    severity_percent = clamp_percentage(severity_percent)
    area_percent = clamp_percentage(area_percent)

    source = await read_image(image, "RGB")
    source_mask = await read_image(mask, "L")
    source_mask = resize_mask(source_mask, source.size)
    api_mask = alter_damage_area(source_mask, area_percent)

    job_id = str(uuid.uuid4())
    prompt = build_prompt(severity_percent, area_percent)

    # Modalità di test: verifica collegamento Base44/API senza consumare crediti AI.
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        preview = source.copy()
        overlay = Image.new("RGBA", source.size, (255, 0, 0, 0))
        editable = 255 - np.array(api_mask.getchannel("A"))
        overlay_alpha = Image.fromarray((editable * 0.25).astype(np.uint8), mode="L")
        overlay.putalpha(overlay_alpha)
        preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")
        result_path = OUTPUT_DIR / f"{job_id}.jpg"
        preview.save(result_path, quality=92)
        return JSONResponse(
            {
                "job_id": job_id,
                "status": "completed",
                "mode": "mock",
                "severity_percent": severity_percent,
                "area_percent": area_percent,
                "result_base64": base64.b64encode(result_path.read_bytes()).decode("ascii"),
                "mime_type": "image/jpeg",
                "note": "Anteprima rossa della superficie elaborata; nessuna modifica AI.",
            }
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY non configurata. Usa MOCK_MODE=true per i test.",
        )

    client = OpenAI(api_key=api_key)
    source_file = pil_to_named_bytes(source, "source.png")[1]
    mask_file = pil_to_named_bytes(api_mask, "mask.png")[1]

    try:
        response = client.images.edit(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
            image=source_file,
            mask=mask_file,
            prompt=prompt,
            input_fidelity="high",
            quality=output_quality,
            size="auto",
            output_format="jpeg",
            output_compression=92,
            n=1,
        )
        encoded = response.data[0].b64_json
        result_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Errore motore immagini: {exc}") from exc

    result_path = OUTPUT_DIR / f"{job_id}.jpg"
    result_path.write_bytes(result_bytes)

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v1",
    }
