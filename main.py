import base64
import io
import os
import re
import traceback
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

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
    version="0.3.0",
    description=(
        "API per modificare gravità e superficie di un danno automotive "
        "usando sempre una fotografia originale e una maschera."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DamageEditBase64Request(BaseModel):
    image_base64: str = Field(..., min_length=16)
    mask_base64: str = Field(..., min_length=16)
    severity_percent: int = Field(..., ge=-100, le=100)
    area_percent: int = Field(..., ge=-100, le=100)
    output_quality: Literal["low", "medium", "high", "auto"] = "medium"
    user_instructions: str = Field(default="", max_length=500)
    selection_mode: str = Field(default="manual")


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
        raise HTTPException(
            status_code=400,
            detail=f"File vuoto: {upload.filename}",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert(mode)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Immagine non valida: {upload.filename}",
        ) from exc


def resize_mask(mask: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return mask


def adjust_selection_mask(mask: Image.Image, area_percent: int) -> Image.Image:
    """
    Restituisce una maschera L:
      bianco = area modificabile
      nero   = area protetta
    """
    gray = np.array(mask.convert("L"))
    binary = np.where(gray >= 128, 255, 0).astype(np.uint8)

    if area_percent != 0:
        height, width = binary.shape
        reference = max(3, round(min(width, height) * 0.10))
        radius = max(1, round(reference * abs(area_percent) / 100))
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        if area_percent > 0:
            binary = cv2.dilate(binary, kernel, iterations=1)
        else:
            binary = cv2.erode(binary, kernel, iterations=1)

    return Image.fromarray(binary, mode="L")


def validate_manual_mask(mask: Image.Image) -> dict:
    gray = np.array(mask.convert("L"))
    selected = gray >= 128
    selected_pixels = int(selected.sum())
    total_pixels = int(selected.size)

    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera è vuota. Disegna la zona da modificare.",
        )

    coverage = selected_pixels / total_pixels

    if coverage > 0.55:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La maschera copre il {coverage * 100:.1f}% della fotografia. "
                "Riduci la selezione per evitare la rigenerazione dell'intera scena."
            ),
        )

    ys, xs = np.where(selected)
    bbox = {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }

    return {"coverage": coverage, "bbox": bbox}


def api_mask_from_selection(selection: Image.Image) -> Image.Image:
    """
    Maschera RGBA per OpenAI:
      alpha 0   = area modificabile
      alpha 255 = area protetta
    """
    binary = np.where(
        np.array(selection.convert("L")) >= 128,
        255,
        0,
    ).astype(np.uint8)

    rgba = np.zeros((binary.shape[0], binary.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = 0
    rgba[:, :, 3] = 255 - binary
    return Image.fromarray(rgba, mode="RGBA")


def severity_instruction(severity: int) -> str:
    if severity == 0:
        return (
            "Mantieni invariata la gravità del danno e armonizza soltanto "
            "la zona modificata."
        )

    magnitude = abs(severity)

    if severity > 0:
        return (
            f"Aumenta la gravità visiva della deformazione di circa {magnitude}% "
            "rispetto all'originale. Accentua profondità e pieghe in modo graduale "
            "e fisicamente plausibile."
        )

    return (
        f"Riduci la gravità visiva della deformazione di circa {magnitude}% "
        "rispetto all'originale. Riduci progressivamente profondità, pieghe, "
        "graffi e rotture nella sola zona selezionata."
    )


def area_instruction(area: int) -> str:
    if area == 0:
        return "Mantieni invariata l'estensione apparente del danno."

    direction = "aumentata" if area > 0 else "ridotta"
    return (
        f"La superficie danneggiata deve risultare {direction} di circa "
        f"{abs(area)}%, restando dentro la zona consentita."
    )


def build_prompt(
    severity: int,
    area: int,
    user_instructions: str = "",
) -> str:
    cleaned = user_instructions.strip()
    extra = ""

    if cleaned:
        extra = f"""
Indicazione aggiuntiva dell'utente:
{cleaned}

Applica questa indicazione soltanto all'interno della zona modificabile.
""".strip()

    return f"""
Modifica fotografica automotive realistica.

La fotografia originale è il riferimento vincolante.
Modifica esclusivamente il danno compreso nella zona trasparente della maschera.

Obiettivi:
- {severity_instruction(severity)}
- {area_instruction(area)}

{extra}

Vincoli obbligatori:
- conserva la stessa automobile, marca, modello, colore e allestimento;
- conserva prospettiva, inquadratura, sfondo e illuminazione;
- conserva targa, logo, ruota, fanali, vetri e maniglie;
- non generare un altro veicolo;
- non cambiare la posizione dell'automobile;
- non aggiungere oggetti, testo, persone o watermark;
- fuori dalla maschera non deve cambiare nulla;
- il risultato deve sembrare una fotografia reale.
""".strip()


def pil_to_file(image: Image.Image, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def output_size_for(source: Image.Image) -> str:
    width, height = source.size
    ratio = width / height if height else 1.0

    if ratio > 1.15:
        return "1536x1024"
    if ratio < 0.87:
        return "1024x1536"
    return "1024x1024"


def call_openai_image_edit(
    source: Image.Image,
    api_mask: Image.Image,
    prompt: str,
    quality: Literal["low", "medium", "high", "auto"],
) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY non configurata.",
        )

    client = OpenAI(api_key=api_key)

    request_args = {
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "image": pil_to_file(source.convert("RGB"), "source.png"),
        "mask": pil_to_file(api_mask.convert("RGBA"), "mask.png"),
        "prompt": prompt,
        "quality": quality,
        "size": output_size_for(source),
        "background": "opaque",
        "output_format": "jpeg",
        "output_compression": 92,
        "n": 1,
    }

    try:
        response = client.images.edit(**request_args)

        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("OpenAI ha risposto senza dati immagine.")

        return base64.b64decode(response.data[0].b64_json)

    except Exception as exc:
        request_id = getattr(exc, "request_id", None)
        print("OPENAI IMAGE ERROR:", repr(exc), flush=True)
        if request_id:
            print("OPENAI REQUEST ID:", request_id, flush=True)
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Errore motore immagini OpenAI",
                "type": type(exc).__name__,
                "error": str(exc),
                "request_id": request_id,
            },
        ) from exc


def composite_generated_full_image(
    source: Image.Image,
    generated_bytes: bytes,
    selection: Image.Image,
) -> bytes:
    """
    Ricompone sempre una fotografia completa:
    - fuori dalla selezione: pixel originali invariati;
    - dentro la selezione: pixel generati dall'AI.
    """
    original = source.convert("RGB")
    generated = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
    generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    mask = selection.convert("L")
    feather_radius = max(1.0, min(original.size) * 0.0025)
    blend_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    result = Image.composite(generated, original, blend_mask)

    # Verifica di sicurezza: il risultato deve avere esattamente
    # le dimensioni della fotografia originale.
    if result.size != original.size:
        raise RuntimeError(
            f"Dimensione risultato non valida: {result.size}, attesa {original.size}"
        )

    buffer = io.BytesIO()
    result.save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )
    return buffer.getvalue()


def edit_manual_full_image_locked(
    source: Image.Image,
    source_mask: Image.Image,
    severity_percent: int,
    area_percent: int,
    user_instructions: str,
    quality: Literal["low", "medium", "high", "auto"],
) -> bytes:
    """
    Invia a OpenAI la fotografia completa e ricompone localmente il risultato.
    In questo modo il backend restituisce sempre l'immagine completa e impedisce
    tecnicamente modifiche fuori dalla maschera.
    """
    selection = adjust_selection_mask(source_mask, area_percent)
    mask_info = validate_manual_mask(selection)

    print(
        "MANUAL MASK:",
        f"coverage={mask_info['coverage'] * 100:.2f}%",
        f"bbox={mask_info['bbox']}",
        f"source_size={source.size}",
        flush=True,
    )

    api_mask = api_mask_from_selection(selection)
    prompt = build_prompt(
        severity_percent,
        area_percent,
        user_instructions,
    )

    generated_bytes = call_openai_image_edit(
        source,
        api_mask,
        prompt,
        quality,
    )

    result_bytes = composite_generated_full_image(
        source,
        generated_bytes,
        selection,
    )

    # Diagnostica utile nei log Render.
    result_image = Image.open(io.BytesIO(result_bytes))
    print(
        "RESULT IMAGE:",
        f"size={result_image.size}",
        f"bytes={len(result_bytes)}",
        flush=True,
    )

    return result_bytes


def _open_image_bytes(raw: bytes, mode: str) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert(mode)
    except Exception:
        pass

    try:
        array = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return None

        if decoded.ndim == 2:
            pil_image = Image.fromarray(decoded, mode="L")
        elif decoded.shape[2] == 4:
            pil_image = Image.fromarray(
                cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA),
                mode="RGBA",
            )
        else:
            pil_image = Image.fromarray(
                cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
                mode="RGB",
            )

        return pil_image.convert(mode)
    except Exception:
        return None


def _looks_like_hex_image(value: str) -> bool:
    cleaned = "".join(value.strip().lower().split())
    return (
        len(cleaned) >= 8
        and len(cleaned) % 2 == 0
        and re.fullmatch(r"[0-9a-f]+", cleaned) is not None
        and (
            cleaned.startswith("ffd8ff")
            or cleaned.startswith("89504e47")
            or cleaned.startswith("52494646")
        )
    )


def _strip_data_url(value: str, label: str) -> str:
    data = value.strip().strip('"').strip("'")
    if data.startswith("data:"):
        if "," not in data:
            raise HTTPException(
                status_code=400,
                detail=f"Data URL non valido: {label}",
            )
        data = data.split(",", 1)[1]

    return "".join(data.split())


def decode_base64_image(value: str, label: str, mode: str) -> Image.Image:
    original = value.strip().strip('"').strip("'")

    if _looks_like_hex_image(original):
        try:
            raw = bytes.fromhex("".join(original.split()))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"HEX non valido: {label}",
            ) from exc

        image = _open_image_bytes(raw, mode)
        if image is not None:
            return image

    data = _strip_data_url(original, label)

    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Base64 non valido: {label}",
        ) from exc

    if not raw:
        raise HTTPException(
            status_code=400,
            detail=f"Immagine vuota: {label}",
        )

    image = _open_image_bytes(raw, mode)
    if image is not None:
        return image

    try:
        nested_text = raw.decode("utf-8").strip().strip('"').strip("'")
    except UnicodeDecodeError:
        nested_text = ""

    if nested_text:
        if _looks_like_hex_image(nested_text):
            try:
                nested_hex_raw = bytes.fromhex("".join(nested_text.split()))
            except ValueError:
                nested_hex_raw = b""

            if nested_hex_raw:
                image = _open_image_bytes(nested_hex_raw, mode)
                if image is not None:
                    return image

        nested_data = _strip_data_url(nested_text, label)
        try:
            nested_raw = base64.b64decode(nested_data, validate=True)
        except Exception:
            nested_raw = b""

        if nested_raw:
            image = _open_image_bytes(nested_raw, mode)
            if image is not None:
                return image

    prefix = raw[:24].hex()
    raise HTTPException(
        status_code=400,
        detail=(
            f"Immagine Base64 non decodificabile: {label} "
            f"({len(raw)} byte, prefisso={prefix})"
        ),
    )


def make_mock_result(
    source: Image.Image,
    selection: Image.Image,
    job_id: str,
    severity_percent: int,
    area_percent: int,
) -> dict:
    preview = source.copy().convert("RGBA")
    overlay = Image.new("RGBA", source.size, (255, 0, 0, 0))
    overlay_alpha = Image.fromarray(
        (np.array(selection.convert("L")) * 0.25).astype(np.uint8),
        mode="L",
    )
    overlay.putalpha(overlay_alpha)
    preview = Image.alpha_composite(preview, overlay).convert("RGB")

    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=92)
    result_bytes = buffer.getvalue()

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "mock",
        "selection_mode": "manual",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "note": "Anteprima rossa della superficie elaborata.",
    }


@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "mode": (
            "mock"
            if os.getenv("MOCK_MODE", "false").lower() == "true"
            else "ai"
        ),
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "prompt_version": "damage-v7-full-image-composite-locked",
    }


@app.post("/v1/damage/edit")
async def edit_damage(
    image: UploadFile = File(..., description="Fotografia originale"),
    mask: UploadFile = File(
        ...,
        description="Maschera: bianco modificabile, nero protetto",
    ),
    severity_percent: int = Form(..., ge=-100, le=100),
    area_percent: int = Form(..., ge=-100, le=100),
    output_quality: Literal["low", "medium", "high", "auto"] = Form("medium"),
    user_instructions: str = Form(""),
):
    severity_percent = clamp_percentage(severity_percent)
    area_percent = clamp_percentage(area_percent)

    source = await read_image(image, "RGB")
    source_mask = await read_image(mask, "L")
    source_mask = resize_mask(source_mask, source.size)

    job_id = str(uuid.uuid4())
    selection = adjust_selection_mask(source_mask, area_percent)

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return JSONResponse(
            make_mock_result(
                source,
                selection,
                job_id,
                severity_percent,
                area_percent,
            )
        )

    result_bytes = edit_manual_full_image_locked(
        source,
        source_mask,
        severity_percent,
        area_percent,
        user_instructions,
        output_quality,
    )

    result_path = OUTPUT_DIR / f"{job_id}.jpg"
    result_path.write_bytes(result_bytes)

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "selection_mode": "manual",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "user_instructions": user_instructions.strip(),
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v7-full-image-composite-locked",
    }


@app.post("/v1/damage/edit-base64")
def edit_damage_base64(payload: DamageEditBase64Request):
    severity_percent = clamp_percentage(payload.severity_percent)
    area_percent = clamp_percentage(payload.area_percent)

    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    )

    source_mask = decode_base64_image(
        payload.mask_base64,
        "mask_base64",
        "L",
    )
    source_mask = resize_mask(source_mask, source.size)

    job_id = str(uuid.uuid4())
    selection = adjust_selection_mask(source_mask, area_percent)

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        result = make_mock_result(
            source,
            selection,
            job_id,
            severity_percent,
            area_percent,
        )
        result["user_instructions"] = payload.user_instructions.strip()
        return result

    result_bytes = edit_manual_full_image_locked(
        source,
        source_mask,
        severity_percent,
        area_percent,
        payload.user_instructions,
        payload.output_quality,
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "selection_mode": "manual",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "user_instructions": payload.user_instructions.strip(),
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v7-full-image-composite-locked",
    }
