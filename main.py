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
from PIL import Image
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
    version="0.2.0",
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


class DamageEditBase64Request(BaseModel):
    image_base64: str = Field(..., min_length=16)
    mask_base64: str = Field(default="")
    severity_percent: int = Field(..., ge=-100, le=100)
    area_percent: int = Field(..., ge=-100, le=100)
    output_quality: Literal["low", "medium", "high", "auto"] = "medium"
    user_instructions: str = Field(default="", max_length=500)
    selection_mode: Literal["manual", "descriptive"] = "manual"


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


def alter_damage_area(mask: Image.Image, area_percent: int) -> Image.Image:
    """
    Maschera ricevuta da Base44:
      bianco = zona modificabile
      nero   = zona protetta

    Maschera inviata all'API OpenAI:
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
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
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


def validate_manual_mask(mask: Image.Image) -> dict:
    """Verifica che la maschera manuale sia realmente localizzata."""
    gray = np.array(mask.convert("L"))
    selected = gray >= 128
    selected_pixels = int(selected.sum())
    total_pixels = int(selected.size)

    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera manuale è vuota. Disegna la zona da modificare.",
        )

    coverage = selected_pixels / total_pixels

    # Una selezione manuale enorme equivale di fatto a rigenerare l'intera foto.
    # Blocchiamo il caso per evitare che il modello sostituisca automobile e scena.
    if coverage > 0.55:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La maschera copre il {coverage * 100:.1f}% della fotografia. "
                "Riduci la selezione: in modalità manuale deve restare localizzata."
            ),
        )

    ys, xs = np.where(selected)
    bbox = {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }

    return {
        "coverage": coverage,
        "bbox": bbox,
    }


def output_size_for(source: Image.Image) -> str:
    """Return only sizes supported by the OpenAI Image API."""
    width, height = source.size
    ratio = width / height if height else 1.0

    if ratio > 1.15:
        return "1536x1024"
    if ratio < 0.87:
        return "1024x1536"
    return "1024x1024"

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


def build_prompt(
    severity: int,
    area: int,
    user_instructions: str = "",
    selection_mode: Literal["manual", "descriptive"] = "manual",
) -> str:
    cleaned_instructions = user_instructions.strip()

    if selection_mode == "descriptive":
        mode_instruction = """
Modalità MODIFICA DESCRITTIVA SENZA MASCHERA MANUALE.
La richiesta testuale identifica la modifica da eseguire e la zona interessata.
Individua autonomamente nella fotografia il danno esistente e gli eventuali
pannelli o componenti citati dall'utente. Modifica soltanto il danno e la sua
estensione fisicamente collegata. Tutti gli altri pixel devono restare il più
possibile identici alla fotografia di partenza.
""".strip()
    else:
        mode_instruction = """
Modalità SELEZIONE MANUALE.
Intervieni esclusivamente nei pixel trasparenti indicati dalla maschera. La maschera ha
priorità assoluta su qualsiasi indicazione testuale. Non reinterpretare né rigenerare
le parti esterne alla maschera: devono restare visivamente identiche all'originale.
""".strip()

    extra_instruction = ""
    if cleaned_instructions:
        extra_instruction = f"""
Indicazione specifica dell'utente:
{cleaned_instructions}

Questa indicazione deve influenzare concretamente la modifica, insieme ai valori
di gravità e superficie.
""".strip()

    return f"""
Modifica fotografica automotive realistica e documentale.

{mode_instruction}

Obiettivo quantitativo:
- {severity_instruction(severity)}
- {area_instruction(area)}

{extra_instruction}

VINCOLI ASSOLUTI DI CONSERVAZIONE:
- usa la fotografia fornita come riferimento vincolante;
- conserva esattamente la stessa automobile, marca, modello e allestimento;
- conserva colore, targa, logo, cerchi, pneumatici, vetri, fanali e maniglie;
- conserva identici prospettiva, distanza, focale, inquadratura e rapporto d'aspetto;
- conserva identici sfondo, strada, edifici, ombre, riflessi e illuminazione;
- non spostare, ruotare, ingrandire o ridurre l'automobile;
- non generare un'altra automobile e non cambiare il punto di vista;
- non aggiungere persone, oggetti, testo, watermark o componenti estranei;
- modifica soltanto lamiera, vernice, graffi, rotture o deformazioni richieste;
- mantieni geometrie, materiali, ombre e riflessi fisicamente plausibili;
- restituisci l'intera fotografia, non un ritaglio e non un rendering.

Se la richiesta non può essere applicata senza alterare l'identità della scena,
applica una modifica più prudente e localizzata invece di rigenerare l'immagine.
""".strip()


def pil_to_file(image: Image.Image, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


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
    """Accetta Data URL, Base64, Base64 doppio e stringhe HEX."""
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
    api_mask: Image.Image,
    job_id: str,
    severity_percent: int,
    area_percent: int,
) -> dict:
    preview = source.copy()
    overlay = Image.new("RGBA", source.size, (255, 0, 0, 0))
    editable = 255 - np.array(api_mask.getchannel("A"))
    overlay_alpha = Image.fromarray(
        (editable * 0.25).astype(np.uint8),
        mode="L",
    )
    overlay.putalpha(overlay_alpha)
    preview = Image.alpha_composite(
        preview.convert("RGBA"),
        overlay,
    ).convert("RGB")

    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=92)
    result_bytes = buffer.getvalue()

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "mock",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "note": "Anteprima rossa della superficie elaborata; nessuna modifica AI.",
    }


def call_openai_image_edit(
    source: Image.Image,
    api_mask: Image.Image | None,
    prompt: str,
    quality: Literal["low", "medium", "high", "auto"],
) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY non configurata. Usa MOCK_MODE=true per i test.",
        )

    client = OpenAI(api_key=api_key)
    source_file = pil_to_file(source, "source.png")

    request_args = {
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "image": source_file,
        "prompt": prompt,
        "quality": quality,
        "size": output_size_for(source),
        "input_fidelity": "high",
        "background": "opaque",
        "output_format": "jpeg",
        "output_compression": 92,
        "n": 1,
    }

    if api_mask is not None:
        request_args["mask"] = pil_to_file(api_mask, "mask.png")

    try:
        response = client.images.edit(**request_args)

        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("OpenAI ha risposto senza dati immagine")

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
):
    severity_percent = clamp_percentage(severity_percent)
    area_percent = clamp_percentage(area_percent)

    source = await read_image(image, "RGB")
    source_mask = await read_image(mask, "L")
    source_mask = resize_mask(source_mask, source.size)
    api_mask = alter_damage_area(source_mask, area_percent)

    job_id = str(uuid.uuid4())
    prompt = build_prompt(severity_percent, area_percent)

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return JSONResponse(
            make_mock_result(
                source,
                api_mask,
                job_id,
                severity_percent,
                area_percent,
            )
        )

    result_bytes = call_openai_image_edit(
        source,
        api_mask,
        prompt,
        output_quality,
    )

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
        "prompt_version": "damage-v4-mask-locked",
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

    if payload.selection_mode == "descriptive":
        if not payload.user_instructions.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "In modalità descrittiva devi inserire una richiesta testuale "
                    "che specifichi la modifica desiderata."
                ),
            )
        api_mask = None
    else:
        if not payload.mask_base64.strip():
            raise HTTPException(
                status_code=422,
                detail="In modalità manuale è necessaria una maschera.",
            )

        source_mask = decode_base64_image(
            payload.mask_base64,
            "mask_base64",
            "L",
        )
        source_mask = resize_mask(source_mask, source.size)
        mask_info = validate_manual_mask(source_mask)
        print(
            "MANUAL MASK:",
            f"coverage={mask_info['coverage'] * 100:.2f}%",
            f"bbox={mask_info['bbox']}",
            flush=True,
        )
        api_mask = alter_damage_area(source_mask, area_percent)

    job_id = str(uuid.uuid4())
    prompt = build_prompt(
        severity_percent,
        area_percent,
        payload.user_instructions,
        payload.selection_mode,
    )

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        if api_mask is None:
            result = source.copy().convert("RGB")
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=92)
            result_bytes = buffer.getvalue()
            return {
                "job_id": job_id,
                "status": "completed",
                "mode": "mock",
                "selection_mode": payload.selection_mode,
                "severity_percent": severity_percent,
                "area_percent": area_percent,
                "user_instructions": payload.user_instructions.strip(),
                "result_base64": base64.b64encode(result_bytes).decode("ascii"),
                "mime_type": "image/jpeg",
                "note": "Mock descrittivo: fotografia restituita senza modifica AI.",
            }

        result = make_mock_result(
            source,
            api_mask,
            job_id,
            severity_percent,
            area_percent,
        )
        result["selection_mode"] = payload.selection_mode
        result["user_instructions"] = payload.user_instructions.strip()
        return result

    result_bytes = call_openai_image_edit(
        source,
        api_mask,
        prompt,
        payload.output_quality,
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "selection_mode": payload.selection_mode,
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "user_instructions": payload.user_instructions.strip(),
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v5-descriptive-no-mask",
    }

