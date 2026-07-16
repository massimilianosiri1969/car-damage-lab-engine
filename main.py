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
from PIL import Image, ImageFile, ImageFilter
from pydantic import BaseModel, Field

ImageFile.LOAD_TRUNCATED_IMAGES = True

APP_NAME = "Car Damage Lab API"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bordo di fusione interno alla selezione, espresso in pixel dell'immagine finale.
# Può essere modificato su Render con la variabile SOFT_COMPOSITE_FEATHER_PX.
SOFT_COMPOSITE_FEATHER_PX = max(
    1,
    int(os.getenv("SOFT_COMPOSITE_FEATHER_PX", "25")),
)

# Ritaglio locale attorno alla maschera.
LOCAL_CROP_CONTEXT_MARGIN = max(
    0.0,
    float(os.getenv("LOCAL_CROP_CONTEXT_MARGIN", "0.30")),
)
LOCAL_CROP_MIN_SIZE = max(
    128,
    int(os.getenv("LOCAL_CROP_MIN_SIZE", "512")),
)

ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]

app = FastAPI(
    title=APP_NAME,
    version="0.4.0",
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
    mask_base64: str = Field(..., min_length=16)
    severity_percent: int = Field(..., ge=-100, le=100)
    area_percent: int = Field(..., ge=-100, le=100)
    output_quality: Literal["low", "medium", "high", "auto"] = "medium"
    user_instructions: str = Field(default="", max_length=500)


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
) -> str:
    cleaned = user_instructions.strip()
    extra_instruction = ""

    if cleaned:
        extra_instruction = f"""
Additional user instruction:
{cleaned}

Apply it only inside the editable transparent portion of the mask and only
when compatible with all preservation rules below.
""".strip()

    severity_direction = (
        "increase"
        if severity > 0
        else "reduce"
        if severity < 0
        else "preserve"
    )
    area_direction = (
        "make the visible deformation slightly broader"
        if area > 0
        else "make the visible deformation slightly narrower"
        if area < 0
        else "preserve the visible damage footprint"
    )

    return f"""
You are editing a LOCAL CROP from a real automotive photograph.

The transparent portion of the mask identifies an existing painted
sheet-metal panel and is the only editable region.

Requested adjustment:
- {severity_direction} the damage severity by approximately {abs(severity)}%;
- {area_direction} by approximately {abs(area)}%.

{extra_instruction}

Create a physically plausible automotive collision deformation.
The surface must still look like stamped, painted automotive steel.

Do not create melted metal, liquid-looking surfaces, plastic-looking folds,
decorative wrinkles, transparent areas, duplicated panels, pasted shapes,
detached overlays, replacement panels or new objects.

Preserve exactly:
- every headlight, tail light, lens, reflector, lamp and optical assembly;
- hood, bumper, wheel and tyre;
- panel gaps and body seams;
- vehicle identity and paint colour;
- camera angle, lighting and workshop background;
- everything outside the editable transparent mask.

Do not deform, redraw, move or change the transparency, shape or internal
content of any optical component, even when it is close to the selected area.

The dent must remain continuous with the surrounding fender or body panel.
Reflections must follow the new geometry naturally.
Do not replace the existing panel with another object.
Return one photorealistic edited image of the same crop.
""".strip()


def pil_to_file(image: Image.Image, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def _open_image_bytes(raw: bytes, mode: str) -> Image.Image | None:
    """
    Decodifica tollerante per JPEG/PNG/WebP.

    Alcuni JPEG prodotti o inoltrati dal browser sono visualizzabili da Chrome
    ma risultano formalmente troncati per Pillow/OpenCV. In quel caso:
    - rimuove eventuali byte dopo il marker JPEG EOI;
    - aggiunge il marker EOI se manca;
    - usa Pillow con LOAD_TRUNCATED_IMAGES;
    - prova infine OpenCV.
    """
    if not raw:
        return None

    candidates: list[bytes] = [raw]

    # JPEG: prova a normalizzare la terminazione del file.
    if raw.startswith(b"\xff\xd8\xff"):
        last_eoi = raw.rfind(b"\xff\xd9")

        if last_eoi >= 0 and last_eoi + 2 < len(raw):
            candidates.append(raw[: last_eoi + 2])
        elif last_eoi < 0:
            candidates.append(raw + b"\xff\xd9")

    seen: set[bytes] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        try:
            image = Image.open(io.BytesIO(candidate))
            image.load()

            if not image.width or not image.height:
                continue

            return image.convert(mode)
        except Exception:
            pass

        try:
            parser = ImageFile.Parser()
            parser.feed(candidate)
            image = parser.close()

            if image and image.width and image.height:
                return image.convert(mode)
        except Exception:
            pass

        try:
            array = np.frombuffer(candidate, dtype=np.uint8)
            decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

            if decoded is None:
                continue

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
            pass

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
        padded = data + ("=" * (-len(data) % 4))
        try:
            raw = base64.b64decode(padded, validate=True)
        except Exception:
            raw = base64.urlsafe_b64decode(padded)
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



def _adjust_binary_selection(
    source_mask: Image.Image,
    area_percent: int,
) -> np.ndarray:
    """
    Converte la maschera Base44 in una selezione binaria:
      255 = zona modificabile
      0   = zona protetta

    Applica anche l'espansione o contrazione richiesta da area_percent.
    """
    selection = np.array(source_mask.convert("L"))
    selection = np.where(selection >= 128, 255, 0).astype(np.uint8)

    if area_percent != 0:
        height, width = selection.shape
        reference = max(3, round(min(width, height) * 0.10))
        radius = max(1, round(reference * abs(area_percent) / 100))
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        if area_percent > 0:
            selection = cv2.dilate(selection, kernel, iterations=1)
        else:
            selection = cv2.erode(selection, kernel, iterations=1)

    return selection


def soft_composite_selected_area(
    source: Image.Image,
    generated_bytes: bytes,
    source_mask: Image.Image,
    area_percent: int,
    feather_radius: int = SOFT_COMPOSITE_FEATHER_PX,
) -> bytes:
    """
    Fonde il risultato AI con l'originale usando un bordo morbido INTERNO.

    - centro della selezione: risultato AI al 100%;
    - bordo interno: fusione graduale;
    - fuori selezione: fotografia originale al 100%.

    La sfumatura non si estende mai fuori dalla selezione, così fari,
    cofano, paraurti e altri elementi non selezionati restano protetti.
    """
    try:
        generated = Image.open(io.BytesIO(generated_bytes))
        generated.load()
        generated = generated.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il motore ha restituito un'immagine non valida.",
        ) from exc

    source_rgb = source.convert("RGB")

    if generated.size != source_rgb.size:
        generated = generated.resize(
            source_rgb.size,
            Image.Resampling.LANCZOS,
        )

    selection = _adjust_binary_selection(
        source_mask,
        area_percent,
    )

    selected_pixels = int((selection >= 128).sum())
    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera è vuota. Disegna la zona da modificare.",
        )

    # Sfuma il bordo verso l'interno, senza autorizzare modifiche all'esterno.
    binary_mask = Image.fromarray(selection, mode="L")
    blurred_mask = binary_mask.filter(
        ImageFilter.GaussianBlur(radius=max(1, feather_radius))
    )

    binary_array = np.array(binary_mask, dtype=np.float32) / 255.0
    blurred_array = np.array(blurred_mask, dtype=np.float32) / 255.0

    # Clip alla selezione originale: nessuna invasione oltre il bordo disegnato.
    inward_alpha = np.minimum(blurred_array, binary_array)

    # Mantiene pieno il centro e sfuma soltanto la fascia vicina al bordo.
    inward_alpha = np.clip(inward_alpha, 0.0, 1.0)
    blend_mask = Image.fromarray(
        np.round(inward_alpha * 255).astype(np.uint8),
        mode="L",
    )

    final_image = Image.composite(
        generated,
        source_rgb,
        blend_mask,
    )

    output = io.BytesIO()
    final_image.save(
        output,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )
    return output.getvalue()



def _original_binary_selection(source_mask: Image.Image) -> np.ndarray:
    """
    Selezione originale senza dilatazione:
      255 = modificabile
      0   = protetto

    In v8 area_percent influenza il prompt, ma non allarga fisicamente
    la zona autorizzata. Questo protegge meglio fari e componenti vicini.
    """
    selection = np.array(source_mask.convert("L"))
    return np.where(selection >= 128, 255, 0).astype(np.uint8)


def _selection_bbox(selection: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(selection)

    if points is None:
        raise HTTPException(
            status_code=422,
            detail="La maschera è vuota. Disegna la zona da modificare.",
        )

    return cv2.boundingRect(points)


def _square_crop_box(
    image_size: tuple[int, int],
    selection_bbox: tuple[int, int, int, int],
    context_margin: float = LOCAL_CROP_CONTEXT_MARGIN,
    minimum_size: int = LOCAL_CROP_MIN_SIZE,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = selection_bbox

    center_x = x + width / 2
    center_y = y + height / 2

    requested_size = round(
        max(width, height) * (1 + context_margin * 2)
    )
    crop_size = max(requested_size, minimum_size)
    crop_size = min(crop_size, image_width, image_height)
    crop_size = max(1, crop_size)

    left = round(center_x - crop_size / 2)
    top = round(center_y - crop_size / 2)

    left = max(0, min(left, image_width - crop_size))
    top = max(0, min(top, image_height - crop_size))

    return (
        left,
        top,
        left + crop_size,
        top + crop_size,
    )


def prepare_local_crop(
    source: Image.Image,
    source_mask: Image.Image,
) -> tuple[
    Image.Image,
    Image.Image,
    np.ndarray,
    tuple[int, int, int, int],
]:
    source_rgb = source.convert("RGB")
    full_selection = _original_binary_selection(source_mask)

    bbox = _selection_bbox(full_selection)
    crop_box = _square_crop_box(source_rgb.size, bbox)

    left, top, right, bottom = crop_box

    source_crop = source_rgb.crop(crop_box)
    selection_crop = Image.fromarray(
        full_selection[top:bottom, left:right],
        mode="L",
    )

    # Nessuna espansione geometrica in v8.
    api_mask_crop = alter_damage_area(selection_crop, 0)

    return (
        source_crop,
        api_mask_crop,
        full_selection,
        crop_box,
    )


def soft_composite_local_crop(
    source: Image.Image,
    generated_crop_bytes: bytes,
    full_selection: np.ndarray,
    crop_box: tuple[int, int, int, int],
    feather_radius: int = SOFT_COMPOSITE_FEATHER_PX,
) -> bytes:
    try:
        generated_crop = Image.open(io.BytesIO(generated_crop_bytes))
        generated_crop.load()
        generated_crop = generated_crop.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il motore ha restituito un ritaglio non valido.",
        ) from exc

    source_rgb = source.convert("RGB")
    left, top, right, bottom = crop_box
    crop_size = (right - left, bottom - top)

    if generated_crop.size != crop_size:
        generated_crop = generated_crop.resize(
            crop_size,
            Image.Resampling.LANCZOS,
        )

    generated_full = source_rgb.copy()
    generated_full.paste(generated_crop, (left, top))

    binary_mask = Image.fromarray(
        np.where(full_selection >= 128, 255, 0).astype(np.uint8),
        mode="L",
    )
    blurred_mask = binary_mask.filter(
        ImageFilter.GaussianBlur(
            radius=max(1, feather_radius),
        )
    )

    binary_array = np.array(binary_mask, dtype=np.float32) / 255.0
    blurred_array = np.array(blurred_mask, dtype=np.float32) / 255.0

    # Feather solo verso l'interno: mai oltre la pennellata originale.
    inward_alpha = np.minimum(blurred_array, binary_array)

    blend_mask = Image.fromarray(
        np.round(
            np.clip(inward_alpha, 0.0, 1.0) * 255
        ).astype(np.uint8),
        mode="L",
    )

    final_image = Image.composite(
        generated_full,
        source_rgb,
        blend_mask,
    )

    output = io.BytesIO()
    final_image.save(
        output,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )
    return output.getvalue()

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
            detail="OPENAI_API_KEY non configurata. Usa MOCK_MODE=true per i test.",
        )

    client = OpenAI(api_key=api_key)
    source_file = pil_to_file(source, "source.png")
    mask_file = pil_to_file(api_mask, "mask.png")

    try:
        response = client.images.edit(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
            image=source_file,
            mask=mask_file,
            prompt=prompt,
            quality=quality,
            size="auto",
            output_format="jpeg",
            output_compression=92,
            n=1,
        )

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

    (
        source_crop,
        api_mask_crop,
        full_selection,
        crop_box,
    ) = prepare_local_crop(
        source,
        source_mask,
    )

    job_id = str(uuid.uuid4())
    prompt = build_prompt(
        severity_percent,
        area_percent,
        "",
    )

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return JSONResponse(
            make_mock_result(
                source,
                api_mask_crop,
                job_id,
                severity_percent,
                area_percent,
            )
        )

    if severity_percent == 0 and area_percent == 0:
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=95, subsampling=0)
        result_bytes = buffer.getvalue()
    else:
        generated_crop_bytes = call_openai_image_edit(
            source_crop,
            api_mask_crop,
            prompt,
            output_quality,
        )

        result_bytes = soft_composite_local_crop(
            source=source,
            generated_crop_bytes=generated_crop_bytes,
            full_selection=full_selection,
            crop_box=crop_box,
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
        "prompt_version": "damage-v8-local-crop-soft-composite",
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

    (
        source_crop,
        api_mask_crop,
        full_selection,
        crop_box,
    ) = prepare_local_crop(
        source,
        source_mask,
    )

    job_id = str(uuid.uuid4())
    prompt = build_prompt(
        severity_percent,
        area_percent,
        payload.user_instructions,
    )

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return make_mock_result(
            source,
            api_mask_crop,
            job_id,
            severity_percent,
            area_percent,
        )

    if (
        severity_percent == 0
        and area_percent == 0
        and not payload.user_instructions.strip()
    ):
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=95, subsampling=0)
        result_bytes = buffer.getvalue()
    else:
        generated_crop_bytes = call_openai_image_edit(
            source_crop,
            api_mask_crop,
            prompt,
            payload.output_quality,
        )

        result_bytes = soft_composite_local_crop(
            source=source,
            generated_crop_bytes=generated_crop_bytes,
            full_selection=full_selection,
            crop_box=crop_box,
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v8-local-crop-soft-composite",
    }
