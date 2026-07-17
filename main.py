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
from PIL import Image, ImageEnhance, ImageFile, ImageFilter
from pydantic import BaseModel, Field

ImageFile.LOAD_TRUNCATED_IMAGES = True

APP_NAME = "Car Damage Lab API"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# V9: fotografia completa al modello, fusione protetta attorno alla pennellata.
COMPOSITE_EXPANSION_PX = max(
    0,
    int(os.getenv("COMPOSITE_EXPANSION_PX", "15")),
)
SOFT_COMPOSITE_FEATHER_PX = max(
    1,
    int(os.getenv("SOFT_COMPOSITE_FEATHER_PX", "12")),
)

ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
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
    protect_mask_base64: str | None = None

    severity_percent: int = Field(..., ge=-100, le=100)
    area_percent: int = Field(..., ge=-100, le=100)
    output_quality: Literal["low", "medium", "high", "auto"] = "medium"

    deformation_type: Literal[
        "dent",
        "crease",
        "crush",
        "sideswipe",
        "multiple",
    ] = "dent"

    impact_direction: Literal[
        "left_to_right",
        "right_to_left",
        "top_to_bottom",
        "bottom_to_top",
        "diagonal_right",
        "diagonal_left",
        "frontal",
    ] = "frontal"

    involved_components: dict[str, bool] = Field(
        default_factory=lambda: {"bodyPanel": True}
    )

    hood_damage_type: str | None = None
    headlight_damage_type: str | None = None
    bumper_damage_type: str | None = None
    wheel_damage_type: str | None = None
    glass_damage_type: str | None = None

    contact_traces_enabled: bool = False
    contact_trace_type: str | None = None
    contact_vehicle_color: str | None = None
    contact_trace_intensity: str | None = None
    contact_trace_direction: str | None = None

    # Compatibilità temporanea con vecchie versioni Base44.
    user_instructions: str = Field(default="", max_length=500)


class DamageFinalizeBase64Request(BaseModel):
    simulation_image_base64: str = Field(..., min_length=16)
    original_image_base64: str | None = None
    edit_mask_base64: str | None = None
    protect_mask_base64: str | None = None
    preserve_geometry: bool = True
    output_quality: Literal["high", "medium"] = "high"
    source_simulation_id: str | None = None


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


def mask_to_binary(mask: Image.Image) -> np.ndarray:
    gray = np.array(mask.convert("L"))
    return np.where(gray >= 128, 255, 0).astype(np.uint8)


def combine_edit_and_protect_masks(
    edit_mask: Image.Image,
    protect_mask: Image.Image | None,
    target_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image | None]:
    edit_mask = resize_mask(edit_mask.convert("L"), target_size)
    edit_binary = mask_to_binary(edit_mask)

    resized_protect: Image.Image | None = None

    if protect_mask is not None:
        resized_protect = resize_mask(protect_mask.convert("L"), target_size)
        protect_binary = mask_to_binary(resized_protect)
        edit_binary = cv2.bitwise_and(
            edit_binary,
            cv2.bitwise_not(protect_binary),
        )

    if int((edit_binary >= 128).sum()) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "La maschera effettiva è vuota. La protezione non può coprire "
                "interamente l'area da modificare."
            ),
        )

    return Image.fromarray(edit_binary, mode="L"), resized_protect


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


DEFORMATION_INSTRUCTIONS = {
    "dent": (
        "Create a localized inward dent with realistic depth and continuous "
        "panel curvature."
    ),
    "crease": (
        "Create one dominant collision crease with limited secondary deformation."
    ),
    "crush": (
        "Create localized panel compression and loss of volume, preserving "
        "recognizable panel boundaries."
    ),
    "sideswipe": (
        "Create a shallow deformation that develops along the impact direction."
    ),
    "multiple": (
        "Create multiple related impact deformations generated by the same event, "
        "without decorative or random wrinkling."
    ),
}

IMPACT_DIRECTION_INSTRUCTIONS = {
    "left_to_right": "The impact force travels from left to right.",
    "right_to_left": "The impact force travels from right to left.",
    "top_to_bottom": "The impact force travels from top to bottom.",
    "bottom_to_top": "The impact force travels from bottom to top.",
    "diagonal_right": "The impact force travels diagonally toward the right.",
    "diagonal_left": "The impact force travels diagonally toward the left.",
    "frontal": (
        "The impact is frontal and localized around the main contact point."
    ),
}

COMPONENT_LABELS = {
    "bodyPanel": "selected painted body panel",
    "body_panel": "selected painted body panel",
    "hood": "hood",
    "headlight": "headlight assembly",
    "bumper": "bumper",
    "wheelArch": "wheel arch",
    "wheel_arch": "wheel arch",
    "wheel": "wheel and tyre",
    "glass": "glass",
}

COMPONENT_DAMAGE_TEXT = {
    "hood": {
        "dented": "dent the hood locally",
        "edge_bent": "bend the hood edge nearest the impact",
        "lifted": "lift the hood edge realistically",
        "misaligned": "misalign the hood along the affected seam",
        "severely_deformed": "severely deform the selected portion of the hood",
    },
    "headlight": {
        "scratched": "scratch the headlight lens",
        "cracked": "create realistic cracks in the headlight lens",
        "broken": "break the headlight lens realistically",
        "partially_missing": "make part of the headlight lens or housing missing",
        "detached": "partially detach the headlight assembly",
    },
    "bumper": {
        "deformed": "deform the bumper locally",
        "scratched": "scratch and abrade the bumper surface",
        "unclipped": "unclip the bumper near the impact",
        "partially_detached": "partially detach the bumper",
        "hanging": "make the bumper hang realistically from the affected side",
    },
    "wheel": {
        "rim_scratched": "scratch the wheel rim",
        "rim_bent": "bend the wheel rim realistically",
        "tyre_damaged": "damage the tyre visibly but plausibly",
        "wheel_misaligned": "misalign the wheel in a physically plausible way",
    },
    "glass": {
        "scratched": "scratch the selected glass",
        "cracked": "create realistic cracks in the selected glass",
        "broken": "break the selected glass",
        "shattered": "shatter the selected glass realistically",
    },
}


def component_is_enabled(components: dict[str, bool], *names: str) -> bool:
    return any(bool(components.get(name, False)) for name in names)


def build_component_instructions(payload: DamageEditBase64Request) -> str:
    components = payload.involved_components or {"bodyPanel": True}
    enabled_labels = [
        label
        for key, label in COMPONENT_LABELS.items()
        if bool(components.get(key, False))
    ]

    lines: list[str] = []

    if enabled_labels:
        lines.append(
            "Components explicitly involved: " + ", ".join(sorted(set(enabled_labels))) + "."
        )

    damage_values = {
        "hood": payload.hood_damage_type,
        "headlight": payload.headlight_damage_type,
        "bumper": payload.bumper_damage_type,
        "wheel": payload.wheel_damage_type,
        "glass": payload.glass_damage_type,
    }

    aliases = {
        "hood": ("hood",),
        "headlight": ("headlight",),
        "bumper": ("bumper",),
        "wheel": ("wheel",),
        "glass": ("glass",),
    }

    for component, damage_type in damage_values.items():
        if component_is_enabled(components, *aliases[component]) and damage_type:
            instruction = COMPONENT_DAMAGE_TEXT.get(component, {}).get(damage_type)
            if instruction:
                lines.append(f"For the {component}: {instruction}.")

    protected_components: list[str] = []

    if not component_is_enabled(components, "hood"):
        protected_components.append("hood")
    if not component_is_enabled(components, "headlight"):
        protected_components.append(
            "headlight, lens, reflector, bulbs and optical assembly"
        )
    if not component_is_enabled(components, "bumper"):
        protected_components.append("bumper")
    if not component_is_enabled(components, "wheel", "wheelArch", "wheel_arch"):
        protected_components.append("wheel, tyre and wheel arch")
    if not component_is_enabled(components, "glass"):
        protected_components.append("glass")

    if protected_components:
        lines.append(
            "Preserve exactly and do not alter: "
            + ", ".join(protected_components)
            + "."
        )

    return "\n".join(lines)


def build_contact_trace_instructions(
    payload: DamageEditBase64Request,
) -> str:
    if not payload.contact_traces_enabled:
        return (
            "Do not add scratches, abrasions, paint transfer or other contact marks."
        )

    trace_type_map = {
        "scratches": "directional scratches",
        "abrasions": "surface abrasions",
        "paint_transfer": "paint transfer from the other vehicle",
        "scratches_and_transfer": "directional scratches and paint transfer",
        "full_contact_marks": (
            "directional scratches, abrasions and paint transfer"
        ),
    }
    intensity_map = {
        "light": "subtle",
        "medium": "moderate",
        "strong": "clearly visible but physically plausible",
    }
    direction_map = {
        "same_as_impact": "following the impact direction",
        "horizontal": "predominantly horizontal",
        "vertical": "predominantly vertical",
        "diagonal": "predominantly diagonal",
        "irregular": "irregular but coherent with the collision",
    }

    trace_type = trace_type_map.get(
        payload.contact_trace_type or "",
        "realistic contact marks",
    )
    intensity = intensity_map.get(
        payload.contact_trace_intensity or "",
        "moderate",
    )
    direction = direction_map.get(
        payload.contact_trace_direction or "",
        "following the impact direction",
    )
    color = payload.contact_vehicle_color or "unspecified"

    return (
        f"Add {intensity} {trace_type}, {direction}, confined to the editable "
        f"region. Use plausible transferred paint from a {color} vehicle when "
        "paint transfer is requested. Contact marks must remain surface effects "
        "and must not look like newly painted shapes. Do not add rust, cuts or "
        "missing metal unless explicitly requested by a selected component."
    )


def build_prompt(
    severity: int,
    area: int,
    deformation_type: str = "dent",
    impact_direction: str = "frontal",
    component_instructions: str = "",
    contact_trace_instructions: str = "",
) -> str:
    severity_text = severity_instruction(severity)
    area_text = area_instruction(area)
    deformation_text = DEFORMATION_INSTRUCTIONS.get(
        deformation_type,
        DEFORMATION_INSTRUCTIONS["dent"],
    )
    direction_text = IMPACT_DIRECTION_INSTRUCTIONS.get(
        impact_direction,
        IMPACT_DIRECTION_INSTRUCTIONS["frontal"],
    )

    return f"""
Realistic automotive photo editing.

Objective:
- {severity_text}
- {area_text}
- {deformation_text}
- {direction_text}

Component rules:
{component_instructions or "Modify only the selected painted body panel."}

Contact traces:
{contact_trace_instructions or "Do not add contact traces."}

General constraints:
- preserve the same vehicle, model, colour, perspective and framing;
- modify only areas permitted by the editable mask;
- the protected mask has absolute priority and must remain pixel-consistent
  with the source photograph;
- create physically plausible automotive collision damage;
- keep paint, shadows and reflections coherent with the new geometry;
- do not create melted metal, liquid surfaces, decorative wrinkles,
  duplicated panels, pasted objects or transparent bodywork;
- preserve all non-involved components and everything outside the editable mask;
- return the complete final photograph, photorealistic and without watermark.
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



def protected_soft_composite(
    source: Image.Image,
    generated_bytes: bytes,
    source_mask: Image.Image,
    protect_mask: Image.Image | None = None,
    expansion_px: int = COMPOSITE_EXPANSION_PX,
    feather_px: int = SOFT_COMPOSITE_FEATHER_PX,
) -> bytes:
    """
    V9:
    - il modello riceve la fotografia completa;
    - la fusione finale usa la pennellata originale leggermente allargata;
    - la sfumatura resta interna alla zona allargata;
    - tutto ciò che è lontano dalla selezione resta identico all'originale.

    area_percent non modifica geometricamente questa maschera: influenza
    soltanto il prompt. In questo modo i fari vicini sono più protetti.
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

    original_selection = mask_to_binary(source_mask)
    protect_selection = (
        mask_to_binary(protect_mask)
        if protect_mask is not None
        else np.zeros_like(original_selection)
    )

    selected_pixels = int((original_selection >= 128).sum())
    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera è vuota. Disegna la zona da modificare.",
        )

    protected_selection = original_selection.copy()

    if expansion_px > 0:
        kernel_size = expansion_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        protected_selection = cv2.dilate(
            protected_selection,
            kernel,
            iterations=1,
        )

    # La maschera Proteggi area ha priorità anche dopo l'espansione.
    protected_selection = cv2.bitwise_and(
        protected_selection,
        cv2.bitwise_not(protect_selection),
    )

    binary_mask = Image.fromarray(
        protected_selection,
        mode="L",
    )

    blurred_mask = binary_mask.filter(
        ImageFilter.GaussianBlur(
            radius=max(1, feather_px),
        )
    )

    binary_array = np.array(
        binary_mask,
        dtype=np.float32,
    ) / 255.0

    blurred_array = np.array(
        blurred_mask,
        dtype=np.float32,
    ) / 255.0

    # Feather soltanto verso l'interno della zona protetta allargata.
    inward_alpha = np.minimum(
        blurred_array,
        binary_array,
    )

    blend_mask = Image.fromarray(
        np.round(
            np.clip(inward_alpha, 0.0, 1.0) * 255
        ).astype(np.uint8),
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
    api_mask = alter_damage_area(source_mask, 0)

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
                api_mask,
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
        generated_bytes = call_openai_image_edit(
            source,
            api_mask,
            prompt,
            output_quality,
        )

        result_bytes = protected_soft_composite(
            source=source,
            generated_bytes=generated_bytes,
            source_mask=source_mask,
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
        "prompt_version": "damage-v13-structured-workflow",
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
    raw_edit_mask = decode_base64_image(
        payload.mask_base64,
        "mask_base64",
        "L",
    )

    raw_protect_mask = (
        decode_base64_image(
            payload.protect_mask_base64,
            "protect_mask_base64",
            "L",
        )
        if payload.protect_mask_base64
        else None
    )

    source_mask, protect_mask = combine_edit_and_protect_masks(
        raw_edit_mask,
        raw_protect_mask,
        source.size,
    )
    api_mask = alter_damage_area(source_mask, 0)

    job_id = str(uuid.uuid4())
    prompt = build_prompt(
        severity=severity_percent,
        area=area_percent,
        deformation_type=payload.deformation_type,
        impact_direction=payload.impact_direction,
        component_instructions=build_component_instructions(payload),
        contact_trace_instructions=build_contact_trace_instructions(payload),
    )

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return make_mock_result(
            source,
            api_mask,
            job_id,
            severity_percent,
            area_percent,
        )

    if (
        severity_percent == 0
        and area_percent == 0
        and payload.deformation_type == "dent"
        and not payload.contact_traces_enabled
    ):
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=95, subsampling=0)
        result_bytes = buffer.getvalue()
    else:
        generated_bytes = call_openai_image_edit(
            source,
            api_mask,
            prompt,
            payload.output_quality,
        )

        result_bytes = protected_soft_composite(
            source=source,
            generated_bytes=generated_bytes,
            source_mask=source_mask,
            protect_mask=protect_mask,
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v13-structured-workflow",
        "deformation_type": payload.deformation_type,
        "impact_direction": payload.impact_direction,
        "contact_traces_enabled": payload.contact_traces_enabled,
        "involved_components": payload.involved_components,
    }


def finalize_without_regeneration(
    simulation: Image.Image,
    max_dimension: int = 4096,
) -> tuple[bytes, tuple[int, int]]:
    """
    Finalizzazione non generativa:
    - non ricrea il danno;
    - non cambia pieghe, forma, posizione o componenti;
    - aumenta la risoluzione fino a 2x, rispettando max_dimension;
    - applica una rifinitura leggera e controllata.

    Questa modalità è preferibile quando preserve_geometry=True.
    """
    image = simulation.convert("RGB")
    width, height = image.size

    scale = min(
        2.0,
        max_dimension / max(width, height),
    )
    scale = max(1.0, scale)

    target_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    if target_size != image.size:
        image = image.resize(
            target_size,
            Image.Resampling.LANCZOS,
        )

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.15,
            percent=70,
            threshold=3,
        )
    )
    image = ImageEnhance.Contrast(image).enhance(1.015)
    image = ImageEnhance.Sharpness(image).enhance(1.05)

    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=97,
        subsampling=0,
        optimize=True,
    )
    return output.getvalue(), image.size


@app.post("/v1/damage/finalize-base64")
def finalize_damage_base64(payload: DamageFinalizeBase64Request):
    simulation = decode_base64_image(
        payload.simulation_image_base64,
        "simulation_image_base64",
        "RGB",
    )

    if not payload.preserve_geometry:
        raise HTTPException(
            status_code=422,
            detail=(
                "Questa versione supporta solo preserve_geometry=true. "
                "La finalizzazione non deve rigenerare il danno."
            ),
        )

    result_bytes, result_size = finalize_without_regeneration(simulation)
    job_id = str(uuid.uuid4())

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "non_generative_finalization",
        "source_simulation_id": payload.source_simulation_id,
        "preserve_geometry": True,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "width": result_size[0],
        "height": result_size[1],
        "prompt_version": "damage-finalize-v1-geometry-locked",
        "note": (
            "Finalizzazione non generativa: geometria del danno preservata."
        ),
    }
