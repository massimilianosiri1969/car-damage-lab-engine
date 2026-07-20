import base64
import io
import json
import os
import re
import traceback
import json
import time
import urllib.error
import urllib.request
import threading
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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
    version="1.6.2",
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

    # Maschere evolute, usate soprattutto in modalità mixed:
    # - bodywork_mask_base64: sola lamiera;
    # - component_masks_base64: una maschera per ogni componente.
    bodywork_mask_base64: str | None = None
    component_masks_base64: dict[str, str] | None = None

    severity_percent: int = Field(..., ge=-100, le=100)
    area_percent: int = Field(..., ge=-100, le=100)
    output_quality: Literal["low", "medium", "high", "auto"] = "medium"

    damage_mode: Literal[
        "auto",
        "bodywork",
        "component_only",
        "mixed",
    ] = "auto"

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

    # Nuovo formato dinamico usato da Base44.
    component_damage_types: dict[str, str] = Field(default_factory=dict)
    vehicle_view: str | None = None

    contact_traces_enabled: bool = False
    contact_trace_type: str | None = None
    contact_vehicle_color: str | None = None
    contact_trace_intensity: str | None = None
    contact_trace_direction: str | None = None

    # Compatibilità temporanea con vecchie versioni Base44.
    user_instructions: str = Field(default="", max_length=500)


class VehicleAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)


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


def apply_area_percent_to_edit_mask(
    mask: Image.Image,
    area_percent: int,
) -> Image.Image:
    """
    Applica geometricamente il parametro Superficie danneggiata.

    - valori negativi: restringono realmente la zona modificabile;
    - 0: mantengono la maschera originale;
    - valori positivi: espandono realmente la zona modificabile.

    Il risultato viene usato sia per la generazione sia per il compositing,
    così l'estensione non rimane una semplice istruzione testuale.
    """
    area_percent = clamp_percentage(area_percent)
    binary = mask_to_binary(mask)

    if area_percent == 0:
        return Image.fromarray(binary, mode="L")

    height, width = binary.shape
    reference = max(3, round(min(width, height) * 0.10))
    radius = max(1, round(reference * abs(area_percent) / 100))
    kernel_size = radius * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    if area_percent > 0:
        adjusted = cv2.dilate(binary, kernel, iterations=1)
    else:
        adjusted = cv2.erode(binary, kernel, iterations=1)

    if int((adjusted >= 128).sum()) == 0:
        # Non lasciamo che una forte erosione cancelli completamente la zona.
        adjusted = binary

    return Image.fromarray(adjusted, mode="L")


def area_transition_feather_px(
    image_size: tuple[int, int],
    area_percent: int,
) -> int:
    """
    Calcola una sfumatura più ampia per le maschere ristrette.

    Quando area_percent è negativo, l'erosione geometrica crea un bordo più netto.
    Questo feather aggiuntivo rende la transizione tra lamiera sana e deformata
    più progressiva, senza perdere il controllo reale dell'estensione.
    """
    area_percent = clamp_percentage(area_percent)
    width, height = image_size
    reference = min(width, height)

    base = SOFT_COMPOSITE_FEATHER_PX

    if area_percent >= 0:
        return base

    extra = round(reference * 0.012 * (abs(area_percent) / 100))
    return max(base, min(28, base + extra))


def area_transition_expansion_px(area_percent: int) -> int:
    """
    Mantiene il compositing leggermente più morbido dopo l'erosione,
    evitando contorni visivamente tagliati.
    """
    area_percent = clamp_percentage(area_percent)

    if area_percent >= 0:
        return COMPOSITE_EXPANSION_PX

    return max(COMPOSITE_EXPANSION_PX, 5)


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


VEHICLE_VIEW_LABELS = {
    "front": "Vista anteriore",
    "front_left": "Anteriore sinistra",
    "front_right": "Anteriore destra",
    "rear": "Vista posteriore",
    "rear_left": "Posteriore sinistra",
    "rear_right": "Posteriore destra",
    "left_side": "Fiancata sinistra",
    "right_side": "Fiancata destra",
    "mixed": "Vista mista",
    "unknown": "Vista non determinata",
}

VEHICLE_COMPONENT_CATALOG = {
    "front_fender": "Parafango anteriore",
    "rear_fender": "Parafango posteriore",
    "front_headlight": "Faro anteriore",
    "rear_light": "Fanale posteriore",
    "front_bumper": "Paraurti anteriore",
    "rear_bumper": "Paraurti posteriore",
    "hood": "Cofano anteriore",
    "tailgate": "Portellone posteriore",
    "front_door": "Porta anteriore",
    "rear_door": "Porta posteriore",
    "wheel_arch": "Passaruota",
    "wheel": "Ruota",
    "windshield": "Parabrezza",
    "rear_window": "Lunotto",
    "side_window": "Vetro laterale",
    "roof": "Tetto",
    "side_mirror": "Specchietto",
}


VEHICLE_COMPONENT_CATEGORIES = {
    "hood": "bodywork",
    "tailgate": "bodywork",
    "front_fender": "bodywork",
    "rear_fender": "bodywork",
    "front_door": "bodywork",
    "rear_door": "bodywork",
    "roof": "bodywork",
    "wheel_arch": "bodywork",
    "front_headlight": "light",
    "rear_light": "light",
    "front_bumper": "bumper",
    "rear_bumper": "bumper",
    "windshield": "glass",
    "rear_window": "glass",
    "side_window": "glass",
    "side_mirror": "mirror",
    "wheel": "wheel",
    "grille": "trim",
}

SEGMENTATION_POLYGON_SCALE = 1000
SEGMENTATION_MIN_AREA_RATIO = float(
    os.getenv("SEGMENTATION_MIN_AREA_RATIO", "0.0008")
)
SEGMENTATION_GRABCUT_ENABLED = (
    os.getenv("SEGMENTATION_GRABCUT_ENABLED", "true").lower() == "true"
)
SMART_COMPOSITE_ENABLED = (
    os.getenv("SMART_COMPOSITE_ENABLED", "true").lower() == "true"
)
SMART_COMPOSITE_PYRAMID_LEVELS = max(
    2,
    min(6, int(os.getenv("SMART_COMPOSITE_PYRAMID_LEVELS", "4"))),
)
SMART_COMPOSITE_COLOR_STRENGTH = max(
    0.0,
    min(1.0, float(os.getenv("SMART_COMPOSITE_COLOR_STRENGTH", "0.72"))),
)


ANALYSIS_MAX_SIDE = max(
    640,
    min(1600, int(os.getenv("ANALYSIS_MAX_SIDE", "1024"))),
)
SEGMENTATION_GRABCUT_MAX_SIDE = max(
    480,
    min(1280, int(os.getenv("SEGMENTATION_GRABCUT_MAX_SIDE", "768"))),
)
SMART_COMPOSITE_MAX_SIDE = max(
    720,
    min(2048, int(os.getenv("SMART_COMPOSITE_MAX_SIDE", "1400"))),
)


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
REPLICATE_SAM2_VERSION = os.getenv(
    "REPLICATE_SAM2_VERSION",
    "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
)


REPLICATE_PROMPTED_SAM2_VERSION = os.getenv(
    "REPLICATE_PROMPTED_SAM2_VERSION",
    "33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d",
)
PROMPTED_SAM_MAX_WORKERS = max(
    1,
    min(4, int(os.getenv("PROMPTED_SAM_MAX_WORKERS", "3"))),
)
PROMPTED_SAM_BOX_INSET_RATIO = max(
    0.05,
    min(0.35, float(os.getenv("PROMPTED_SAM_BOX_INSET_RATIO", "0.18"))),
)
PROMPTED_SAM_NEGATIVE_MARGIN_RATIO = max(
    0.02,
    min(0.30, float(os.getenv("PROMPTED_SAM_NEGATIVE_MARGIN_RATIO", "0.10"))),
)
PROMPTED_SAM_COMPONENT_LIMIT = max(
    1,
    min(20, int(os.getenv("PROMPTED_SAM_COMPONENT_LIMIT", "12"))),
)
REPLICATE_POLL_SECONDS = max(
    0.5,
    min(5.0, float(os.getenv("REPLICATE_POLL_SECONDS", "1.0"))),
)
REPLICATE_TIMEOUT_SECONDS = max(
    30,
    min(300, int(os.getenv("REPLICATE_TIMEOUT_SECONDS", "180"))),
)
SAM2_POINTS_PER_SIDE = max(
    16,
    min(64, int(os.getenv("SAM2_POINTS_PER_SIDE", "32"))),
)
SAM2_PRED_IOU_THRESH = max(
    0.50,
    min(0.99, float(os.getenv("SAM2_PRED_IOU_THRESH", "0.82"))),
)
SAM2_STABILITY_SCORE_THRESH = max(
    0.50,
    min(0.99, float(os.getenv("SAM2_STABILITY_SCORE_THRESH", "0.88"))),
)
SAM2_MIN_BOX_COVERAGE = max(
    0.05,
    min(0.95, float(os.getenv("SAM2_MIN_BOX_COVERAGE", "0.06"))),
)
SAM2_MAX_OUTSIDE_RATIO = max(
    0.05,
    min(0.98, float(os.getenv("SAM2_MAX_OUTSIDE_RATIO", "0.90"))),
)


ANALYSIS_JOB_TTL_SECONDS = max(
    300,
    min(86400, int(os.getenv("ANALYSIS_JOB_TTL_SECONDS", "3600"))),
)
ANALYSIS_JOB_MAX_COUNT = max(
    10,
    min(500, int(os.getenv("ANALYSIS_JOB_MAX_COUNT", "100"))),
)
ANALYSIS_JOB_DIR = Path(
    os.getenv("ANALYSIS_JOB_DIR", "/tmp/car-damage-lab-jobs")
)
ANALYSIS_JOB_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_JOBS: dict[str, dict] = {}
ANALYSIS_JOBS_LOCK = threading.Lock()

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

DYNAMIC_COMPONENT_DAMAGE_TEXT = {
    "front_fender": {
        "dented": "dent the front fender",
        "creased": "create a realistic crease in the front fender",
        "crushed": "crush the selected portion of the front fender",
        "scratched": "scratch the front fender",
        "torn": "tear the selected edge of the front fender realistically",
    },
    "rear_fender": {
        "dented": "dent the rear fender",
        "creased": "create a realistic crease in the rear fender",
        "crushed": "crush the selected portion of the rear fender",
        "scratched": "scratch the rear fender",
        "torn": "tear the selected edge of the rear fender realistically",
    },
    "front_headlight": {
        "scratched": (
            "add light superficial scratches to the selected front headlight lens "
            "while preserving transparency, reflectors and the overall assembly"
        ),
        "cracked": (
            "create exactly one main crack in the selected front headlight lens, "
            "with no more than two or three short, thin secondary branches. Keep at "
            "least 70 to 80 percent of the lens perfectly intact, clear and optically "
            "unchanged. Do not add missing fragments, cloudy areas, widespread "
            "distortion, dense crack networks, spiderweb patterns or shattered-glass "
            "effects. Preserve the original reflector details and overall lamp shape"
        ),
        "broken": (
            "break the selected front headlight lens in a clearly visible but still "
            "localized way. Show several intersecting cracks and one or two small "
            "missing fragments, while keeping the housing, reflector geometry and "
            "most of the optical assembly recognizable. Do not turn the entire lamp "
            "into uniformly shattered glass"
        ),
        "shattered": (
            "heavily shatter most of the selected front headlight lens with dense "
            "crack networks, multiple missing fragments and exposed internal optical "
            "elements, while preserving the overall lamp housing position"
        ),
        "partially_missing": (
            "remove a limited portion of the selected front headlight lens, with "
            "realistic broken edges and visible internal optical elements, while "
            "preserving the rest of the lamp assembly"
        ),
        "detached": (
            "displace the selected front headlight slightly from its mounting while "
            "keeping the complete assembly recognizable and connected to the vehicle"
        ),
    },
    "rear_light": {
        "scratched": (
            "add light superficial scratches to the selected rear light lens while "
            "preserving colour, transparency and reflector details"
        ),
        "cracked": (
            "create exactly one main crack in the selected rear light lens, with no "
            "more than two or three short, thin secondary branches. Keep at least "
            "70 to 80 percent of the lens perfectly intact, clear and unchanged. Do "
            "not add missing fragments, cloudy areas, dense crack networks, spiderweb "
            "patterns or shattered-glass effects. Preserve the original colour, "
            "reflector details and overall lamp shape"
        ),
        "broken": (
            "break the selected rear light lens in a clearly visible but localized "
            "way. Show several intersecting cracks and one or two small missing "
            "fragments while preserving the housing and most of the light assembly"
        ),
        "shattered": (
            "heavily shatter most of the selected rear light lens with dense crack "
            "networks, multiple missing fragments and visible internal reflectors, "
            "while preserving the overall housing position"
        ),
        "partially_missing": (
            "remove a limited portion of the selected rear light lens, with realistic "
            "broken edges and exposed internal reflectors, while preserving the rest "
            "of the assembly"
        ),
        "detached": (
            "displace the selected rear light slightly from its mounting while "
            "keeping the complete assembly recognizable"
        ),
    },
    "front_bumper": {
        "scratched": (
            "add localized superficial scratches to the existing front bumper "
            "surface while preserving exactly its original shape, colour, trim, "
            "seams, grilles, openings and mounting position"
        ),
        "cracked": (
            "create one or two localized cracks in the existing front bumper plastic. "
            "Preserve exactly the original bumper shape, colour, trim, seams, grilles, "
            "openings, mounting position and surrounding bodywork. Do not redesign, "
            "replace or recolour the bumper. Do not add parking sensors, covers, "
            "grilles, mouldings, vents, fog lights, openings or details that were not "
            "present in the source image"
        ),
        "deformed": (
            "create a localized plastic deformation in the existing front bumper "
            "surface while preserving the bumper's overall original geometry, colour, "
            "trim, seams, grilles, openings and mounting position. Do not replace or "
            "redesign the bumper and do not invent new parts or details"
        ),
        "broken": (
            "create a clearly visible but localized break in the existing front "
            "bumper plastic, with realistic cracked edges and limited missing material. "
            "Keep the bumper recognizable and preserve its original colour, shape, "
            "trim, seams, grilles, openings and mounting position. Do not reconstruct "
            "the bumper as a different component"
        ),
        "unclipped": (
            "slightly unclip the existing front bumper at one local mounting point "
            "while keeping the complete original bumper recognizable and preserving "
            "its colour, trim, geometry and surrounding bodywork"
        ),
        "partially_detached": (
            "partially release the existing front bumper from one mounting point while "
            "keeping the complete original bumper recognizable and preserving its "
            "colour, trim and geometry. Do not redesign or replace the bumper"
        ),
        "hanging": (
            "make one side of the existing front bumper hang slightly from its original "
            "mounting while preserving the original component identity, colour, trim "
            "and geometry. Do not replace or redesign the bumper"
        ),
    },
    "rear_bumper": {
        "scratched": (
            "add localized superficial scratches to the existing rear bumper "
            "surface while preserving exactly its original shape, colour, trim, "
            "seams, openings and mounting position"
        ),
        "cracked": (
            "create one or two localized cracks in the existing rear bumper plastic. "
            "Preserve exactly the original bumper shape, colour, trim, seams, openings, "
            "mounting position and surrounding bodywork. Do not redesign, replace or "
            "recolour the bumper. Do not add parking sensors, covers, grilles, "
            "mouldings, vents, openings or details that were not present in the source "
            "image"
        ),
        "deformed": (
            "create a localized plastic deformation in the existing rear bumper "
            "surface while preserving the bumper's overall original geometry, colour, "
            "trim, seams, openings and mounting position. Do not replace or redesign "
            "the bumper and do not invent new parts or details"
        ),
        "broken": (
            "create a clearly visible but localized break in the existing rear bumper "
            "plastic, with realistic cracked edges and limited missing material. Keep "
            "the bumper recognizable and preserve its original colour, shape, trim, "
            "seams, openings and mounting position. Do not reconstruct the bumper as "
            "a different component"
        ),
        "unclipped": (
            "slightly unclip the existing rear bumper at one local mounting point "
            "while keeping the complete original bumper recognizable and preserving "
            "its colour, trim, geometry and surrounding bodywork"
        ),
        "partially_detached": (
            "partially release the existing rear bumper from one mounting point while "
            "keeping the complete original bumper recognizable and preserving its "
            "colour, trim and geometry. Do not redesign or replace the bumper"
        ),
        "hanging": (
            "make one side of the existing rear bumper hang slightly from its original "
            "mounting while preserving the original component identity, colour, trim "
            "and geometry. Do not replace or redesign the bumper"
        ),
    },
    "hood": COMPONENT_DAMAGE_TEXT["hood"],
    "tailgate": {
        "dented": "dent the tailgate",
        "edge_bent": "bend the selected tailgate edge",
        "misaligned": "misalign the tailgate along the affected seam",
        "partially_open": "make the tailgate appear partially open from impact",
        "severely_deformed": "severely deform the selected tailgate area",
    },
    "front_door": {
        "dented": "dent the front door",
        "creased": "create a realistic crease in the front door",
        "scratched": "scratch the front door",
        "misaligned": "misalign the front door along the affected seam",
        "jammed": "make the front door appear jammed by the collision",
    },
    "rear_door": {
        "dented": "dent the rear door",
        "creased": "create a realistic crease in the rear door",
        "scratched": "scratch the rear door",
        "misaligned": "misalign the rear door along the affected seam",
        "jammed": "make the rear door appear jammed by the collision",
    },
    "wheel_arch": {
        "dented": "dent the wheel arch",
        "creased": "create a realistic crease in the wheel arch",
        "crushed": "crush the selected wheel arch area",
        "scratched": "scratch the wheel arch",
    },
    "wheel": COMPONENT_DAMAGE_TEXT["wheel"],
    "windshield": COMPONENT_DAMAGE_TEXT["glass"],
    "rear_window": COMPONENT_DAMAGE_TEXT["glass"],
    "side_window": COMPONENT_DAMAGE_TEXT["glass"],
    "roof": {
        "dented": "dent the roof",
        "creased": "create a realistic crease in the roof",
        "crushed": "crush the selected roof area",
        "scratched": "scratch the roof",
    },
    "side_mirror": {
        "scratched": (
            "add realistic scratches to the selected side mirror while keeping "
            "the mirror assembly, glass and mounting recognizable"
        ),
        "cracked": (
            "create realistic cracks in the mirror glass or outer housing, "
            "without turning the mirror into a black silhouette"
        ),
        "broken": (
            "break the painted outer housing with realistic cracks and limited "
            "missing fragments; keep the mirror glass, mounting base and internal "
            "mechanism recognizable; do not erase the mirror and do not replace it "
            "with a black blob"
        ),
        "detached": (
            "partially detach the mirror assembly from its mounting base; keep the "
            "mirror recognizable and naturally displaced; show the mounting point "
            "and, where plausible, a short electrical cable; do not create a black "
            "hole or completely erase the mirror"
        ),
        "hanging": (
            "make the mirror assembly hang naturally from its mounting or cable, "
            "while preserving recognizable glass, housing and attachment details"
        ),
        "glass_cracked": (
            "crack only the mirror glass while preserving the outer housing"
        ),
        "glass_shattered": (
            "shatter the mirror glass with realistic fragments remaining in the "
            "housing; preserve the housing and mounting"
        ),
        "housing_broken": (
            "break only the painted outer plastic housing of the selected side "
            "mirror. Show visible cracked plastic edges, realistic plastic "
            "thickness, one or two missing housing fragments, scratches and "
            "fracture lines, with the internal support structure only partially "
            "visible. Keep the mirror glass opaque, reflective and recognizable. "
            "Keep the mounting base attached and preserve the original size, "
            "position and orientation of the mirror assembly. Do not create "
            "transparent or translucent plastic, melted material, an empty shell, "
            "an organic-looking interior, a black blob or a featureless dark shape."
        ),
        "partially_detached": (
            "partially detach the mirror from the door while keeping the complete "
            "assembly recognizable"
        ),
    },
}



def component_is_enabled(components: dict[str, bool], *names: str) -> bool:
    return any(bool(components.get(name, False)) for name in names)


BODYWORK_COMPONENT_CODES = {
    "bodyPanel",
    "body_panel",
    "front_fender",
    "rear_fender",
    "hood",
    "tailgate",
    "front_door",
    "rear_door",
    "wheel_arch",
    "roof",
}

NON_BODY_COMPONENT_CODES = {
    "front_headlight",
    "rear_light",
    "front_bumper",
    "rear_bumper",
    "wheel",
    "windshield",
    "rear_window",
    "side_window",
    "side_mirror",
}


def infer_damage_mode(payload: DamageEditBase64Request) -> str:
    if payload.damage_mode != "auto":
        return payload.damage_mode

    selected = {
        code
        for code, enabled in (payload.involved_components or {}).items()
        if bool(enabled)
    }

    has_bodywork = bool(selected & BODYWORK_COMPONENT_CODES)
    has_non_body = bool(selected & NON_BODY_COMPONENT_CODES)

    if has_bodywork and has_non_body:
        return "mixed"
    if has_non_body and not has_bodywork:
        return "component_only"
    return "bodywork"


def has_component_damage_request(payload: DamageEditBase64Request) -> bool:
    return any(
        bool(value)
        for value in (payload.component_damage_types or {}).values()
    ) or any(
        bool(value)
        for value in (
            payload.hood_damage_type,
            payload.headlight_damage_type,
            payload.bumper_damage_type,
            payload.wheel_damage_type,
            payload.glass_damage_type,
        )
    )


def selected_bodywork_component_codes(
    payload: DamageEditBase64Request,
) -> list[str]:
    components = payload.involved_components or {}
    return [
        code
        for code, enabled in components.items()
        if bool(enabled) and code in BODYWORK_COMPONENT_CODES
    ]


def selected_non_body_component_codes(
    payload: DamageEditBase64Request,
) -> list[str]:
    components = payload.involved_components or {}
    return [
        code
        for code, enabled in components.items()
        if bool(enabled) and code in NON_BODY_COMPONENT_CODES
    ]



def copy_request_model(
    payload: DamageEditBase64Request,
    updates: dict,
) -> DamageEditBase64Request:
    if hasattr(payload, "model_copy"):
        return payload.model_copy(update=updates)
    return payload.copy(update=updates)


def payload_for_single_component(
    payload: DamageEditBase64Request,
    component_code: str,
) -> DamageEditBase64Request:
    """
    Crea una copia del payload limitata a un solo componente.
    Serve per generare ogni danno non-lamiera in un passaggio indipendente.
    """
    selected_damage_type = (
        payload.component_damage_types or {}
    ).get(component_code)

    return copy_request_model(
        payload,
        {
            "damage_mode": "component_only",
            "involved_components": {component_code: True},
            "component_damage_types": (
                {component_code: selected_damage_type}
                if selected_damage_type
                else {}
            ),
            "bodywork_mask_base64": None,
            "component_masks_base64": {},
        }
    )


def jpeg_bytes_to_rgb_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def build_component_instructions(payload: DamageEditBase64Request) -> str:
    components = payload.involved_components or {"bodyPanel": True}
    dynamic_damage_types = payload.component_damage_types or {}

    enabled_codes = [
        code
        for code, enabled in components.items()
        if bool(enabled)
    ]

    lines: list[str] = []

    if payload.vehicle_view:
        lines.append(f"Vehicle view reported by the interface: {payload.vehicle_view}.")

    if enabled_codes:
        enabled_labels = [
            VEHICLE_COMPONENT_CATALOG.get(
                code,
                COMPONENT_LABELS.get(code, code.replace("_", " ")),
            )
            for code in enabled_codes
        ]
        lines.append(
            "Components explicitly involved: "
            + ", ".join(enabled_labels)
            + "."
        )

    # Nuovo formato dinamico.
    for component_code, damage_type in dynamic_damage_types.items():
        if not component_is_enabled(components, component_code):
            continue

        instruction = DYNAMIC_COMPONENT_DAMAGE_TEXT.get(
            component_code,
            {},
        ).get(damage_type)

        if instruction:
            label = VEHICLE_COMPONENT_CATALOG.get(
                component_code,
                component_code.replace("_", " "),
            )
            lines.append(f"For {label}: {instruction}.")

    # Compatibilità con i campi precedenti.
    legacy_damage_values = {
        "hood": payload.hood_damage_type,
        "headlight": payload.headlight_damage_type,
        "bumper": payload.bumper_damage_type,
        "wheel": payload.wheel_damage_type,
        "glass": payload.glass_damage_type,
    }

    legacy_aliases = {
        "hood": ("hood",),
        "headlight": ("headlight", "front_headlight", "rear_light"),
        "bumper": ("bumper", "front_bumper", "rear_bumper"),
        "wheel": ("wheel",),
        "glass": ("glass", "windshield", "rear_window", "side_window"),
    }

    for component, damage_type in legacy_damage_values.items():
        if not damage_type:
            continue
        if not component_is_enabled(components, *legacy_aliases[component]):
            continue

        instruction = COMPONENT_DAMAGE_TEXT.get(component, {}).get(damage_type)
        if instruction:
            lines.append(f"For the {component}: {instruction}.")

    # Protegge soltanto le categorie non coinvolte.
    protected_components: list[str] = []

    if not component_is_enabled(components, "hood"):
        protected_components.append("hood")

    if not component_is_enabled(
        components,
        "headlight",
        "front_headlight",
        "rear_light",
    ):
        protected_components.append(
            "all headlights, rear lights, lenses, reflectors and optical assemblies"
        )

    if not component_is_enabled(
        components,
        "bumper",
        "front_bumper",
        "rear_bumper",
    ):
        protected_components.append("front and rear bumpers")

    if not component_is_enabled(
        components,
        "wheel",
        "wheelArch",
        "wheel_arch",
    ):
        protected_components.append("wheels, tyres and wheel arches")

    if not component_is_enabled(
        components,
        "glass",
        "windshield",
        "rear_window",
        "side_window",
    ):
        protected_components.append("all vehicle glass")

    if not component_is_enabled(components, "tailgate"):
        protected_components.append("tailgate")

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


def bodywork_geometry_instruction(
    severity: int,
    area: int,
) -> str:
    """
    Traduce gravità ed estensione in vincoli geometrici più concreti.

    Obiettivo:
    - ridurre deformazioni troppo ampie a bassa intensità;
    - evitare onde lisce, pieghe decorative e pannelli interamente modellati;
    - rendere più stabile la relazione tra percentuali e risultato visivo.
    """
    severity_abs = abs(severity)
    area_abs = abs(area)

    if severity_abs <= 30:
        severity_rule = (
            "Create one localized primary impact depression and no more than two "
            "short secondary creases. Keep most of the selected panel visually "
            "unchanged. Preserve the original wheel-arch profile, panel perimeter "
            "and adjacent seams."
        )
    elif severity_abs <= 60:
        severity_rule = (
            "Create one clear primary impact depression with two or three related "
            "creases. Allow moderate depth, but preserve the panel identity, main "
            "perimeter, wheel-arch profile and neighbouring seams."
        )
    else:
        severity_rule = (
            "Create a strong but physically plausible collision deformation with "
            "a dominant impact zone and a limited number of connected folds. Avoid "
            "random wrinkling and preserve recognizable panel boundaries."
        )

    if area_abs <= 20:
        area_rule = (
            "Concentrate the visible deformation in a compact portion of the "
            "editable mask. Do not use the entire mask just because it is available."
        )
    elif area_abs <= 60:
        area_rule = (
            "Use a medium portion of the editable mask, leaving clearly unchanged "
            "areas around the primary impact zone."
        )
    else:
        area_rule = (
            "The damage may use a broad portion of the editable mask, but it must "
            "still have one dominant impact area rather than uniform deformation."
        )

    return f"{severity_rule} {area_rule}"


def build_prompt(
    severity: int,
    area: int,
    damage_mode: str = "bodywork",
    deformation_type: str = "dent",
    impact_direction: str = "frontal",
    component_instructions: str = "",
    contact_trace_instructions: str = "",
) -> str:
    direction_text = IMPACT_DIRECTION_INSTRUCTIONS.get(
        impact_direction,
        IMPACT_DIRECTION_INSTRUCTIONS["frontal"],
    )

    common_constraints = """
General constraints:
- preserve the same vehicle, model, colour, perspective and framing;
- modify only areas permitted by the editable mask;
- the protected mask has absolute priority;
- preserve all non-involved components and everything outside the editable mask;
- maintain realistic materials, edges, thickness, shadows and reflections;
- do not create black blobs, black silhouettes, featureless patches,
  liquid surfaces, duplicated objects, pasted shapes or transparent bodywork;
- return the complete final photograph, photorealistic and without watermark.
""".strip()

    if damage_mode == "component_only":
        intensity = abs(severity)
        intensity_text = (
            "light"
            if intensity <= 30
            else "moderate"
            if intensity <= 65
            else "strong but physically plausible"
        )

        return f"""
Realistic automotive COMPONENT repair-estimate simulation.

This is a component-only edit. Do not apply body-panel dent logic.

Requested component damage:
{component_instructions}

Impact:
- {direction_text}
- damage intensity: {intensity_text} ({severity:+d}% reference).

Contact traces:
{contact_trace_instructions or "Do not add contact traces."}

Component-only rules:
- execute the selected component damage literally and locally;
- keep the damaged component recognizable unless the selected damage explicitly
  requests missing or detached parts;
- for lights and glass, strictly distinguish between cracked, broken and shattered:
  cracked means exactly one main crack, at most two or three short branches, no
  missing fragments and at least 70 to 80 percent of the lens perfectly intact;
  broken means several cracks plus one or two small missing fragments;
  shattered means dense cracking across most of the lens with multiple fragments;
- show realistic plastic, glass, metal, mounting points and material thickness;
- when breaking or detaching a component, preserve believable attachment details;
- for bumpers, preserve exactly the original component identity, colour, shape,
  trim, seams, openings, grilles and mounting position unless the selected damage
  explicitly requests a limited local detachment;
- never redesign, replace, recolour or invent bumper details such as parking sensors,
  covers, grilles, mouldings, vents, fog lights or openings not present in the source;
- do not add dents, folds or crushed sheet metal unless bodywork is also selected;
- do not interpret broken or detached as a solid black shape or erased area.

{common_constraints}
""".strip()

    severity_text = severity_instruction(severity)
    area_text = area_instruction(area)
    deformation_text = DEFORMATION_INSTRUCTIONS.get(
        deformation_type,
        DEFORMATION_INSTRUCTIONS["dent"],
    )
    geometry_text = bodywork_geometry_instruction(
        severity=severity,
        area=area,
    )

    if damage_mode == "mixed":
        mode_heading = (
            "This is a mixed edit: apply bodywork deformation only to selected "
            "sheet-metal components, and apply each specific component instruction "
            "to its corresponding selected component."
        )
    else:
        mode_heading = (
            "This is a bodywork edit. Apply deformation only to selected "
            "sheet-metal components."
        )

    return f"""
Realistic automotive collision photo editing.

{mode_heading}

Bodywork objective:
- {severity_text}
- {area_text}
- {deformation_text}
- {direction_text}

Specific component rules:
{component_instructions or "Modify only the selected painted body panel."}

Contact traces:
{contact_trace_instructions or "Do not add contact traces."}

Bodywork geometry control:
- {geometry_text}
- create one dominant impact point;
- keep large portions of the selected panel unchanged at low and medium severity;
- do not deform the whole panel unless severity and area are both high;
- avoid smooth sculpted waves, inflated surfaces, repeated folds,
  decorative wrinkles and uniformly softened metal;
- preserve the original wheel-arch curve, panel perimeter and nearby shut lines
  at low and medium severity.

Bodywork rules:
- keep deformation concentrated in the selected sheet-metal area;
- create physically plausible stamped-metal damage;
- maintain coherent paint, panel edges, shadows and reflections;
- do not apply dent, crease or crush logic to glass, lights, mirrors or wheels;
- in mixed mode, execute non-body component damage separately and literally.

{common_constraints}
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



def _lab_color_match_inside_mask(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    mask_binary: np.ndarray,
    strength: float = SMART_COMPOSITE_COLOR_STRENGTH,
) -> np.ndarray:
    if strength <= 0:
        return generated_rgb

    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    generated_lab = cv2.cvtColor(
        generated_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    ring_kernel = np.ones((21, 21), np.uint8)
    outer = cv2.dilate(mask_binary, ring_kernel, iterations=1)
    inner = cv2.erode(mask_binary, np.ones((7, 7), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)

    sample = ring > 0
    if int(sample.sum()) < 64:
        sample = mask_binary > 0

    if int(sample.sum()) < 32:
        return generated_rgb

    corrected = generated_lab.copy()

    for channel in range(3):
        source_values = source_lab[:, :, channel][sample]
        generated_values = generated_lab[:, :, channel][sample]

        source_mean = float(source_values.mean())
        generated_mean = float(generated_values.mean())
        source_std = max(float(source_values.std()), 1.0)
        generated_std = max(float(generated_values.std()), 1.0)

        matched = (
            (generated_lab[:, :, channel] - generated_mean)
            * (source_std / generated_std)
            + source_mean
        )

        corrected[:, :, channel] = (
            generated_lab[:, :, channel] * (1.0 - strength)
            + matched * strength
        )

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)

    # Applica il color match soltanto dentro e poco attorno alla maschera.
    local_zone = cv2.dilate(
        mask_binary,
        np.ones((15, 15), np.uint8),
        iterations=1,
    )
    local_alpha = (
        cv2.GaussianBlur(local_zone, (0, 0), sigmaX=5)
        .astype(np.float32)
        / 255.0
    )[:, :, None]

    return np.clip(
        generated_rgb.astype(np.float32) * (1.0 - local_alpha)
        + corrected_rgb.astype(np.float32) * local_alpha,
        0,
        255,
    ).astype(np.uint8)


def _laplacian_pyramid_blend(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    alpha: np.ndarray,
    levels: int = SMART_COMPOSITE_PYRAMID_LEVELS,
) -> np.ndarray:
    source = source_rgb.astype(np.float32)
    generated = generated_rgb.astype(np.float32)
    mask = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    if mask.ndim == 2:
        mask = mask[:, :, None]

    gaussian_source = [source]
    gaussian_generated = [generated]
    gaussian_mask = [mask]

    for _ in range(levels):
        if min(gaussian_source[-1].shape[:2]) < 16:
            break

        gaussian_source.append(cv2.pyrDown(gaussian_source[-1]))
        gaussian_generated.append(cv2.pyrDown(gaussian_generated[-1]))
        gaussian_mask.append(cv2.pyrDown(gaussian_mask[-1]))

    laplacian_source = []
    laplacian_generated = []

    for index in range(len(gaussian_source) - 1):
        size = (
            gaussian_source[index].shape[1],
            gaussian_source[index].shape[0],
        )
        source_up = cv2.pyrUp(
            gaussian_source[index + 1],
            dstsize=size,
        )
        generated_up = cv2.pyrUp(
            gaussian_generated[index + 1],
            dstsize=size,
        )
        laplacian_source.append(gaussian_source[index] - source_up)
        laplacian_generated.append(
            gaussian_generated[index] - generated_up
        )

    laplacian_source.append(gaussian_source[-1])
    laplacian_generated.append(gaussian_generated[-1])

    blended_levels = []

    for source_level, generated_level, mask_level in zip(
        laplacian_source,
        laplacian_generated,
        gaussian_mask,
    ):
        if mask_level.ndim == 2:
            mask_level = mask_level[:, :, None]

        blended_levels.append(
            generated_level * mask_level
            + source_level * (1.0 - mask_level)
        )

    result = blended_levels[-1]

    for index in range(len(blended_levels) - 2, -1, -1):
        size = (
            blended_levels[index].shape[1],
            blended_levels[index].shape[0],
        )
        result = cv2.pyrUp(result, dstsize=size) + blended_levels[index]

    return np.clip(result, 0, 255).astype(np.uint8)


def smart_component_composite(
    source: Image.Image,
    generated_bytes: bytes,
    source_mask: Image.Image,
    protect_mask: Image.Image | None = None,
    expansion_px: int = COMPOSITE_EXPANSION_PX,
    feather_px: int = SOFT_COMPOSITE_FEATHER_PX,
) -> bytes:
    if not SMART_COMPOSITE_ENABLED:
        return protected_soft_composite(
            source=source,
            generated_bytes=generated_bytes,
            source_mask=source_mask,
            protect_mask=protect_mask,
            expansion_px=expansion_px,
            feather_px=feather_px,
        )

    generated = _open_image_bytes(generated_bytes, "RGB")
    if generated is None:
        raise HTTPException(
            status_code=502,
            detail="Il risultato generato non è un'immagine valida.",
        )

    original_size = source.size

    work_source, scale = resize_image_for_processing(
        source.convert("RGB"),
        SMART_COMPOSITE_MAX_SIDE,
    )
    work_generated = generated.resize(
        work_source.size,
        Image.Resampling.LANCZOS,
    )
    work_mask = resize_mask(
        source_mask.convert("L"),
        work_source.size,
    )

    if protect_mask is not None:
        work_protect = resize_mask(
            protect_mask.convert("L"),
            work_source.size,
        )
    else:
        work_protect = None

    adjusted_expansion = max(
        0,
        round(expansion_px * scale),
    )
    adjusted_feather = max(
        2,
        round(feather_px * scale),
    )

    source_rgb = np.asarray(work_source, dtype=np.uint8)
    generated_rgb = np.asarray(work_generated, dtype=np.uint8)

    mask_binary = mask_to_binary(work_mask)

    if work_protect is not None:
        protect_binary = mask_to_binary(work_protect)
        mask_binary = cv2.bitwise_and(
            mask_binary,
            cv2.bitwise_not(protect_binary),
        )
    else:
        protect_binary = np.zeros_like(mask_binary)

    if adjusted_expansion > 0:
        kernel_size = max(3, adjusted_expansion * 2 + 1)
        expanded = cv2.dilate(
            mask_binary,
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        )
    else:
        expanded = mask_binary.copy()

    expanded = cv2.bitwise_and(
        expanded,
        cv2.bitwise_not(protect_binary),
    )

    if int((expanded > 0).sum()) == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera effettiva del compositing è vuota.",
        )

    corrected_generated = _lab_color_match_inside_mask(
        source_rgb,
        generated_rgb,
        expanded,
    )

    distance_inside = cv2.distanceTransform(
        np.where(expanded > 0, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    feather = max(2.0, float(adjusted_feather))
    alpha = np.clip(distance_inside / feather, 0.0, 1.0)

    alpha[mask_binary > 0] = np.maximum(
        alpha[mask_binary > 0],
        0.92,
    )

    alpha = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=1.0,
    )
    alpha[protect_binary > 0] = 0.0

    blended = _laplacian_pyramid_blend(
        source_rgb,
        corrected_generated,
        alpha,
        levels=SMART_COMPOSITE_PYRAMID_LEVELS,
    )
    blended[protect_binary > 0] = source_rgb[protect_binary > 0]

    work_result = Image.fromarray(blended, mode="RGB")

    if work_result.size != original_size:
        work_result = work_result.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    output = io.BytesIO()
    work_result.save(
        output,
        format="JPEG",
        quality=94,
        subsampling=0,
    )

    del generated
    del work_source
    del work_generated
    del work_mask
    del source_rgb
    del generated_rgb
    del mask_binary
    del protect_binary
    del expanded
    del corrected_generated
    del distance_inside
    del alpha
    del blended
    del work_result

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





def resize_image_for_processing(
    image: Image.Image,
    max_side: int,
) -> tuple[Image.Image, float]:
    width, height = image.size
    current_max = max(width, height)

    if current_max <= max_side:
        return image.copy(), 1.0

    scale = max_side / float(current_max)
    resized = image.resize(
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def scale_polygon_points(
    points: list[tuple[int, int]],
    scale: float,
) -> list[tuple[int, int]]:
    if scale == 1.0:
        return list(points)
    return [
        (
            max(0, round(x * scale)),
            max(0, round(y * scale)),
        )
        for x, y in points
    ]


def upscale_mask_to_original(
    mask: Image.Image,
    original_size: tuple[int, int],
) -> Image.Image:
    if mask.size == original_size:
        return mask.convert("L")
    return mask.convert("L").resize(
        original_size,
        Image.Resampling.NEAREST,
    )


def component_category(code: str) -> str:
    return VEHICLE_COMPONENT_CATEGORIES.get(code, "trim")


def mask_image_to_data_url(mask: Image.Image) -> str:
    normalized = mask.convert("L")
    buffer = io.BytesIO()
    normalized.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_box(
    raw_box,
    width: int,
    height: int,
) -> dict | None:
    values = None

    if isinstance(raw_box, dict):
        aliases = [
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("xmin", "ymin", "xmax", "ymax"),
        ]
        for names in aliases:
            if all(name in raw_box for name in names):
                values = [raw_box[name] for name in names]
                break

        if values is None and all(
            name in raw_box for name in ("x", "y", "width", "height")
        ):
            x = raw_box["x"]
            y = raw_box["y"]
            values = [
                x,
                y,
                float(x) + float(raw_box["width"]),
                float(y) + float(raw_box["height"]),
            ]

    elif isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        values = list(raw_box[:4])

    if values is None:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except Exception:
        return None

    maximum = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if maximum <= 1.5:
        scale_x = width - 1
        scale_y = height - 1
    elif maximum <= 1100:
        scale_x = (width - 1) / 1000.0
        scale_y = (height - 1) / 1000.0
    else:
        scale_x = scale_y = 1.0

    x1 = round(x1 * scale_x)
    y1 = round(y1 * scale_y)
    x2 = round(x2 * scale_x)
    y2 = round(y2 * scale_y)

    left = max(0, min(width - 1, min(x1, x2)))
    top = max(0, min(height - 1, min(y1, y2)))
    right = max(left + 1, min(width, max(x1, x2)))
    bottom = max(top + 1, min(height, max(y1, y2)))

    if right - left < 4 or bottom - top < 4:
        return None

    return {
        "x1": int(left),
        "y1": int(top),
        "x2": int(right),
        "y2": int(bottom),
        "x": int(left),
        "y": int(top),
        "width": int(right - left),
        "height": int(bottom - top),
    }


def _replicate_json_request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    if not REPLICATE_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "REPLICATE_API_TOKEN non configurato su Render.",
                "analysis_version": "vehicle-segmentation-v16.2-prompted-component-segmentation",
            },
        )

    body = None
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REPLICATE_TIMEOUT_SECONDS,
        ) as response:
            content = response.read().decode("utf-8")
            return json.loads(content or "{}")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(
            "REPLICATE HTTP ERROR:",
            exc.code,
            url,
            error_body[:2000],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Errore API Replicate.",
                "http_status": exc.code,
                "request_url": url,
                "replicate_detail": error_body[:2000],
                "analysis_version": "vehicle-segmentation-v16.2-prompted-component-segmentation",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Connessione a Replicate non riuscita.",
                "error": f"{type(exc).__name__}: {str(exc)}"[:1200],
                "analysis_version": "vehicle-segmentation-v16.2-prompted-component-segmentation",
            },
        ) from exc


def _download_binary_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "image/png,image/webp,image/jpeg,image/*;q=0.9,*/*;q=0.1",
            "User-Agent": "CarDamageLab/16.1.5",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REPLICATE_TIMEOUT_SECONDS,
        ) as response:
            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .strip()
                .lower()
            )
            raw = response.read()

            if not raw:
                raise ValueError("Risposta file vuota.")

            if content_type and not (
                content_type.startswith("image/")
                or content_type == "application/octet-stream"
            ):
                preview = raw[:500].decode(
                    "utf-8",
                    errors="replace",
                )
                raise ValueError(
                    f"Content-Type inatteso: {content_type}; "
                    f"preview={preview}"
                )

            return raw

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            "REPLICATE OUTPUT FILE HTTP ERROR:",
            exc.code,
            url,
            body[:1200],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Download output Replicate non riuscito.",
                "http_status": exc.code,
                "output_url": url,
                "response_preview": body[:1200],
                "analysis_version": (
                    "vehicle-segmentation-v16.1.5-"
                    "replicate-output-download-fix"
                ),
            },
        ) from exc

    except Exception as exc:
        print(
            "REPLICATE OUTPUT FILE ERROR:",
            url,
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Download output Replicate non riuscito.",
                "output_url": url,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
                "analysis_version": (
                    "vehicle-segmentation-v16.1.5-"
                    "replicate-output-download-fix"
                ),
            },
        ) from exc


def call_replicate_sam2(image_data_url: str) -> dict:
    prediction = _replicate_json_request(
        "https://api.replicate.com/v1/predictions",
        method="POST",
        payload={
            "version": REPLICATE_SAM2_VERSION,
            "input": {
                "image": image_data_url,
                "points_per_side": SAM2_POINTS_PER_SIDE,
                "pred_iou_thresh": SAM2_PRED_IOU_THRESH,
                "stability_score_thresh": SAM2_STABILITY_SCORE_THRESH,
                "use_m2m": True,
            },
        },
    )

    prediction_id = prediction.get("id")
    if not prediction_id:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Replicate non ha restituito un prediction id.",
                "response": prediction,
            },
        )

    deadline = time.monotonic() + REPLICATE_TIMEOUT_SECONDS
    current = prediction

    while current.get("status") not in {
        "succeeded",
        "failed",
        "canceled",
    }:
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "Timeout durante la segmentazione SAM 2.",
                    "prediction_id": prediction_id,
                },
            )

        time.sleep(REPLICATE_POLL_SECONDS)
        current = _replicate_json_request(
            f"https://api.replicate.com/v1/predictions/{prediction_id}"
        )

    if current.get("status") != "succeeded":
        print(
            "REPLICATE SAM2 FAILED:",
            prediction_id,
            current.get("status"),
            current.get("error"),
            str(current.get("logs") or "")[-2000:],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "La segmentazione SAM 2 non è riuscita.",
                "prediction_id": prediction_id,
                "status": current.get("status"),
                "error": current.get("error"),
                "logs": str(current.get("logs") or "")[-1600:],
            },
        )

    output = current.get("output") or {}

    if not isinstance(output, dict):
        print(
            "REPLICATE UNEXPECTED OUTPUT:",
            prediction_id,
            type(output).__name__,
            repr(output)[:2000],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Formato output SAM 2 non riconosciuto.",
                "prediction_id": prediction_id,
                "output_type": type(output).__name__,
                "output_preview": repr(output)[:1200],
            },
        )

    individual_masks = output.get("individual_masks") or []

    print(
        "REPLICATE SAM2 SUCCEEDED:",
        prediction_id,
        "individual_masks=",
        len(individual_masks) if isinstance(individual_masks, list) else -1,
        "metrics=",
        current.get("metrics") or {},
    )

    if not isinstance(individual_masks, list) or not individual_masks:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "SAM 2 non ha restituito maschere individuali.",
                "prediction_id": prediction_id,
                "output_keys": list(output.keys()),
                "logs": str(current.get("logs") or "")[-1200:],
            },
        )

    return {
        "prediction_id": prediction_id,
        "individual_masks": individual_masks,
        "combined_mask": output.get("combined_mask"),
        "metrics": current.get("metrics") or {},
        "logs": current.get("logs") or "",
    }



def clean_component_mask(mask: Image.Image) -> Image.Image:
    """
    Pulisce una maschera binaria:
    - converte in bianco/nero;
    - chiude piccoli buchi;
    - rimuove rumore e componenti troppo piccoli;
    - conserva le aree principali.
    """
    binary = mask_to_binary(mask)
    height, width = binary.shape

    kernel_size = max(3, round(min(width, height) * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(cleaned > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )

    if count <= 1:
        return Image.fromarray(cleaned, mode="L")

    minimum_area = max(
        16,
        round(width * height * SEGMENTATION_MIN_AREA_RATIO),
    )

    selected = np.zeros_like(cleaned)

    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
    ]
    largest_area = max(component_areas, default=0)

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])

        if area >= minimum_area or (
            largest_area > 0 and area >= largest_area * 0.20
        ):
            selected[labels == index] = 255

    if int((selected > 0).sum()) == 0:
        selected = cleaned

    return Image.fromarray(selected, mode="L")


def _decode_sam_mask_image(
    raw: bytes,
    target_size: tuple[int, int],
) -> tuple[Image.Image, str]:
    original = Image.open(io.BytesIO(raw))
    rgba = original.convert("RGBA")
    rgba = rgba.resize(
        target_size,
        Image.Resampling.NEAREST,
    )

    array = np.asarray(rgba, dtype=np.uint8)
    rgb = array[:, :, :3]
    alpha = array[:, :, 3]

    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    alpha_active = alpha > 8
    alpha_ratio = float(alpha_active.mean())

    # Prefer transparency when it clearly identifies a limited foreground.
    if 0.0001 < alpha_ratio < 0.95:
        binary = np.where(alpha_active, 255, 0).astype(np.uint8)
        mode = "alpha"
    else:
        bright = luminance > 127
        bright_ratio = float(bright.mean())

        # SAM output variants can be white-on-black or black-on-white.
        if bright_ratio <= 0.50:
            binary = np.where(bright, 255, 0).astype(np.uint8)
            mode = "bright_foreground"
        else:
            binary = np.where(~bright, 255, 0).astype(np.uint8)
            mode = "dark_foreground_inverted"

    # Reject an almost full-frame foreground and try the opposite polarity.
    ratio = float((binary > 0).mean())
    if ratio > 0.92:
        binary = cv2.bitwise_not(binary)
        mode += "_auto_inverted"

    binary = np.where(binary > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L"), mode


def load_sam_candidate_masks(
    mask_urls: list[str],
    target_size: tuple[int, int],
) -> list[dict]:
    candidates: list[dict] = []
    download_errors: list[dict] = []

    for index, url in enumerate(mask_urls):
        try:
            raw = _download_binary_url(url)
            image, decode_mode = _decode_sam_mask_image(
                raw,
                target_size,
            )
            image = clean_component_mask(image)
            binary = mask_to_binary(image)
            area = int((binary > 0).sum())
            ratio = area / max(1, binary.size)

            if area < 32 or ratio > 0.95:
                continue

            candidates.append({
                "index": index,
                "url": url,
                "mask": image,
                "binary": binary,
                "area": area,
                "area_ratio": round(ratio, 6),
                "decode_mode": decode_mode,
            })

        except Exception as exc:
            error_item = {
                "index": index,
                "url": str(url),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            download_errors.append(error_item)
            print(
                "SAM2 MASK DOWNLOAD ERROR:",
                error_item,
            )

    if not candidates:
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Le maschere SAM 2 sono state create ma non sono "
                    "state decodificate correttamente."
                ),
                "mask_url_count": len(mask_urls),
                "download_error_count": len(download_errors),
                "download_errors_preview": download_errors[:5],
                "analysis_version": (
                    "vehicle-segmentation-v16.1.4-"
                    "sam-mask-decode-fallback"
                ),
            },
        )

    print(
        "SAM2 MASKS DECODED:",
        len(candidates),
        [
            {
                "index": item["index"],
                "mode": item["decode_mode"],
                "ratio": item["area_ratio"],
            }
            for item in candidates[:10]
        ],
    )

    return candidates


def score_mask_for_box(
    candidate_binary: np.ndarray,
    box: dict,
) -> tuple[float, dict]:
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]

    box_area = max(1, (x2 - x1) * (y2 - y1))
    mask_area = max(1, int((candidate_binary > 0).sum()))
    intersection = int(
        (candidate_binary[y1:y2, x1:x2] > 0).sum()
    )

    box_coverage = intersection / box_area
    mask_inside_ratio = intersection / mask_area
    outside_ratio = 1.0 - mask_inside_ratio

    ys, xs = np.where(candidate_binary > 0)
    if len(xs) > 0:
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        cx = cy = 0.0

    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    box_diag = max(
        1.0,
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
    )
    center_distance = (
        ((cx - box_cx) ** 2 + (cy - box_cy) ** 2) ** 0.5
    ) / box_diag
    center_score = max(0.0, 1.0 - center_distance)

    candidate_box = cv2.boundingRect(candidate_binary)
    candidate_box_area = max(1, candidate_box[2] * candidate_box[3])
    size_ratio = min(box_area, candidate_box_area) / max(
        box_area,
        candidate_box_area,
    )

    score = (
        box_coverage * 0.42
        + mask_inside_ratio * 0.24
        + center_score * 0.22
        + size_ratio * 0.12
    )

    return score, {
        "intersection_pixels": intersection,
        "box_coverage": round(box_coverage, 4),
        "mask_inside_ratio": round(mask_inside_ratio, 4),
        "outside_ratio": round(outside_ratio, 4),
        "center_score": round(center_score, 4),
        "size_ratio": round(size_ratio, 4),
    }



def build_box_fallback_mask(
    source_size: tuple[int, int],
    box: dict,
) -> Image.Image:
    width, height = source_size
    mask = np.zeros((height, width), dtype=np.uint8)

    margin_x = max(2, round(box["width"] * 0.04))
    margin_y = max(2, round(box["height"] * 0.04))

    x1 = max(0, box["x1"] + margin_x)
    y1 = max(0, box["y1"] + margin_y)
    x2 = min(width, box["x2"] - margin_x)
    y2 = min(height, box["y2"] - margin_y)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = (
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
        )

    mask[y1:y2, x1:x2] = 255

    feather = max(3, round(min(box["width"], box["height"]) * 0.03))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
    mask = np.where(mask > 32, 255, 0).astype(np.uint8)

    return Image.fromarray(mask, mode="L")



def component_detection_box(
    component: dict,
    width: int,
    height: int,
) -> dict | None:
    raw_box = (
        component.get("bounding_box")
        or component.get("box")
        or component.get("bbox")
    )

    box = normalize_box(raw_box, width, height)
    if box is not None:
        return box

    # Backward compatibility: derive a box from polygon points if returned.
    polygon = component.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        xs = []
        ys = []

        for point in polygon:
            if isinstance(point, dict):
                raw_x = point.get("x")
                raw_y = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                raw_x, raw_y = point[0], point[1]
            else:
                continue

            try:
                xs.append(float(raw_x))
                ys.append(float(raw_y))
            except Exception:
                continue

        if len(xs) >= 3 and len(ys) >= 3:
            return normalize_box(
                {
                    "x1": min(xs),
                    "y1": min(ys),
                    "x2": max(xs),
                    "y2": max(ys),
                },
                width,
                height,
            )

    return None


def assign_sam_masks_to_components(
    source: Image.Image,
    raw_components: list[dict],
    candidates: list[dict],
) -> list[dict]:
    assignments: list[dict] = []

    sorted_components = sorted(
        raw_components,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )

    for component in sorted_components:
        code = str(component.get("code", "")).strip()
        if code not in VEHICLE_COMPONENT_CATALOG:
            continue

        box = component_detection_box(
            component,
            source.width,
            source.height,
        )
        if box is None:
            print(
                "SAM2 COMPONENT SKIPPED - INVALID BOX:",
                code,
                component,
            )
            continue

        ranked: list[dict] = []

        for candidate in candidates:
            score, diagnostics = score_mask_for_box(
                candidate["binary"],
                box,
            )
            ranked.append({
                "candidate": candidate,
                "score": score,
                "diagnostics": diagnostics,
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)

        selected_mask = None
        best = ranked[0] if ranked else None
        match_quality = "fallback"
        mask_source = "openai_box_fallback"
        diagnostics = {
            "intersection_pixels": 0,
            "box_coverage": 0.0,
            "mask_inside_ratio": 0.0,
            "outside_ratio": 1.0,
            "center_score": 0.0,
            "size_ratio": 0.0,
        }
        candidate_index = None
        match_score = 0.0
        decode_mode = None

        if best is not None:
            diagnostics = best["diagnostics"]
            candidate_index = best["candidate"]["index"]
            match_score = best["score"]
            decode_mode = best["candidate"].get("decode_mode")

            if diagnostics["intersection_pixels"] > 0:
                candidate_binary = best["candidate"]["binary"].copy()

                margin_x = max(8, round(box["width"] * 0.22))
                margin_y = max(8, round(box["height"] * 0.22))

                crop_x1 = max(0, box["x1"] - margin_x)
                crop_y1 = max(0, box["y1"] - margin_y)
                crop_x2 = min(source.width, box["x2"] + margin_x)
                crop_y2 = min(source.height, box["y2"] + margin_y)

                allowed = np.zeros_like(candidate_binary)
                allowed[crop_y1:crop_y2, crop_x1:crop_x2] = 255
                candidate_binary = cv2.bitwise_and(
                    candidate_binary,
                    allowed,
                )

                if int((candidate_binary > 0).sum()) >= 24:
                    candidate_mask = clean_component_mask(
                        Image.fromarray(candidate_binary, mode="L")
                    )
                    candidate_pixels = int(
                        (mask_to_binary(candidate_mask) > 0).sum()
                    )

                    if candidate_pixels >= 24:
                        selected_mask = candidate_mask
                        mask_source = (
                            "replicate_sam2_robust_decode_box_match"
                        )
                        match_quality = "high"

                        if (
                            diagnostics["box_coverage"] < 0.04
                            or diagnostics["center_score"] < 0.20
                        ):
                            match_quality = "review"

        if selected_mask is None:
            selected_mask = build_box_fallback_mask(
                source.size,
                box,
            )
            match_quality = "manual_review_required"

        binary = mask_to_binary(selected_mask)
        if int((binary > 0).sum()) < 24:
            continue

        x, y, width, height = cv2.boundingRect(binary)

        try:
            confidence = float(component.get("confidence", 0.70))
        except Exception:
            confidence = 0.70

        assignments.append({
            "code": code,
            "label": VEHICLE_COMPONENT_CATALOG[code],
            "category": component_category(code),
            "confidence": round(max(0.0, min(confidence, 1.0)), 2),
            "mask_base64": mask_image_to_data_url(selected_mask),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
            "detection_box": box,
            "mask_source": mask_source,
            "sam_candidate_index": candidate_index,
            "sam_match_score": round(match_score, 4),
            "sam_match_quality": match_quality,
            "requires_review": match_quality != "high",
            "sam_decode_mode": decode_mode,
            "sam_match_diagnostics": diagnostics,
        })

    print(
        "SAM2 COMPONENT ASSIGNMENTS:",
        len(assignments),
        [
            {
                "code": item["code"],
                "source": item["mask_source"],
                "quality": item["sam_match_quality"],
            }
            for item in assignments
        ],
    )

    return assignments




def image_to_data_url(image: Image.Image) -> str:
    """
    Normalizza l'immagine in JPEG per l'analisi visuale.
    Riduce solo immagini molto grandi per contenere latenza e payload.
    """
    normalized = image.convert("RGB")
    max_side = max(normalized.size)

    if max_side > 1800:
        scale = 1800 / max_side
        normalized = normalized.resize(
            (
                max(1, round(normalized.width * scale)),
                max(1, round(normalized.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    normalized.save(
        buffer,
        format="JPEG",
        quality=88,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_object(raw_text: str) -> dict:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Nessun oggetto JSON trovato nella risposta.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("La risposta JSON non è un oggetto.")
    return parsed



def _create_replicate_prediction(
    version: str,
    input_payload: dict,
) -> dict:
    prediction = _replicate_json_request(
        "https://api.replicate.com/v1/predictions",
        method="POST",
        payload={
            "version": version,
            "input": input_payload,
        },
    )

    prediction_id = prediction.get("id")
    if not prediction_id:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Replicate non ha restituito un prediction id.",
                "response": prediction,
            },
        )

    deadline = time.monotonic() + REPLICATE_TIMEOUT_SECONDS
    current = prediction

    while current.get("status") not in {
        "succeeded",
        "failed",
        "canceled",
    }:
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "Timeout durante la segmentazione guidata SAM 2.",
                    "prediction_id": prediction_id,
                },
            )

        time.sleep(REPLICATE_POLL_SECONDS)
        current = _replicate_json_request(
            f"https://api.replicate.com/v1/predictions/{prediction_id}"
        )

    if current.get("status") != "succeeded":
        raise HTTPException(
            status_code=502,
            detail={
                "message": "La segmentazione guidata SAM 2 non è riuscita.",
                "prediction_id": prediction_id,
                "status": current.get("status"),
                "error": current.get("error"),
                "logs": str(current.get("logs") or "")[-1600:],
            },
        )

    return current


def prompted_points_for_box(
    box: dict,
    image_width: int,
    image_height: int,
) -> tuple[str, str, str, str]:
    x1, y1, x2, y2 = (
        box["x1"],
        box["y1"],
        box["x2"],
        box["y2"],
    )
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    inset_x = width * PROMPTED_SAM_BOX_INSET_RATIO
    inset_y = height * PROMPTED_SAM_BOX_INSET_RATIO

    positive_points = [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (x1 + inset_x, y1 + inset_y),
        (x2 - inset_x, y2 - inset_y),
    ]

    margin_x = max(4.0, width * PROMPTED_SAM_NEGATIVE_MARGIN_RATIO)
    margin_y = max(4.0, height * PROMPTED_SAM_NEGATIVE_MARGIN_RATIO)

    negative_points = [
        (x1 - margin_x, y1 - margin_y),
        (x2 + margin_x, y1 - margin_y),
        (x1 - margin_x, y2 + margin_y),
        (x2 + margin_x, y2 + margin_y),
    ]

    all_points = positive_points + negative_points

    clipped = [
        (
            max(0, min(image_width - 1, round(x))),
            max(0, min(image_height - 1, round(y))),
        )
        for x, y in all_points
    ]

    coordinates = ",".join(
        f"[{x},{y}]" for x, y in clipped
    )
    labels = ",".join(
        ["1"] * len(positive_points)
        + ["0"] * len(negative_points)
    )
    frames = ",".join(["0"] * len(clipped))
    object_ids = ",".join(["component"] * len(clipped))

    return coordinates, labels, frames, object_ids


def _extract_prompted_output_url(output) -> str:
    if isinstance(output, str):
        return output

    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first

    if isinstance(output, dict):
        for key in (
            "black_white_masks",
            "masks",
            "output",
            "frames",
        ):
            value = output.get(key)
            if isinstance(value, list) and value:
                if isinstance(value[0], str):
                    return value[0]

        for key in (
            "combined_mask",
            "mask",
            "black_white_video",
        ):
            value = output.get(key)
            if isinstance(value, str):
                return value

    raise HTTPException(
        status_code=502,
        detail={
            "message": "Formato output SAM 2 guidato non riconosciuto.",
            "output_type": type(output).__name__,
            "output_preview": repr(output)[:1200],
        },
    )


def _mask_from_prompted_output(
    output_url: str,
    target_size: tuple[int, int],
    box: dict,
) -> Image.Image:
    raw = _download_binary_url(output_url)
    mask, _ = _decode_sam_mask_image(raw, target_size)
    binary = mask_to_binary(mask)

    # Keep only the connected component touching the positive center.
    center_x = max(0, min(target_size[0] - 1, round((box["x1"] + box["x2"]) / 2)))
    center_y = max(0, min(target_size[1] - 1, round((box["y1"] + box["y2"]) / 2)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(binary > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )

    selected = np.zeros_like(binary)

    if count > 1:
        center_label = int(labels[center_y, center_x])

        if center_label > 0:
            selected[labels == center_label] = 255
        else:
            best_label = 0
            best_intersection = 0

            for index in range(1, count):
                component = labels == index
                intersection = int(
                    component[
                        box["y1"]:box["y2"],
                        box["x1"]:box["x2"],
                    ].sum()
                )
                if intersection > best_intersection:
                    best_intersection = intersection
                    best_label = index

            if best_label > 0:
                selected[labels == best_label] = 255
    else:
        selected = binary

    if int((selected > 0).sum()) < 24:
        selected = binary

    # Hard safety crop around the detection box.
    margin_x = max(10, round(box["width"] * 0.28))
    margin_y = max(10, round(box["height"] * 0.28))
    allowed = np.zeros_like(selected)
    crop_x1 = max(0, box["x1"] - margin_x)
    crop_y1 = max(0, box["y1"] - margin_y)
    crop_x2 = min(target_size[0], box["x2"] + margin_x)
    crop_y2 = min(target_size[1], box["y2"] + margin_y)
    allowed[crop_y1:crop_y2, crop_x1:crop_x2] = 255
    selected = cv2.bitwise_and(selected, allowed)

    return clean_component_mask(
        Image.fromarray(selected, mode="L")
    )


def run_prompted_sam_for_component(
    image_data_url: str,
    source_size: tuple[int, int],
    component: dict,
) -> dict:
    code = str(component.get("code", "")).strip()

    box = component_detection_box(
        component,
        source_size[0],
        source_size[1],
    )
    if box is None:
        return {
            "code": code,
            "status": "failed",
            "error": "invalid_bounding_box",
        }

    coordinates, labels, frames, object_ids = prompted_points_for_box(
        box,
        source_size[0],
        source_size[1],
    )

    prediction = _create_replicate_prediction(
        REPLICATE_PROMPTED_SAM2_VERSION,
        {
            "input_video": image_data_url,
            "click_coordinates": coordinates,
            "click_labels": labels,
            "click_frames": frames,
            "click_object_ids": object_ids,
            "mask_type": "binary",
            "annotation_type": "mask",
            "output_video": False,
            "output_format": "png",
            "output_quality": 100,
            "output_frame_interval": 1,
        },
    )

    output_url = _extract_prompted_output_url(
        prediction.get("output")
    )
    mask = _mask_from_prompted_output(
        output_url,
        source_size,
        box,
    )

    binary = mask_to_binary(mask)
    area = int((binary > 0).sum())

    if area < 24:
        return {
            "code": code,
            "status": "failed",
            "error": "empty_prompted_mask",
            "prediction_id": prediction.get("id"),
        }

    x, y, width, height = cv2.boundingRect(binary)

    try:
        confidence = float(component.get("confidence", 0.70))
    except Exception:
        confidence = 0.70

    return {
        "code": code,
        "status": "succeeded",
        "component": {
            "code": code,
            "label": VEHICLE_COMPONENT_CATALOG.get(code, code),
            "category": component_category(code),
            "confidence": round(max(0.0, min(confidence, 1.0)), 2),
            "mask_base64": mask_image_to_data_url(mask),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
            "detection_box": box,
            "mask_source": "replicate_sam2_prompted_points",
            "sam_prediction_id": prediction.get("id"),
            "sam_match_quality": "prompted",
            "requires_review": False,
            "prompt_coordinates": coordinates,
        },
    }


def prompted_segment_components(
    image_data_url: str,
    source: Image.Image,
    raw_components: list[dict],
) -> tuple[list[dict], list[dict]]:
    valid_components = [
        component
        for component in raw_components[:PROMPTED_SAM_COMPONENT_LIMIT]
        if isinstance(component, dict)
        and str(component.get("code", "")).strip()
        in VEHICLE_COMPONENT_CATALOG
    ]

    results: list[dict] = []
    failures: list[dict] = []

    with ThreadPoolExecutor(
        max_workers=PROMPTED_SAM_MAX_WORKERS
    ) as executor:
        future_map = {
            executor.submit(
                run_prompted_sam_for_component,
                image_data_url,
                source.size,
                component,
            ): component
            for component in valid_components
        }

        for future in as_completed(future_map):
            component = future_map[future]

            try:
                item = future.result()
            except Exception as exc:
                failures.append({
                    "code": component.get("code"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                })
                continue

            if item.get("status") == "succeeded":
                results.append(item["component"])
            else:
                failures.append(item)

    results.sort(
        key=lambda item: item.get("confidence", 0.0),
        reverse=True,
    )

    return results, failures


def normalize_vehicle_analysis(
    raw: dict,
    source: Image.Image,
    prompted_components: list[dict],
    prompted_failures: list[dict],
) -> dict:
    vehicle_view = str(
        raw.get("vehicle_view", raw.get("view", "unknown"))
    ).strip()

    if vehicle_view not in VEHICLE_VIEW_LABELS:
        vehicle_view = "unknown"

    return {
        "view": vehicle_view,
        "view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "vehicle_view": vehicle_view,
        "vehicle_view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "components": prompted_components,
        "visible_components": [
            {
                "code": item["code"],
                "label": item["label"],
                "category": item["category"],
                "confidence": item["confidence"],
                "mask_base64": item["mask_base64"],
                "bounding_box": item["bounding_box"],
                "mask_source": item["mask_source"],
            }
            for item in prompted_components
        ],
        "prompted_segmentation_failures": prompted_failures,
        "segmentation_strategy": "per_component_prompted_sam2_points",
    }



def call_openai_vehicle_analysis(image_data_url: str) -> dict:
    client = OpenAI()
    configured_model = os.getenv(
        "OPENAI_VISION_MODEL",
        "gpt-4.1-mini",
    )

    allowed_codes = list(VEHICLE_COMPONENT_CATALOG.keys())

    prompt = f"""
Analyze this automotive photograph and segment the visible vehicle components.

Return ONLY a valid JSON object with this exact structure:
{{
  "vehicle_view": one of [
    "front", "front_left", "front_right",
    "rear", "rear_left", "rear_right",
    "left_side", "right_side", "mixed", "unknown"
  ],
  "components": [
    {{
      "code": one of {allowed_codes},
      "confidence": number from 0 to 1,
      "bounding_box": {{
        "x1": integer from 0 to 1000,
        "y1": integer from 0 to 1000,
        "x2": integer from 0 to 1000,
        "y2": integer from 0 to 1000
      }}
    }}
  ]
}}

Bounding-box rules:
- Coordinates are normalized to the complete image: top-left is 0,0 and bottom-right is 1000,1000.
- Draw the smallest practical box containing only the visible portion of the requested physical component.
- Follow the true location of panel seams, lamps, glass, wheels, bumpers and mirrors.
- Do not include floor, workshop equipment, shadows, another vehicle or empty background.
- Do not merge adjacent components into one box.
- Return only components genuinely visible in the photograph.
- A damaged component must still be identified by its original automotive function.
- Do not describe damage severity, people or background.
- Do not invent component codes.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise automotive component segmentation "
                "assistant. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    first_error: Exception | None = None

    # Tentativo 1: JSON mode.
    try:
        response = client.chat.completions.create(
            model=configured_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )

        content = response.choices[0].message.content or "{}"
        parsed = extract_json_object(content)

        if not isinstance(parsed, dict):
            raise ValueError("La risposta JSON non è un oggetto.")

        return parsed

    except Exception as exc:
        first_error = exc
        print(
            "OPENAI VEHICLE SEGMENTATION PRIMARY ERROR:",
            type(exc).__name__,
            str(exc),
        )
        traceback.print_exc()

    # Tentativo 2: fallback senza response_format.
    try:
        response = client.chat.completions.create(
            model=configured_model,
            temperature=0,
            messages=messages,
        )

        content = response.choices[0].message.content or "{}"
        parsed = extract_json_object(content)

        if not isinstance(parsed, dict):
            raise ValueError("La risposta fallback non è un oggetto JSON.")

        return parsed

    except Exception as fallback_exc:
        print(
            "OPENAI VEHICLE SEGMENTATION FALLBACK ERROR:",
            type(fallback_exc).__name__,
            str(fallback_exc),
        )
        traceback.print_exc()

        primary_message = (
            f"{type(first_error).__name__}: {str(first_error)}"
            if first_error is not None
            else "nessun dettaglio"
        )
        fallback_message = (
            f"{type(fallback_exc).__name__}: {str(fallback_exc)}"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Il motore V16 non ha completato la segmentazione "
                    "dei componenti."
                ),
                "model": configured_model,
                "primary_error": primary_message[:800],
                "fallback_error": fallback_message[:800],
                "analysis_version": "vehicle-segmentation-v16.2-prompted-component-segmentation",
            },
        ) from fallback_exc



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_unlink(path_value: str | None) -> None:
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_analysis_jobs() -> None:
    now = datetime.now(timezone.utc).timestamp()

    with ANALYSIS_JOBS_LOCK:
        removable: list[str] = []

        for job_id, job in ANALYSIS_JOBS.items():
            age = now - float(job.get("updated_at_epoch", now))
            if age > ANALYSIS_JOB_TTL_SECONDS:
                removable.append(job_id)

        if len(ANALYSIS_JOBS) - len(removable) > ANALYSIS_JOB_MAX_COUNT:
            remaining = [
                (job_id, float(job.get("updated_at_epoch", 0.0)))
                for job_id, job in ANALYSIS_JOBS.items()
                if job_id not in removable
            ]
            remaining.sort(key=lambda item: item[1])
            overflow = (
                len(ANALYSIS_JOBS)
                - len(removable)
                - ANALYSIS_JOB_MAX_COUNT
            )
            removable.extend(
                job_id for job_id, _ in remaining[:max(0, overflow)]
            )

        for job_id in set(removable):
            job = ANALYSIS_JOBS.pop(job_id, None)
            if job:
                _safe_unlink(job.get("image_path"))


def set_analysis_job(job_id: str, **updates) -> None:
    with ANALYSIS_JOBS_LOCK:
        current = ANALYSIS_JOBS.get(job_id, {})
        current.update(updates)
        current["updated_at"] = utc_now_iso()
        current["updated_at_epoch"] = datetime.now(
            timezone.utc
        ).timestamp()
        ANALYSIS_JOBS[job_id] = current


def get_analysis_job(job_id: str) -> dict | None:
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _persist_analysis_source(
    image_base64: str,
    job_id: str,
) -> tuple[str, tuple[int, int]]:
    source = decode_base64_image(
        image_base64,
        "image_base64",
        "RGB",
    )

    image_path = ANALYSIS_JOB_DIR / f"{job_id}.jpg"
    source.save(
        image_path,
        format="JPEG",
        quality=92,
        subsampling=0,
        optimize=True,
    )
    size = source.size
    del source
    return str(image_path), size


def run_async_vehicle_analysis(job_id: str) -> None:
    job = get_analysis_job(job_id)
    if not job:
        return

    image_path = job.get("image_path")

    try:
        set_analysis_job(
            job_id,
            status="processing",
            progress_stage="loading_image",
            progress_percent=5,
        )

        source = Image.open(image_path).convert("RGB")
        analysis_source, _ = resize_image_for_processing(
            source,
            ANALYSIS_MAX_SIDE,
        )
        image_data_url = image_to_data_url(analysis_source)

        set_analysis_job(
            job_id,
            progress_stage="detecting_components",
            progress_percent=20,
        )
        raw_analysis = call_openai_vehicle_analysis(image_data_url)

        raw_components = raw_analysis.get(
            "components",
            raw_analysis.get("visible_components", []),
        )
        if not isinstance(raw_components, list):
            raw_components = []

        set_analysis_job(
            job_id,
            progress_stage="segmenting_components_with_prompted_sam2",
            progress_percent=40,
        )
        prompted_components, prompted_failures = (
            prompted_segment_components(
                image_data_url,
                source,
                raw_components,
            )
        )

        set_analysis_job(
            job_id,
            progress_stage="building_component_masks",
            progress_percent=88,
        )
        normalized = normalize_vehicle_analysis(
            raw_analysis,
            source,
            prompted_components,
            prompted_failures,
        )

        if not normalized["components"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "SAM 2 guidato non ha prodotto maschere "
                        "utilizzabili per i componenti rilevati."
                    ),
                    "raw_component_count": len(raw_components),
                    "prompted_failure_count": len(prompted_failures),
                    "prompted_failures_preview": prompted_failures[:8],
                    "segmentation_strategy": (
                        "per_component_prompted_sam2_points"
                    ),
                    "analysis_version": (
                        "vehicle-segmentation-v16.2-"
                        "prompted-component-segmentation"
                    ),
                },
            )

        result = {
            **normalized,
            "model": os.getenv(
                "OPENAI_VISION_MODEL",
                "gpt-4.1-mini",
            ),
            "analysis_version": (
                "vehicle-segmentation-v16.2-prompted-component-segmentation"
            ),
            "mask_format": "data:image/png;base64",
            "mask_semantics": "white_component_black_background",
            "segmentation_provider": "replicate/meta-sam-2-video",
            "segmentation_strategy": "per_component_prompted_points",
        }

        set_analysis_job(
            job_id,
            status="succeeded",
            progress_stage="completed",
            progress_percent=100,
            result=result,
            error=None,
        )

    except HTTPException as exc:
        set_analysis_job(
            job_id,
            status="failed",
            progress_stage="failed",
            error={
                "http_status": exc.status_code,
                "detail": exc.detail,
            },
        )
        print(
            "ASYNC VEHICLE ANALYSIS HTTP ERROR:",
            job_id,
            exc.status_code,
            exc.detail,
        )

    except Exception as exc:
        traceback.print_exc()
        set_analysis_job(
            job_id,
            status="failed",
            progress_stage="failed",
            error={
                "http_status": 502,
                "detail": {
                    "message": "Errore inatteso durante l'analisi asincrona.",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1600],
                    "analysis_version": (
                        "vehicle-segmentation-v16.2-prompted-component-segmentation"
                    ),
                },
            },
        )

    finally:
        _safe_unlink(image_path)
        set_analysis_job(job_id, image_path=None)


@app.post("/v1/vehicle/analyze-components/start")
def start_vehicle_component_analysis(
    payload: VehicleAnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    cleanup_analysis_jobs()

    job_id = uuid.uuid4().hex
    image_path, original_size = _persist_analysis_source(
        payload.image_base64,
        job_id,
    )
    now = datetime.now(timezone.utc)

    with ANALYSIS_JOBS_LOCK:
        ANALYSIS_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress_stage": "queued",
            "progress_percent": 0,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "updated_at_epoch": now.timestamp(),
            "image_path": image_path,
            "original_size": {
                "width": original_size[0],
                "height": original_size[1],
            },
            "result": None,
            "error": None,
            "analysis_version": (
                "vehicle-segmentation-v16.2-prompted-component-segmentation"
            ),
        }

    background_tasks.add_task(
        run_async_vehicle_analysis,
        job_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "progress_stage": "queued",
        "progress_percent": 0,
        "poll_url": (
            f"/v1/vehicle/analyze-components/status/{job_id}"
        ),
        "analysis_version": (
            "vehicle-segmentation-v16.2-prompted-component-segmentation"
        ),
    }


@app.get("/v1/vehicle/analyze-components/status/{job_id}")
def get_vehicle_component_analysis_status(job_id: str):
    cleanup_analysis_jobs()
    job = get_analysis_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Job di analisi non trovato o scaduto.",
                "job_id": job_id,
            },
        )

    response = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress_stage": job.get("progress_stage"),
        "progress_percent": job.get("progress_percent", 0),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "analysis_version": job.get("analysis_version"),
    }

    if job.get("status") == "succeeded":
        response["result"] = job.get("result")

    if job.get("status") == "failed":
        response["error"] = job.get("error")

    return response


@app.delete("/v1/vehicle/analyze-components/status/{job_id}")
def delete_vehicle_component_analysis_job(job_id: str):
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.pop(job_id, None)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Job di analisi non trovato.",
                "job_id": job_id,
            },
        )

    _safe_unlink(job.get("image_path"))

    return {
        "job_id": job_id,
        "deleted": True,
    }


@app.post("/v1/vehicle/analyze-components")
def analyze_vehicle_components_sync_disabled(
    payload: VehicleAnalyzeRequest,
):
    raise HTTPException(
        status_code=409,
        detail={
            "message": (
                "La V16.1.2 usa l'analisi asincrona per evitare timeout. "
                "Usare POST /v1/vehicle/analyze-components/start e poi "
                "GET /v1/vehicle/analyze-components/status/{job_id}."
            ),
            "start_endpoint": (
                "/v1/vehicle/analyze-components/start"
            ),
            "status_endpoint_template": (
                "/v1/vehicle/analyze-components/status/{job_id}"
            ),
            "analysis_version": (
                "vehicle-segmentation-v16.2-prompted-component-segmentation"
            ),
        },
    )


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
        severity=severity_percent,
        area=area_percent,
        damage_mode="bodywork",
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
        "prompt_version": "damage-v16.2-prompted-component-segmentation",
    }


@app.post("/v1/damage/edit-base64")
def edit_damage_base64(payload: DamageEditBase64Request):
    if payload.component_masks_base64 is None:
        payload = copy_request_model(
            payload,
            {"component_masks_base64": {}},
        )

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

    job_id = str(uuid.uuid4())
    resolved_damage_mode = infer_damage_mode(payload)

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        adjusted_mock_mask = apply_area_percent_to_edit_mask(
            source_mask,
            area_percent,
        )
        api_mock_mask = alter_damage_area(adjusted_mock_mask, 0)
        return make_mock_result(
            source,
            api_mock_mask,
            job_id,
            severity_percent,
            area_percent,
        )

    if (
        severity_percent == 0
        and area_percent == 0
        and payload.deformation_type == "dent"
        and not payload.contact_traces_enabled
        and not has_component_damage_request(payload)
    ):
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=95, subsampling=0)
        result_bytes = buffer.getvalue()

    elif resolved_damage_mode != "mixed":
        # Modalità singola: controllo geometrico reale della superficie.
        effective_mask = apply_area_percent_to_edit_mask(
            source_mask,
            area_percent,
        )
        api_mask = alter_damage_area(effective_mask, 0)

        prompt = build_prompt(
            severity=severity_percent,
            area=area_percent,
            damage_mode=resolved_damage_mode,
            deformation_type=payload.deformation_type,
            impact_direction=payload.impact_direction,
            component_instructions=build_component_instructions(payload),
            contact_trace_instructions=build_contact_trace_instructions(payload),
        )

        generated_bytes = call_openai_image_edit(
            source,
            api_mask,
            prompt,
            payload.output_quality,
        )

        result_bytes = smart_component_composite(
            source=source,
            generated_bytes=generated_bytes,
            source_mask=effective_mask,
            protect_mask=protect_mask,
            expansion_px=area_transition_expansion_px(area_percent),
            feather_px=area_transition_feather_px(
                source.size,
                area_percent,
            ),
        )

    else:
        # Modalità mista evoluta: un passaggio per la lamiera e uno per ogni
        # componente non-lamiera, con maschere indipendenti.
        bodywork_codes = selected_bodywork_component_codes(payload)
        component_codes = selected_non_body_component_codes(payload)

        if bodywork_codes and not payload.bodywork_mask_base64:
            raise HTTPException(
                status_code=422,
                detail=(
                    "La modalità mixed richiede bodywork_mask_base64 per separare "
                    "la lamiera dagli altri componenti."
                ),
            )

        missing_component_masks = [
            code
            for code in component_codes
            if not (payload.component_masks_base64 or {}).get(code)
        ]
        if missing_component_masks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Mancano le maschere separate per: "
                    + ", ".join(missing_component_masks)
                ),
            )

        current_image = source
        sequential_steps: list[str] = []

        # Passaggio 1: sola lamiera.
        if bodywork_codes:
            raw_bodywork_mask = decode_base64_image(
                payload.bodywork_mask_base64,
                "bodywork_mask_base64",
                "L",
            )
            bodywork_mask, _ = combine_edit_and_protect_masks(
                raw_bodywork_mask,
                raw_protect_mask,
                source.size,
            )
            effective_bodywork_mask = apply_area_percent_to_edit_mask(
                bodywork_mask,
                area_percent,
            )
            api_bodywork_mask = alter_damage_area(
                effective_bodywork_mask,
                0,
            )

            bodywork_payload = copy_request_model(
                payload,
                {
                    "damage_mode": "bodywork",
                    "involved_components": {
                        code: True for code in bodywork_codes
                    },
                    "component_damage_types": {
                        code: damage_type
                        for code, damage_type in (
                            payload.component_damage_types or {}
                        ).items()
                        if code in bodywork_codes
                    },
                    "bodywork_mask_base64": None,
                    "component_masks_base64": {},
                }
            )

            bodywork_prompt = build_prompt(
                severity=severity_percent,
                area=area_percent,
                damage_mode="bodywork",
                deformation_type=payload.deformation_type,
                impact_direction=payload.impact_direction,
                component_instructions=build_component_instructions(
                    bodywork_payload
                ),
                contact_trace_instructions=build_contact_trace_instructions(
                    bodywork_payload
                ),
            )

            generated_bodywork = call_openai_image_edit(
                current_image,
                api_bodywork_mask,
                bodywork_prompt,
                payload.output_quality,
            )

            bodywork_result = smart_component_composite(
                source=current_image,
                generated_bytes=generated_bodywork,
                source_mask=effective_bodywork_mask,
                protect_mask=protect_mask,
                expansion_px=area_transition_expansion_px(area_percent),
                feather_px=area_transition_feather_px(
                    current_image.size,
                    area_percent,
                ),
            )
            current_image = jpeg_bytes_to_rgb_image(bodywork_result)
            sequential_steps.append("bodywork")

        # Passaggi successivi: un componente per volta.
        for component_code in component_codes:
            raw_component_mask = decode_base64_image(
                (payload.component_masks_base64 or {})[component_code],
                f"component_masks_base64[{component_code}]",
                "L",
            )
            component_mask, _ = combine_edit_and_protect_masks(
                raw_component_mask,
                raw_protect_mask,
                source.size,
            )

            # La superficie danneggiata riguarda la lamiera: non altera
            # geometricamente fari, vetri, specchietti, ruote o paraurti.
            api_component_mask = alter_damage_area(component_mask, 0)

            component_payload = payload_for_single_component(
                payload,
                component_code,
            )

            component_prompt = build_prompt(
                severity=severity_percent,
                area=0,
                damage_mode="component_only",
                deformation_type=payload.deformation_type,
                impact_direction=payload.impact_direction,
                component_instructions=build_component_instructions(
                    component_payload
                ),
                contact_trace_instructions=build_contact_trace_instructions(
                    component_payload
                ),
            )

            generated_component = call_openai_image_edit(
                current_image,
                api_component_mask,
                component_prompt,
                payload.output_quality,
            )

            component_result = smart_component_composite(
                source=current_image,
                generated_bytes=generated_component,
                source_mask=component_mask,
                protect_mask=protect_mask,
            )
            current_image = jpeg_bytes_to_rgb_image(component_result)
            sequential_steps.append(component_code)

        output = io.BytesIO()
        current_image.save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        result_bytes = output.getvalue()

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v16.2-prompted-component-segmentation",
        "deformation_type": payload.deformation_type,
        "impact_direction": payload.impact_direction,
        "contact_traces_enabled": payload.contact_traces_enabled,
        "involved_components": payload.involved_components,
        "damage_mode": resolved_damage_mode,
        "area_control": "geometric_mask_transform_smart_composite",
        "area_transition_feather_px": area_transition_feather_px(
            source.size,
            area_percent,
        ),
        "composite_strategy": "lab_color_match_multiband",
        "segmentation_contract": "per_component_png_masks",
        "mixed_strategy": (
            "sequential_per_component"
            if resolved_damage_mode == "mixed"
            else "single_pass"
        ),
        "sequential_steps": (
            sequential_steps
            if resolved_damage_mode == "mixed"
            else []
        ),
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
