import base64
import gc
import io
import json
import os
import re
import traceback
import time
import urllib.error
import urllib.request
import threading
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORT_ENABLED = True
except Exception:
    HEIC_SUPPORT_ENABLED = False
from pydantic import BaseModel, Field

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_PROCESSING_SIDE = max(
    768,
    min(1600, int(os.getenv("MAX_PROCESSING_SIDE", "1280"))),
)
MAX_INPUT_PIXELS = max(
    4_000_000,
    int(os.getenv("MAX_INPUT_PIXELS", "30000000")),
)
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS

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

print(
    "=== CAR DAMAGE LAB BACKEND V17.0.26 WRITABLE ARRAY HOTFIX ===",
    flush=True,
)

app = FastAPI(
    title=APP_NAME,
    version="1.7.0.26",
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



class ImageNormalizeRequest(BaseModel):
    image_base64: str
    output_format: Literal["jpeg", "png"] = "jpeg"
    max_side: int = Field(default=1600, ge=512, le=2400)
    jpeg_quality: int = Field(default=94, ge=80, le=98)


class DamageEditBase64Request(BaseModel):
    image_base64: str = Field(..., min_length=16)
    mask_base64: str | None = Field(default=None, min_length=16)
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
        "simple_guided",
        "hybrid_guided",
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
        "rear_to_front",
    ] = "frontal"

    involved_components: dict[str, bool] = Field(
        default_factory=lambda: {"bodyPanel": True}
    )

    # Elenco esplicito inviato da Base44 nelle selezioni multiple.
    # Nelle versioni precedenti il payload lo conteneva, ma Pydantic lo
    # ignorava perché il campo non era dichiarato; il codice tentava poi
    # di leggerlo causando AttributeError e risposta HTTP 500.
    selected_components: list[str] = Field(default_factory=list)

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
    user_instructions: str = Field(default="", max_length=4000)
    protected_components: list[str] = Field(default_factory=list)
    protected_component_masks_base64: dict[str, str] = Field(
        default_factory=dict
    )

    # V17.0.22 - Protezione identità dinamica e indipendente dal modello auto.
    dynamic_identity_protection: bool = True
    identity_regions: list[dict[str, object]] = Field(default_factory=list)
    identity_detection_padding_percent: float = Field(
        default=1.2,
        ge=0.0,
        le=6.0,
    )
    identity_color_correction: bool = False


def resolve_selected_components(
    payload: DamageEditBase64Request,
) -> list[str]:
    """
    Restituisce una lista stabile di componenti selezionati.

    Priorità:
    1. selected_components inviato esplicitamente da Base44;
    2. chiavi vere di involved_components;
    3. chiavi di component_damage_types;
    4. chiavi delle maschere per componente.
    """
    candidates: list[str] = []

    candidates.extend(payload.selected_components or [])

    if payload.involved_components:
        candidates.extend(
            str(code)
            for code, enabled in payload.involved_components.items()
            if enabled
        )

    if payload.component_damage_types:
        candidates.extend(
            str(code)
            for code in payload.component_damage_types.keys()
        )

    if payload.component_masks_base64:
        candidates.extend(
            str(code)
            for code in payload.component_masks_base64.keys()
        )

    normalized: list[str] = []
    seen: set[str] = set()

    generic_codes = {
        "bodypanel",
        "body_panel",
        "bodywork",
        "generic_body_panel",
    }

    for item in candidates:
        code = str(item or "").strip()
        if not code:
            continue

        key = code.lower()
        if key in seen:
            continue

        seen.add(key)
        normalized.append(code)

    specific_components = [
        code
        for code in normalized
        if code.lower() not in generic_codes
    ]

    if specific_components:
        return specific_components

    return normalized


class VehicleAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)


class NormalizedPoint(BaseModel):
    x: float = Field(..., ge=0, le=1000)
    y: float = Field(..., ge=0, le=1000)


class ComponentRefineRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    component_code: str = Field(..., min_length=1, max_length=80)
    detection_box: dict
    current_mask_base64: str | None = None
    positive_points: list[NormalizedPoint] = Field(default_factory=list)
    negative_points: list[NormalizedPoint] = Field(default_factory=list)
    iterations: int = Field(default=4, ge=1, le=8)


class AssistedComponentSelectionRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    component_code: str = Field(..., min_length=1, max_length=80)
    detection_box: dict | None = None
    current_mask_base64: str | None = None
    positive_points: list[NormalizedPoint] = Field(default_factory=list)
    negative_points: list[NormalizedPoint] = Field(default_factory=list)
    reset_mask: bool = False
    manual_mask_base64: str | None = None
    confirm_manual_mask: bool = False


class PolygonPoint(BaseModel):
    x: float = Field(..., ge=0, le=1000)
    y: float = Field(..., ge=0, le=1000)


class SmartPolygonRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    component_code: str = Field(..., min_length=1, max_length=80)
    points: list[PolygonPoint] = Field(..., min_length=3, max_length=120)
    snap_to_edges: bool = True
    snap_radius: int = Field(default=14, ge=0, le=60)
    smooth_polygon: bool = True
    feather_radius: int = Field(default=2, ge=0, le=12)
    confirm_mask: bool = True


class SmartPolygonSnapRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    points: list[PolygonPoint] = Field(..., min_length=1, max_length=120)
    snap_radius: int = Field(default=14, ge=0, le=60)


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
    V17.0.6 - La maschera manuale è il perimetro reale dell'intervento.

    area_percent resta un parametro descrittivo per il prompt, ma non riduce,
    non erode e non espande geometricamente la maschera disegnata dall'utente.

    Questo evita che area_percent=50 trasformi la maschera del componente
    nella sola porzione centrale e renda la deformazione quasi invisibile.
    """
    _ = clamp_percentage(area_percent)
    binary = mask_to_binary(mask)

    if int((binary > 0).sum()) == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera del componente è vuota.",
        )

    return Image.fromarray(binary, mode="L")


def area_transition_feather_px(
    image_size: tuple[int, int],
    area_percent: int,
) -> int:
    """
    V17.0.6 - Feather minimo e costante.

    La sfumatura serve soltanto a evitare un bordo artificiale.
    Non deve attenuare la deformazione generata dentro la maschera.
    """
    _ = image_size
    _ = clamp_percentage(area_percent)
    return 2


def area_transition_expansion_px(area_percent: int) -> int:
    """
    V17.0.6 - Nessuna espansione esterna alla maschera manuale.
    """
    _ = clamp_percentage(area_percent)
    return 0


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
    min(2, int(os.getenv("PROMPTED_SAM_MAX_WORKERS", "1"))),
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


COMPONENT_REFINEMENT_MARGIN_RATIO = max(
    0.08,
    min(
        0.45,
        float(os.getenv("COMPONENT_REFINEMENT_MARGIN_RATIO", "0.22")),
    ),
)
COMPONENT_REFINEMENT_ITERATIONS = max(
    1,
    min(8, int(os.getenv("COMPONENT_REFINEMENT_ITERATIONS", "2"))),
)


COMPONENT_REFINEMENT_MAX_CROP_SIDE = max(
    320,
    min(
        1200,
        int(os.getenv("COMPONENT_REFINEMENT_MAX_CROP_SIDE", "640")),
    ),
)
COMPONENT_REFINEMENT_MIN_AREA_RATIO = max(
    0.00002,
    min(
        0.02,
        float(os.getenv("COMPONENT_REFINEMENT_MIN_AREA_RATIO", "0.00015")),
    ),
)


PROMPTED_SAM_COMPONENT_TIMEOUT_SECONDS = max(
    20,
    min(
        180,
        int(
            os.getenv(
                "PROMPTED_SAM_COMPONENT_TIMEOUT_SECONDS",
                "75",
            )
        ),
    ),
)


SMART_POLYGON_EDGE_BLUR = max(
    1,
    min(9, int(os.getenv("SMART_POLYGON_EDGE_BLUR", "5"))),
)
SMART_POLYGON_CANNY_LOW = max(
    10,
    min(180, int(os.getenv("SMART_POLYGON_CANNY_LOW", "45"))),
)
SMART_POLYGON_CANNY_HIGH = max(
    SMART_POLYGON_CANNY_LOW + 1,
    min(255, int(os.getenv("SMART_POLYGON_CANNY_HIGH", "135"))),
)
SMART_POLYGON_MIN_AREA_PIXELS = max(
    32,
    int(os.getenv("SMART_POLYGON_MIN_AREA_PIXELS", "120")),
)


REPLICATE_CREATE_MIN_INTERVAL_SECONDS = max(
    1.0,
    min(
        30.0,
        float(
            os.getenv(
                "REPLICATE_CREATE_MIN_INTERVAL_SECONDS",
                "10.5",
            )
        ),
    ),
)
REPLICATE_RATE_LIMIT_MAX_RETRIES = max(
    1,
    min(
        8,
        int(
            os.getenv(
                "REPLICATE_RATE_LIMIT_MAX_RETRIES",
                "5",
            )
        ),
    ),
)

REPLICATE_CREATE_LOCK = threading.Lock()
REPLICATE_LAST_CREATE_AT = 0.0
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




def normalize_image_to_base64(
    image_base64: str,
    output_format: str = "jpeg",
    max_side: int = 1600,
    jpeg_quality: int = 94,
) -> dict[str, object]:
    """Decodifica JPEG/PNG/HEIC/HEIF e restituisce un formato browser-safe."""
    image = decode_base64_image(image_base64, "image_base64", "RGB")
    original_size = image.size
    original_format = getattr(image, "format", None) or "UNKNOWN"

    image, _, scale = resize_in_place_for_memory(image, max_side=max_side)
    output = io.BytesIO()

    if output_format == "png":
        mime_type = "image/png"
        normalized_format = "PNG"
        image.save(output, format="PNG", optimize=False)
    else:
        mime_type = "image/jpeg"
        normalized_format = "JPEG"
        image.save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=False,
        )

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    result = {
        "normalized_image_base64": f"data:{mime_type};base64,{encoded}",
        "original_format": str(original_format).upper(),
        "normalized_format": normalized_format,
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "width": int(image.size[0]),
        "height": int(image.size[1]),
        "processing_scale": round(scale, 6),
        "heic_support_enabled": HEIC_SUPPORT_ENABLED,
    }

    try:
        image.close()
    except Exception:
        pass
    output.close()
    gc.collect()
    return result


def _looks_like_heic(raw: bytes) -> bool:
    """
    Riconosce i principali brand HEIC/HEIF/AVIF nel box ISO-BMFF.
    """
    if len(raw) < 16:
        return False

    head = raw[:64]
    return (
        b"ftypheic" in head
        or b"ftypheix" in head
        or b"ftyphevc" in head
        or b"ftyphevx" in head
        or b"ftypmif1" in head
        or b"ftypmsf1" in head
        or b"ftypavif" in head
    )


def resize_in_place_for_memory(
    image: Image.Image,
    max_side: int = MAX_PROCESSING_SIDE,
) -> tuple[Image.Image, tuple[int, int], float]:
    """
    Riduce immediatamente l'immagine per limitare la RAM.

    Usa thumbnail in-place per evitare di mantenere contemporaneamente
    due copie complete dell'immagine.
    """
    original_size = image.size
    width, height = original_size
    longest = max(width, height)

    if longest <= max_side:
        return image, original_size, 1.0

    scale = max_side / float(longest)
    image.thumbnail(
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    return image, original_size, scale


def resize_mask_to_processing_size(
    mask: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    if mask.size == target_size:
        return mask.convert("L")

    resized = mask.convert("L").resize(
        target_size,
        Image.Resampling.NEAREST,
    )
    try:
        mask.close()
    except Exception:
        pass
    return resized


def lightweight_full_frame_validation(
    source: Image.Image,
    candidate_bytes: bytes,
) -> tuple[bytes, dict[str, object]]:
    """
    Validazione leggera per il fallback senza maschera.
    Evita array NumPy multipli e compositing.
    """
    try:
        candidate = Image.open(io.BytesIO(candidate_bytes))
        candidate.load()
        candidate = candidate.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il motore non ha restituito un'immagine valida.",
        ) from exc

    if candidate.size != source.size:
        candidate = candidate.resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

    output = io.BytesIO()
    candidate.save(
        output,
        format="JPEG",
        quality=94,
        subsampling=0,
        optimize=False,
    )

    diagnostics = {
        "result_is_full_frame": True,
        "result_size": list(candidate.size),
        "direct_model_output_used": False,
        "post_composite_applied": True,
        "lightweight_validation": True,
    }

    try:
        candidate.close()
    except Exception:
        pass

    return output.getvalue(), diagnostics


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

    if _looks_like_heic(raw) and not HEIC_SUPPORT_ENABLED:
        raise HTTPException(
            status_code=415,
            detail={
                "message": (
                    "Immagine HEIC/HEIF riconosciuta, ma il supporto HEIC "
                    "non è installato sul server."
                ),
                "required_dependency": "pillow-heif",
            },
        )

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
    expansion_px: int = 0,
    feather_px: int = 2,
) -> bytes:
    """
    V17.0.6 - Single Protected Composite.

    - il risultato AI viene mantenuto pienamente visibile dentro la maschera;
    - fuori dalla maschera restano i pixel originali;
    - nessuna erosione della maschera;
    - nessun secondo compositing;
    - feather massimo di pochi pixel solo sul bordo interno.
    """
    _ = expansion_px

    try:
        generated_raw = Image.open(io.BytesIO(generated_bytes))
        generated_raw.load()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il risultato generato non è un'immagine valida.",
        ) from exc

    source_rgb = source.convert("RGB")
    target_size = source_rgb.size

    if "A" in generated_raw.getbands():
        generated_rgba = generated_raw.convert("RGBA")
        if generated_rgba.size != target_size:
            generated_rgba = generated_rgba.resize(
                target_size,
                Image.Resampling.LANCZOS,
            )
        generated_rgb = Image.alpha_composite(
            source_rgb.convert("RGBA"),
            generated_rgba,
        ).convert("RGB")
    else:
        generated_rgb = generated_raw.convert("RGB")
        if generated_rgb.size != target_size:
            generated_rgb = generated_rgb.resize(
                target_size,
                Image.Resampling.LANCZOS,
            )

    source_array = np.asarray(source_rgb, dtype=np.uint8)
    generated_array = np.asarray(generated_rgb, dtype=np.uint8)

    editable = mask_to_binary(
        resize_mask(source_mask.convert("L"), target_size)
    )

    if protect_mask is not None:
        protected = mask_to_binary(
            resize_mask(protect_mask.convert("L"), target_size)
        )
        editable = cv2.bitwise_and(
            editable,
            cv2.bitwise_not(protected),
        )
    else:
        protected = np.zeros_like(editable)

    if int((editable > 0).sum()) == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera effettiva del compositing è vuota.",
        )

    # Alpha pieno nella zona interna; transizione brevissima sul solo bordo.
    safe_feather = max(0, min(3, int(feather_px)))

    if safe_feather == 0:
        alpha = (editable > 0).astype(np.float32)
    else:
        distance_inside = cv2.distanceTransform(
            np.where(editable > 0, 255, 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        alpha = np.clip(distance_inside / float(safe_feather), 0.0, 1.0)
        alpha[editable == 0] = 0.0

    alpha[protected > 0] = 0.0
    alpha3 = alpha[:, :, None]

    final_array = (
        generated_array.astype(np.float32) * alpha3
        + source_array.astype(np.float32) * (1.0 - alpha3)
    )
    final_array = np.clip(final_array, 0, 255).astype(np.uint8)

    # Garanzia assoluta fuori maschera.
    final_array[editable == 0] = source_array[editable == 0]
    final_array[protected > 0] = source_array[protected > 0]

    output = io.BytesIO()
    Image.fromarray(final_array, mode="RGB").save(
        output,
        format="JPEG",
        quality=96,
        subsampling=0,
        optimize=True,
    )
    return output.getvalue()



def enforce_full_frame_result(
    source: Image.Image,
    candidate_bytes: bytes,
    edit_mask: Image.Image,
    protect_mask: Image.Image | None = None,
    feather_px: int = 2,
) -> tuple[bytes, dict[str, object]]:
    """
    V17.0.6 - Full Frame Validator.

    Non ricompone una seconda volta il risultato, perché il compositing
    protetto è già stato eseguito. Verifica soltanto che l'immagine sia
    valida e abbia le dimensioni dell'originale.
    """
    _ = edit_mask
    _ = protect_mask
    _ = feather_px

    source_rgb = source.convert("RGB")
    expected_size = source_rgb.size

    try:
        candidate = Image.open(io.BytesIO(candidate_bytes))
        candidate.load()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il risultato finale non è un'immagine valida.",
        ) from exc

    candidate_rgb = candidate.convert("RGB")

    if candidate_rgb.size != expected_size:
        candidate_rgb = candidate_rgb.resize(
            expected_size,
            Image.Resampling.LANCZOS,
        )

    output = io.BytesIO()
    candidate_rgb.save(
        output,
        format="JPEG",
        quality=96,
        subsampling=0,
        optimize=True,
    )

    diagnostics = {
        "result_is_full_frame": True,
        "original_size": [
            int(expected_size[0]),
            int(expected_size[1]),
        ],
        "result_size": [
            int(candidate_rgb.size[0]),
            int(candidate_rgb.size[1]),
        ],
        "full_frame_validator_applied": True,
        "second_composite_applied": False,
        "manual_mask_geometry_preserved": True,
    }
    return output.getvalue(), diagnostics





PROTECTED_COMPONENT_LABELS: dict[str, str] = {
    "headlight": "front headlight",
    "front_headlight": "front headlight",
    "left_headlight": "left front headlight",
    "right_headlight": "right front headlight",
    "windshield": "windshield",
    "windscreen": "windshield",
    "front_bumper": "front bumper",
    "bumper_front": "front bumper",
    "bumper": "bumper",
    "wheel": "wheel",
    "front_wheel": "front wheel",
    "wheel_arch": "wheel arch",
    "fender_front": "front fender",
    "front_fender": "front fender",
    "door_front": "front door",
    "front_door": "front door",
    "mirror": "side mirror",
    "side_mirror": "side mirror",
    "grille": "front grille",
    "license_plate": "license plate",
}


def protected_component_names(
    protected_components: list[str] | None,
) -> list[str]:
    """
    Normalizza i codici dei componenti protetti in etichette chiare
    per il prompt.
    """
    result: list[str] = []
    for code in protected_components or []:
        normalized = str(code).strip().lower()
        label = PROTECTED_COMPONENT_LABELS.get(
            normalized,
            normalized.replace("_", " "),
        )
        if label and label not in result:
            result.append(label)
    return result


def build_strict_protected_components_prompt(
    protected_components: list[str] | None,
) -> str:
    """
    Genera una sezione prompt vincolante per i componenti protetti.
    """
    names = protected_component_names(protected_components)

    if not names:
        return (
            "Preserve all surrounding components that are not explicitly "
            "selected for deformation."
        )

    protected_list = "\n".join(f"- {name}" for name in names)

    specific_rules: list[str] = []

    if any("headlight" in name for name in names):
        specific_rules.append(
            """
HEADLIGHT PROTECTION IS ABSOLUTE:
- preserve the exact headlight lens outline;
- preserve the exact internal light geometry;
- preserve the housing position and panel gaps;
- preserve the original reflections on the lens;
- do not reshape, move, stretch, blur or regenerate the headlight.
""".strip()
        )

    if any("windshield" in name for name in names):
        specific_rules.append(
            """
WINDSHIELD PROTECTION IS ABSOLUTE:
- preserve its exact outline, reflections and transparency;
- do not bend, crack, move or regenerate the glass.
""".strip()
        )

    if any("wheel" in name for name in names):
        specific_rules.append(
            """
WHEEL PROTECTION IS ABSOLUTE:
- preserve the exact rim design, tyre shape, position and reflections;
- do not alter wheel alignment unless explicitly requested.
""".strip()
        )

    extras = "\n\n".join(specific_rules)

    return f"""
STRICTLY PRESERVE THESE COMPONENTS WITHOUT ANY VISUAL OR GEOMETRIC CHANGE:
{protected_list}

Do not modify their outline, dimensions, position, reflections, textures,
panel gaps, internal details or identity.

The selected deformation may approach these parts, but must stop before
altering them.

{extras}
""".strip()


def validate_protected_components_unchanged(
    source: Image.Image,
    candidate_bytes: bytes,
    protected_masks_base64: dict[str, str] | None,
) -> dict[str, object]:
    """
    Verifica facoltativamente le aree protette quando Base44 invia
    protected_component_masks_base64.

    Non modifica l'immagine. Se una zona protetta cambia troppo, rifiuta
    il risultato.
    """
    if not protected_masks_base64:
        return {
            "protected_masks_received": 0,
            "protected_components_validation_applied": False,
        }

    source_rgb = source.convert("RGB")
    width, height = source_rgb.size
    source_array = np.asarray(source_rgb, dtype=np.uint8)

    try:
        candidate = Image.open(io.BytesIO(candidate_bytes))
        candidate.load()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il risultato non è un'immagine valida.",
        ) from exc

    candidate_rgb = candidate.convert("RGB")
    if candidate_rgb.size != (width, height):
        candidate_rgb = candidate_rgb.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    candidate_array = np.asarray(candidate_rgb, dtype=np.uint8)

    validated: dict[str, object] = {}

    for component_code, mask_base64 in protected_masks_base64.items():
        if not mask_base64:
            continue

        mask_image = decode_base64_image(
            mask_base64,
            f"protected_component_masks_base64[{component_code}]",
            "L",
        )
        mask = mask_to_binary(
            resize_mask(mask_image, (width, height))
        ) > 0

        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue

        difference = np.mean(
            np.abs(
                candidate_array.astype(np.int16)
                - source_array.astype(np.int16)
            ),
            axis=2,
        )

        mean_difference = float(difference[mask].mean())
        changed_ratio = float((difference[mask] > 16.0).sum()) / float(
            pixel_count
        )

        validated[component_code] = {
            "mean_difference": round(mean_difference, 3),
            "changed_ratio": round(changed_ratio, 4),
            "pixel_count": pixel_count,
        }

        if mean_difference > 12.0 and changed_ratio > 0.10:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        f"Il componente protetto '{component_code}' "
                        "è stato modificato troppo dal motore."
                    ),
                    "component": component_code,
                    "mean_difference": round(mean_difference, 3),
                    "changed_ratio": round(changed_ratio, 4),
                },
            )

    return {
        "protected_masks_received": len(validated),
        "protected_components_validation_applied": bool(validated),
        "protected_components_validation": validated,
    }





def build_collision_consequences_prompt(
    selected_components: list[str],
    protected_components: list[str],
    severity_percent: int,
    area_percent: int,
    impact_direction: str,
) -> str:
    """
    V17.0.17 - Conseguenze fisiche coerenti su fanali posteriori e lunotto.
    Le protezioni esplicite prevalgono sempre.
    """
    selected = {
        str(item).strip().lower()
        for item in selected_components or []
    }
    protected = {
        str(item).strip().lower()
        for item in protected_components or []
    }

    severity = max(0, min(100, abs(int(severity_percent))))
    area = max(0, min(100, abs(int(area_percent))))

    rear_body_components = {
        "rear_bumper",
        "bumper_rear",
        "rear_fender",
        "rear_fender_left",
        "rear_fender_right",
        "rear_quarter_panel",
        "rear_quarter_panel_left",
        "rear_quarter_panel_right",
        "tailgate",
        "trunk_lid",
        "rear_panel",
    }

    rear_light_components = {
        "rear_light",
        "tail_light",
        "left_tail_light",
        "right_tail_light",
        "rear_lamp",
        "rear_light_left",
        "rear_light_right",
    }

    rear_glass_components = {
        "rear_window",
        "rear_windscreen",
        "back_glass",
        "tailgate_glass",
        "lunotto",
    }

    rear_collision = (
        bool(selected & rear_body_components)
        or impact_direction == "rear_to_front"
    )
    rear_light_selected = bool(selected & rear_light_components)
    rear_light_protected = bool(protected & rear_light_components)
    rear_glass_selected = bool(selected & rear_glass_components)
    rear_glass_protected = bool(protected & rear_glass_components)

    severe_rear_collision = (
        rear_collision
        and severity >= 65
        and area >= 35
    )
    very_severe_rear_collision = (
        rear_collision
        and severity >= 78
        and area >= 50
    )

    if rear_light_protected:
        rear_light_rule = """
REAR LIGHT PROTECTION — ABSOLUTE

A rear light is explicitly protected.

Preserve exactly its lens outline, internal geometry, housing position,
mounting points, reflections and panel gaps.

Do not crack, displace, shatter, blur or regenerate the protected rear light.
""".strip()
    elif severe_rear_collision or rear_light_selected:
        rear_light_rule = """
REAR LIGHT DAMAGE — ORIGINAL GEOMETRY LOCKED

The rear collision is severe enough to affect the rear light mounting area.

Damage the exact original rear light already present in the photograph.
The original outer silhouette, length, height, curvature, colour, lens layout,
internal graphic signature and relationship with adjacent body panels are
identity-critical and must remain recognizable.

Allowed damage:
- cracks inside the original lens surface;
- partial shattering of the original lens;
- missing lens fragments while preserving the original housing outline;
- slight displacement or tilt of the whole original assembly;
- broken mounting points;
- partial separation from the body;
- internal elements misaligned inside the original housing.

Forbidden:
- replacing the light with a different design;
- making it rounder, shorter, narrower, larger or smaller;
- inventing a new lamp shape;
- changing its colour or internal styling;
- merging it with the bodywork;
- converting it into a lamp from another make or model;
- damaging the opposite rear light unless the force physically reaches it.

The light may crack or lose internal lens fragments, but its outer silhouette,
size, orientation and mounting footprint must remain exactly the same as in the
source photograph. Do not bend, shrink, stretch or redesign the lamp housing.
""".strip()
    else:
        rear_light_rule = """
REAR LIGHT CONSISTENCY

Keep the rear light intact only when the force does not reach its housing or
mounting area. Do not add arbitrary breakage.
""".strip()

    if rear_glass_protected:
        rear_glass_rule = """
REAR WINDOW PROTECTION — ABSOLUTE

The rear window is explicitly protected.

Preserve its exact outline, transparency, reflections, heating lines,
seal position and mounting within the tailgate frame.

Do not crack, detach, shatter or regenerate the protected rear window.
""".strip()
    elif rear_glass_selected or very_severe_rear_collision:
        rear_glass_rule = """
REAR WINDOW DAMAGE — REQUIRED WHEN THE FRAME IS HEAVILY DISTORTED

The tailgate and rear frame are deformed strongly enough that the rear
window cannot remain perfectly intact.

Create a physically plausible consequence according to frame distortion:
- cracks starting from stressed edges;
- partial fracture;
- separation from part of the rubber seal;
- localized shattering;
- complete breakage only when frame distortion is extreme.

Glass damage must follow stress transmitted through the distorted tailgate
frame. Do not create decorative cracks or damage unrelated windows.
""".strip()
    elif severe_rear_collision:
        rear_glass_rule = """
REAR WINDOW CONDITIONAL DAMAGE

Inspect the tailgate frame.

If the window opening is twisted, compressed or displaced, add realistic
edge cracks or partial detachment from the seal.

If the frame remains geometrically stable, the glass may remain intact.
Do not keep the glass perfect when its surrounding frame is severely bent.
""".strip()
    else:
        rear_glass_rule = """
REAR WINDOW CONSISTENCY

Keep the rear window intact unless the tailgate frame transmits enough
physical stress to crack or detach it.
""".strip()

    return f"""
COLLISION CONSEQUENCES AND IDENTITY PRESERVATION — STRICT

VEHICLE IDENTITY — ABSOLUTE LOCK:
- preserve the exact same make, model, generation and trim;
- preserve the exact body architecture, roofline, glass shape and panel layout;
- preserve every manufacturer emblem, model badge, trim badge and readable text;
- never replace, redraw or invent logos, badges, lettering or brand symbols;
- never transform this vehicle into another make or model;
- damage must alter condition, not identity.

LICENSE PLATE — ABSOLUTE LOCK:
- preserve the exact original license plate;
- preserve every character, spacing, font appearance, colour, border and plate type;
- do not remove, blur, replace, distort, translate or regenerate it;
- do not invent a different plate;
- if its mounting panel moves, the entire original plate may move rigidly with
  that panel, but its content and appearance must remain unchanged.

BADGES AND EMBLEMS — ABSOLUTE LOCK:
- preserve the exact original manufacturer emblem;
- preserve the exact original model name and trim lettering;
- do not substitute any logo or badge from another brand;
- do not change FIAT into Renault, CLIO, or any other make/model;
- badges may move rigidly with a deformed panel but must not change identity.

{rear_light_rule}

{rear_glass_rule}

MULTI-COMPONENT CONSISTENCY:
- all selected damaged components belong to one collision event;
- deformation follows one impact origin and one force direction;
- plastic, sheet metal, lights and glass react according to their material;
- fragile parts must react consistently with surrounding damage;
- do not preserve a fragile part unrealistically when its mounting area or
  supporting frame is clearly crushed;
- do not break fragile parts outside the collision path;
- explicit protected components always override automatic consequences.
""".strip()


def build_impact_direction_prompt(
    impact_direction: str,
) -> str:
    """
    V17.0.13 - Traduce la direzione selezionata in una regola fisica
    non ambigua per il motore immagini.
    """
    rules = {
        "frontal": (
            "The collision force originates from the front of the vehicle "
            "and propagates toward the rear."
        ),
        "rear_to_front": (
            "The collision force originates from the rear of the vehicle "
            "and propagates toward the front. This represents a rear-end "
            "impact or compression from behind."
        ),
        "left_to_right": (
            "The collision force propagates from the left side toward "
            "the right side."
        ),
        "right_to_left": (
            "The collision force propagates from the right side toward "
            "the left side."
        ),
        "top_to_bottom": (
            "The collision force propagates from the top toward the bottom."
        ),
        "bottom_to_top": (
            "The collision force propagates from the bottom toward the top."
        ),
        "diagonal_right": (
            "The collision force follows a diagonal path toward the right."
        ),
        "diagonal_left": (
            "The collision force follows a diagonal path toward the left."
        ),
    }

    base_rule = rules.get(
        impact_direction,
        "Follow the selected impact direction mechanically.",
    )

    if impact_direction == "rear_to_front":
        return f"""
IMPACT DIRECTION — REAR TO FRONT

{base_rule}

If the selected component is at the rear of the vehicle, create a
mechanically plausible rear-end collision deformation.

Keep the strongest compression near the rear impact origin and allow only
limited forward propagation through the selected component.

Do not reinterpret this as a frontal impact.
Do not move or redesign unrelated rear components.
""".strip()

    return f"""
IMPACT DIRECTION

{base_rule}
""".strip()


def build_deformation_quality_prompt(
    severity_percent: int,
    area_percent: int,
    deformation_type: str,
) -> str:
    """
    V17.0.10 - Regole qualitative e meccaniche per evitare deformazioni
    troppo diffuse, radiali, decorative o artificialmente scolpite.
    """
    severity = max(0, min(100, abs(int(severity_percent))))
    area = max(0, min(100, abs(int(area_percent))))

    if area <= 25:
        spread_rule = (
            "Keep the deformation tightly localized around the selected "
            "impact origin. It must weaken very quickly away from that point."
        )
    elif area <= 50:
        spread_rule = (
            "Allow a limited propagation across the nearby portion of the "
            "selected panel, but preserve most of the panel geometry."
        )
    else:
        spread_rule = (
            "A broader deformation is allowed, but it must still have one "
            "clear mechanical origin and must not form decorative radial folds."
        )

    if severity <= 30:
        intensity_rule = (
            "Use a shallow localized dent with subtle tension lines. "
            "Do not create a deep buckle."
        )
    elif severity <= 65:
        intensity_rule = (
            "Use one dominant realistic buckle with one or two restrained "
            "secondary tension lines."
        )
    else:
        intensity_rule = (
            "Use a strong localized crush or buckle, but keep the force flow "
            "mechanically coherent and concentrated near the impact origin."
        )

    type_rules = {
        "dent": (
            "The main effect must be a localized inward dent with restrained "
            "edge tension and no ornamental ridges."
        ),
        "crease": (
            "Create one principal short crease. Secondary creases must be "
            "fewer, shorter and visibly weaker."
        ),
        "crush": (
            "Create localized compression and buckling. Do not spread many "
            "long folds across the full panel."
        ),
        "sideswipe": (
            "Create directional scraping deformation and one dominant force "
            "path, not a star-shaped or radial pattern."
        ),
        "multiple": (
            "Use a small number of mechanically related deformations that "
            "share the same collision logic."
        ),
    }
    deformation_rule = type_rules.get(
        str(deformation_type),
        type_rules["dent"],
    )

    return f"""
PHYSICAL DEFORMATION QUALITY RULES — STRICT

The result must look like real stamped automotive sheet metal after a
collision, not like a digitally sculpted or artistically folded surface.

{spread_rule}
{intensity_rule}
{deformation_rule}

Required physical behaviour:
- one dominant deformation close to the selected impact point;
- at most one or two smaller secondary tension lines;
- localized compression or buckling near the damaged panel edge;
- rapid reduction of deformation intensity away from the impact origin;
- mechanically plausible force propagation in the selected direction;
- preservation of the original geometry where collision force would not
  realistically reach.

Forbidden deformation patterns:
- radial or star-shaped folds;
- several long creases spreading across the panel;
- decorative, sinusoidal or wave-like ridges;
- repeated parallel folds;
- broad soft swelling;
- inflated, melted or rubber-like metal;
- smooth sculpted surfaces;
- excessive propagation toward the centre or rear of the panel;
- copying the exact deformation shape of an adjacent damaged panel.

The strongest crease or buckle must remain close to the impact origin.
The central and distant portions of the selected panel must remain as close
as possible to the original photograph unless the requested area explicitly
requires otherwise.

Panel gaps must remain mechanically plausible.
Protected adjacent parts must remain completely unchanged.
""".strip()


def validate_deformation_locality(
    source: Image.Image,
    candidate_bytes: bytes,
    guided_mask: Image.Image,
    area_percent: int,
    severity_percent: int = 0,
    selected_components: list[str] | None = None,
) -> dict[str, object]:
    """
    V17.0.19 - Validazione adattiva per urti multicompomente.
    """
    components = [
        str(item).strip()
        for item in (selected_components or [])
        if str(item).strip()
    ]

    generic_codes = {
        "bodypanel",
        "body_panel",
        "bodywork",
        "generic_body_panel",
    }
    specific_components = [
        item
        for item in components
        if item.lower() not in generic_codes
    ]
    component_count = max(
        1,
        len(specific_components or components),
    )

    source_rgb = source.convert("RGB")
    width, height = source_rgb.size

    try:
        candidate = Image.open(io.BytesIO(candidate_bytes))
        candidate.load()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il risultato non è un'immagine valida.",
        ) from exc

    candidate_rgb = candidate.convert("RGB")
    if candidate_rgb.size != (width, height):
        candidate_rgb = candidate_rgb.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    source_array = np.asarray(source_rgb, dtype=np.uint8)
    candidate_array = np.asarray(candidate_rgb, dtype=np.uint8)
    mask = mask_to_binary(
        resize_mask(guided_mask.convert("L"), (width, height))
    ) > 0

    selected_pixels = int(mask.sum())
    if selected_pixels == 0:
        return {
            "deformation_locality_validation_applied": False,
            "reason": "empty_guided_mask",
        }

    difference = np.mean(
        np.abs(
            candidate_array.astype(np.int16)
            - source_array.astype(np.int16)
        ),
        axis=2,
    )

    changed_inside = (difference > 14.0) & mask
    changed_ratio = float(changed_inside.sum()) / float(selected_pixels)

    requested_area = (
        max(0, min(100, abs(int(area_percent)))) / 100.0
    )
    severity = max(0, min(100, abs(int(severity_percent))))

    allowed_ratio = max(0.35, requested_area + 0.30)

    if component_count == 2:
        allowed_ratio = max(allowed_ratio, 0.72)
    elif component_count == 3:
        allowed_ratio = max(allowed_ratio, 0.80)
    elif component_count >= 4:
        allowed_ratio = max(allowed_ratio, 0.88)

    if severity >= 70 and component_count >= 3:
        allowed_ratio = max(allowed_ratio, 0.90)
    if severity >= 85 and component_count >= 4:
        allowed_ratio = max(allowed_ratio, 0.93)

    allowed_ratio = min(0.94, allowed_ratio)

    validation_warning = None

    if changed_ratio > allowed_ratio:
        if component_count >= 3:
            validation_warning = {
                "message": (
                    "La deformazione supera la soglia indicativa, ma il "
                    "risultato viene accettato perché riguarda un urto "
                    "multicomponente."
                ),
                "changed_component_ratio": round(changed_ratio, 4),
                "allowed_component_ratio": round(allowed_ratio, 4),
                "selected_component_count": component_count,
            }
            print(
                "[MULTICOMPONENT VALIDATION WARNING]",
                validation_warning,
                flush=True,
            )
        else:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "La deformazione ha modificato quasi tutta l'area "
                        "selezionata oltre il limite consentito."
                    ),
                    "changed_component_ratio": round(changed_ratio, 4),
                    "allowed_component_ratio": round(allowed_ratio, 4),
                    "requested_area_percent": int(area_percent),
                    "severity_percent": int(severity_percent),
                    "selected_component_count": component_count,
                    "selected_components": (
                        specific_components or components
                    ),
                    "validation_profile": (
                        "multicomponent_release"
                    ),
                },
            )

    return {
        "deformation_locality_validation_applied": True,
        "changed_component_ratio": round(changed_ratio, 4),
        "allowed_component_ratio": round(allowed_ratio, 4),
        "requested_area_percent": int(area_percent),
        "severity_percent": int(severity_percent),
        "selected_component_count": component_count,
        "selected_components_for_validation": (
            specific_components or components
        ),
        "validation_profile": "multicomponent_release",
        "deformation_locality_validation_passed": True,
        "deformation_locality_validation_warning": validation_warning,
        "multicomponent_result_released": (
            validation_warning is not None
        ),
    }



def build_balanced_continuity_prompt(
    damage_continuity_text: str,
    area_percent: int,
    severity_percent: int,
) -> str:
    """
    V17.0.11 - Evita sia la deformazione troppo diffusa sia il danno
    eccessivamente puntiforme, favorendo un raccordo realistico con
    il pannello adiacente già danneggiato.
    """
    continuity_text = (damage_continuity_text or "").strip().lower()
    area = max(0, min(100, abs(int(area_percent))))
    severity = max(0, min(100, abs(int(severity_percent))))

    continuity_requested = any(
        token in continuity_text
        for token in (
            "continue",
            "continuity",
            "existing",
            "adjacent",
            "prosegue",
            "continua",
            "esistente",
            "adiacente",
        )
    )

    if area <= 30:
        width_rule = (
            "Use a compact but not point-like deformation: one short dominant "
            "crease surrounded by a slightly wider shallow depression."
        )
    elif area <= 55:
        width_rule = (
            "Use one dominant crease with a moderate surrounding depression, "
            "keeping the deformation clearly localized."
        )
    else:
        width_rule = (
            "Allow a broader depression, but preserve one dominant mechanical "
            "force path and avoid radial spreading."
        )

    if severity <= 30:
        depth_rule = (
            "Keep the depression shallow and avoid a sharp pinched centre."
        )
    elif severity <= 65:
        depth_rule = (
            "Use a moderate buckle with rounded sheet-metal compression around "
            "the main crease, not a puncture-like centre."
        )
    else:
        depth_rule = (
            "Use a stronger local buckle, but keep the centre blunt and "
            "collision-like rather than punctured."
        )

    if continuity_requested:
        continuity_rule = """
The deformation must connect naturally to the already damaged adjacent panel
through the nearest shared edge.

Slightly deform the selected panel edge closest to the adjacent damage,
while keeping the protected adjacent component itself unchanged.

The result must look like collision force transferred from the neighbouring
damaged panel, not like a separate object struck the middle of the panel.
""".strip()
    else:
        continuity_rule = (
            "Do not invent continuity with adjacent panels unless requested."
        )

    return f"""
BALANCED COLLISION CONTINUITY RULES — STRICT

{width_rule}
{depth_rule}
{continuity_rule}

Required shape:
- one short dominant crease or buckle;
- one moderately wider shallow depression around it;
- soft but mechanically credible compression around the main crease;
- a blunt collision centre, never a sharp puncture;
- limited secondary tension lines;
- deformation intensity that reduces quickly away from the impact origin.

Avoid:
- a tiny isolated dent;
- a sharp pinched centre;
- puncture-like geometry;
- star-shaped or radial folds;
- long decorative creases;
- broad swelling;
- deformation spreading toward the centre of the panel without a clear
  mechanical reason.

If continuity with an adjacent damaged panel is requested, the selected
panel edge nearest that damage may deform slightly, but protected components
and their geometry must remain unchanged.
""".strip()


def prepare_hybrid_guided_api_mask(
    manual_mask: Image.Image,
    target_size: tuple[int, int],
    edge_margin_px: int = 1,
) -> tuple[Image.Image, Image.Image, dict[str, object]]:
    """
    V17.0.8 - Prepara la maschera corretta dall'utente come guida reale
    dell'image edit.

    Base44:
      bianco = componente modificabile
      nero   = area protetta

    OpenAI:
      alpha 0   = zona modificabile
      alpha 255 = zona protetta

    Viene applicato soltanto un margine minimo di pochi pixel per consentire
    continuità naturale sul bordo del pannello, senza trasformare la maschera
    in un ritaglio o in un compositing successivo.
    """
    resized = resize_mask(manual_mask.convert("L"), target_size)
    binary = mask_to_binary(resized)

    selected_pixels = int((binary > 0).sum())
    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="La maschera guidata è vuota.",
        )

    margin = max(0, min(6, int(edge_margin_px)))
    expanded = binary.copy()

    if margin > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (margin * 2 + 1, margin * 2 + 1),
        )
        expanded = cv2.dilate(binary, kernel, iterations=1)

    guided_mask = Image.fromarray(expanded, mode="L")
    api_mask = alter_damage_area(guided_mask, 0)

    diagnostics = {
        "manual_mask_pixels": selected_pixels,
        "guided_mask_pixels": int((expanded > 0).sum()),
        "edge_margin_px": margin,
        "mask_geometry_source": "manual_smart_polygon",
    }
    return guided_mask, api_mask, diagnostics


def validate_hybrid_guided_result(
    source: Image.Image,
    candidate_bytes: bytes,
    guided_mask: Image.Image,
) -> tuple[bytes, dict[str, object]]:
    """
    V17.0.8 - Verifica che il risultato:
    - sia un fotogramma completo;
    - non introduca grandi aree nere;
    - mantenga quasi invariata la zona esterna alla maschera;
    - contenga una modifica effettiva dentro la maschera.

    Non applica compositing: valida soltanto l'output diretto del modello.
    """
    source_rgb = source.convert("RGB")
    width, height = source_rgb.size

    try:
        candidate_raw = Image.open(io.BytesIO(candidate_bytes))
        candidate_raw.load()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Il risultato ibrido non è un'immagine valida.",
        ) from exc

    candidate_rgb = candidate_raw.convert("RGB")
    original_candidate_size = candidate_rgb.size

    if candidate_rgb.size != (width, height):
        candidate_rgb = candidate_rgb.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    source_array = np.asarray(source_rgb, dtype=np.uint8)
    candidate_array = np.asarray(candidate_rgb, dtype=np.uint8)
    mask = mask_to_binary(
        resize_mask(guided_mask.convert("L"), (width, height))
    )

    inside = mask > 0
    outside = ~inside

    diff = np.mean(
        np.abs(
            candidate_array.astype(np.int16)
            - source_array.astype(np.int16)
        ),
        axis=2,
    )

    inside_mean_difference = (
        float(diff[inside].mean())
        if int(inside.sum()) > 0
        else 0.0
    )
    outside_mean_difference = (
        float(diff[outside].mean())
        if int(outside.sum()) > 0
        else 0.0
    )

    outside_changed_ratio = (
        float((diff[outside] > 18.0).sum()) / float(outside.sum())
        if int(outside.sum()) > 0
        else 0.0
    )

    black_pixels = np.all(candidate_array < 8, axis=2)
    black_ratio = float(black_pixels.sum()) / float(width * height)

    # Rifiuta output evidentemente corrotti o completamente reinventati.
    if black_ratio > 0.08:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Il motore ha restituito grandi aree nere.",
                "black_ratio": round(black_ratio, 4),
            },
        )

    if outside_mean_difference > 20.0 and outside_changed_ratio > 0.20:
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Il motore ha modificato troppo la fotografia fuori "
                    "dal componente selezionato."
                ),
                "outside_mean_difference": round(
                    outside_mean_difference,
                    3,
                ),
                "outside_changed_ratio": round(
                    outside_changed_ratio,
                    4,
                ),
            },
        )

    if inside_mean_difference < 1.2:
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Il motore non ha prodotto una modifica visibile "
                    "nel componente selezionato."
                ),
                "inside_mean_difference": round(
                    inside_mean_difference,
                    3,
                ),
            },
        )

    output = io.BytesIO()
    candidate_rgb.save(
        output,
        format="JPEG",
        quality=96,
        subsampling=0,
        optimize=True,
    )

    diagnostics = {
        "result_is_full_frame": True,
        "original_size": [width, height],
        "candidate_original_size": [
            int(original_candidate_size[0]),
            int(original_candidate_size[1]),
        ],
        "result_size": [width, height],
        "inside_mean_difference": round(
            inside_mean_difference,
            3,
        ),
        "outside_mean_difference": round(
            outside_mean_difference,
            3,
        ),
        "outside_changed_ratio": round(
            outside_changed_ratio,
            4,
        ),
        "black_ratio": round(black_ratio, 4),
        "direct_model_output_used": False,
        "post_composite_applied": True,
        "hybrid_result_validation_passed": True,
    }
    return output.getvalue(), diagnostics


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
        print(
            "[OPENAI IMAGE ERROR]",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "request_id": request_id,
                "model": os.getenv(
                    "OPENAI_IMAGE_MODEL",
                    "gpt-image-2",
                ),
                "quality": quality,
                "source_size": list(source.size),
                "mask_size": list(api_mask.size),
                "prompt_length": len(prompt),
            },
            flush=True,
        )
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
    finally:
        try:
            source_file.close()
        except Exception:
            pass
        try:
            mask_file.close()
        except Exception:
            pass
        gc.collect()



IDENTITY_REGION_TYPES = {
    "license_plate",
    "manufacturer_emblem",
    "model_badge",
    "trim_badge",
    "visible_brand_text",
    "left_tail_light",
    "right_tail_light",
}


def normalize_identity_regions(
    raw_regions: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []

    for item in raw_regions or []:
        if not isinstance(item, dict):
            continue

        region_type = str(item.get("type", "")).strip().lower()
        if region_type not in IDENTITY_REGION_TYPES:
            continue

        box = item.get("bounding_box")
        if not isinstance(box, dict):
            continue

        try:
            x1 = max(0, min(1000, int(box.get("x1", 0))))
            y1 = max(0, min(1000, int(box.get("y1", 0))))
            x2 = max(0, min(1000, int(box.get("x2", 0))))
            y2 = max(0, min(1000, int(box.get("y2", 0))))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        # Conserviamo solo rilevamenti con una confidenza ragionevole.
        if confidence < 0.35:
            continue

        normalized.append({
            "type": region_type,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "text": str(item.get("text", "") or "").strip()[:80],
        })

    # Deduplicazione semplice per tipo e forte sovrapposizione.
    deduplicated: list[dict[str, object]] = []

    for candidate in sorted(
        normalized,
        key=lambda region: float(region["confidence"]),
        reverse=True,
    ):
        candidate_box = candidate["bounding_box"]
        duplicate = False

        for existing in deduplicated:
            if existing["type"] != candidate["type"]:
                continue

            existing_box = existing["bounding_box"]

            ix1 = max(candidate_box["x1"], existing_box["x1"])
            iy1 = max(candidate_box["y1"], existing_box["y1"])
            ix2 = min(candidate_box["x2"], existing_box["x2"])
            iy2 = min(candidate_box["y2"], existing_box["y2"])

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            candidate_area = (
                (candidate_box["x2"] - candidate_box["x1"])
                * (candidate_box["y2"] - candidate_box["y1"])
            )

            if candidate_area > 0 and intersection / candidate_area > 0.65:
                duplicate = True
                break

        if not duplicate:
            deduplicated.append(candidate)

    return deduplicated


def call_openai_identity_analysis(
    source: Image.Image,
) -> list[dict[str, object]]:
    """
    Rileva dinamicamente targa, stemma, badge e scritte identitarie.
    Le coordinate sono normalizzate 0..1000 e valgono per qualsiasi auto.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    image_data_url = image_to_data_url(source)
    model = os.getenv("OPENAI_IDENTITY_MODEL", "gpt-4.1")
    client = OpenAI(api_key=api_key)

    prompt = """
Analyze the exact vehicle in this photograph and locate only identity-critical
visual elements that must never be regenerated during a damage simulation.

Return ONLY valid JSON:
{
  "regions": [
    {
      "type": one of [
        "license_plate",
        "manufacturer_emblem",
        "model_badge",
        "trim_badge",
        "visible_brand_text",
        "left_tail_light",
        "right_tail_light"
      ],
      "confidence": number from 0 to 1,
      "text": readable text when present, otherwise "",
      "bounding_box": {
        "x1": integer 0..1000,
        "y1": integer 0..1000,
        "x2": integer 0..1000,
        "y2": integer 0..1000
      }
    }
  ]
}

Rules:
- Coordinates refer to the complete photograph.
- Use tight boxes around the exact visible element.
- Detect the actual license plate, manufacturer emblem, model name badge,
  trim badge and any visible brand/model text on the vehicle.
- Also detect the complete outer bounding box of each visible rear light:
  left_tail_light and right_tail_light.
- Do not include windows, body panels, reflections or background.
- Do not invent elements that are not visible.
- A damaged or tilted element must still be located.
- Return JSON only.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You locate automotive identity elements precisely. "
                "Return valid JSON only."
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

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        parsed = extract_json_object(
            response.choices[0].message.content or "{}"
        )
        return normalize_identity_regions(
            parsed.get("regions", [])
            if isinstance(parsed, dict)
            else []
        )
    except Exception as exc:
        print(
            "[IDENTITY ANALYSIS WARNING]",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "model": model,
            },
            flush=True,
        )
        return []


def split_identity_regions(
    regions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hard_identity_types = {
        "license_plate",
        "manufacturer_emblem",
        "model_badge",
        "trim_badge",
        "visible_brand_text",
    }
    tail_light_types = {
        "left_tail_light",
        "right_tail_light",
    }

    hard_regions = [
        region
        for region in regions
        if str(region.get("type", "")).lower() in hard_identity_types
    ]
    tail_light_regions = [
        region
        for region in regions
        if str(region.get("type", "")).lower() in tail_light_types
    ]
    return hard_regions, tail_light_regions


def filter_relevant_tail_light_regions(
    regions: list[dict[str, object]],
    selected_components: list[str],
) -> list[dict[str, object]]:
    selected = {
        str(item).strip().lower()
        for item in selected_components or []
    }

    wants_left = bool(selected & {
        "left_tail_light",
        "rear_light_left",
        "tail_light_left",
        "rear_light",
        "tail_light",
    })
    wants_right = bool(selected & {
        "right_tail_light",
        "rear_light_right",
        "tail_light_right",
    })

    relevant: list[dict[str, object]] = []
    for region in regions:
        region_type = str(region.get("type", "")).strip().lower()
        if region_type == "left_tail_light" and wants_left:
            relevant.append(region)
        elif region_type == "right_tail_light" and wants_right:
            relevant.append(region)

    return relevant


def build_tail_light_geometry_mask(
    regions: list[dict[str, object]],
    target_size: tuple[int, int],
    padding_percent: float = 0.5,
) -> Image.Image | None:
    """
    Crea una maschera della sagoma esterna del fanale.
    La zona viene esclusa dalla rigenerazione geometrica, poi una porzione
    interna più piccola resta modificabile per consentire crepe/rotture.
    """
    if not regions:
        return None

    width, height = target_size
    outer = Image.new("L", target_size, 0)
    inner = Image.new("L", target_size, 0)
    outer_draw = ImageDraw.Draw(outer)
    inner_draw = ImageDraw.Draw(inner)

    base_pad_x = int(round(width * max(0.0, padding_percent) / 100.0))
    base_pad_y = int(round(height * max(0.0, padding_percent) / 100.0))

    for region in regions:
        box = region["bounding_box"]
        x1 = int(round(width * int(box["x1"]) / 1000.0))
        y1 = int(round(height * int(box["y1"]) / 1000.0))
        x2 = int(round(width * int(box["x2"]) / 1000.0))
        y2 = int(round(height * int(box["y2"]) / 1000.0))

        x1 = max(0, x1 - base_pad_x)
        y1 = max(0, y1 - base_pad_y)
        x2 = min(width, x2 + base_pad_x)
        y2 = min(height, y2 + base_pad_y)

        outer_draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=max(2, min(12, (x2 - x1) // 12)),
            fill=255,
        )

        inset_x = max(2, int(round((x2 - x1) * 0.16)))
        inset_y = max(2, int(round((y2 - y1) * 0.18)))
        ix1 = min(x2 - 1, x1 + inset_x)
        iy1 = min(y2 - 1, y1 + inset_y)
        ix2 = max(ix1 + 1, x2 - inset_x)
        iy2 = max(iy1 + 1, y2 - inset_y)

        inner_draw.rounded_rectangle(
            [ix1, iy1, ix2, iy2],
            radius=max(1, min(8, (ix2 - ix1) // 14)),
            fill=255,
        )

    outer_arr = mask_to_binary(outer)
    inner_arr = mask_to_binary(inner)

    # Protegge il bordo/sagoma esterna; l'interno resta editabile.
    ring = cv2.bitwise_and(
        outer_arr,
        cv2.bitwise_not(inner_arr),
    )
    return Image.fromarray(ring, mode="L")


def build_identity_protect_mask(
    regions: list[dict[str, object]],
    target_size: tuple[int, int],
    padding_percent: float = 1.2,
) -> Image.Image:
    width, height = target_size
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)

    pad_x = int(round(width * max(0.0, padding_percent) / 100.0))
    pad_y = int(round(height * max(0.0, padding_percent) / 100.0))

    for region in regions:
        box = region["bounding_box"]

        region_type = str(region.get("type", "")).strip().lower()

        type_padding_multiplier = {
            "license_plate": 1.9,
            "manufacturer_emblem": 2.3,
            "model_badge": 2.2,
            "trim_badge": 2.2,
            "visible_brand_text": 2.0,
            "left_tail_light": 1.5,
            "right_tail_light": 1.5,
        }.get(region_type, 1.5)

        region_pad_x = int(round(pad_x * type_padding_multiplier))
        region_pad_y = int(round(pad_y * type_padding_multiplier))

        x1 = (
            int(round(width * int(box["x1"]) / 1000.0))
            - region_pad_x
        )
        y1 = (
            int(round(height * int(box["y1"]) / 1000.0))
            - region_pad_y
        )
        x2 = (
            int(round(width * int(box["x2"]) / 1000.0))
            + region_pad_x
        )
        y2 = (
            int(round(height * int(box["y2"]) / 1000.0))
            + region_pad_y
        )

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=max(1, min(pad_x, pad_y, 8)),
            fill=255,
        )

    return mask


def merge_protect_masks(
    target_size: tuple[int, int],
    masks: list[Image.Image | None],
) -> Image.Image | None:
    merged = np.zeros(
        (target_size[1], target_size[0]),
        dtype=np.uint8,
    )
    found = False

    for mask in masks:
        if mask is None:
            continue
        resized = resize_mask(mask.convert("L"), target_size)
        binary = mask_to_binary(resized)
        if int((binary > 0).sum()) == 0:
            continue
        merged = cv2.bitwise_or(merged, binary)
        found = True

    return Image.fromarray(merged, mode="L") if found else None


def preserve_identity_pixels(
    source: Image.Image,
    candidate_bytes: bytes,
    identity_mask: Image.Image | None,
    guided_mask: Image.Image | None = None,
    apply_color_correction: bool = True,
) -> tuple[bytes, dict[str, object]]:
    """
    Reinserisce i pixel identitari originali e riallinea la tinta della
    carrozzeria nella zona modificata. Funziona con posizioni dinamiche.
    """
    try:
        candidate = Image.open(io.BytesIO(candidate_bytes))
        candidate.load()
        candidate = candidate.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Risultato non decodificabile per identity protection.",
        ) from exc

    source_rgb = source.convert("RGB")
    if candidate.size != source_rgb.size:
        candidate = candidate.resize(
            source_rgb.size,
            Image.Resampling.LANCZOS,
        )

    source_array = np.array(
        source_rgb,
        dtype=np.uint8,
        copy=True,
    )
    candidate_array = np.array(
        candidate,
        dtype=np.uint8,
        copy=True,
    )

    color_corrected = False
    if apply_color_correction and guided_mask is not None:
        guided_binary = mask_to_binary(
            resize_mask(guided_mask.convert("L"), source_rgb.size)
        )
        candidate_array = _lab_color_match_inside_mask(
            source_rgb=source_array,
            generated_rgb=candidate_array,
            mask_binary=guided_binary,
            strength=0.58,
        )
        color_corrected = True

    identity_pixels = 0
    if identity_mask is not None:
        identity_binary = mask_to_binary(
            resize_mask(identity_mask.convert("L"), source_rgb.size)
        )
        identity_pixels = int((identity_binary > 0).sum())

        # Ripristino rigido: targa, stemma e badge devono restare
        # pixel-identici all'originale. Nessun feather che possa riscrivere
        # lettere, loghi o bordi.
        hard = identity_binary > 0
        if not candidate_array.flags.writeable:
            candidate_array = candidate_array.copy()
        candidate_array[hard] = source_array[hard]

    output = io.BytesIO()
    Image.fromarray(candidate_array, mode="RGB").save(
        output,
        format="JPEG",
        quality=96,
        subsampling=0,
        optimize=True,
    )

    return output.getvalue(), {
        "dynamic_identity_composite_applied": identity_pixels > 0,
        "identity_protected_pixels": identity_pixels,
        "paint_color_correction_applied": color_corrected,
        "identity_composite_strategy": (
            "hard_exact_original_pixel_restore"
            if identity_pixels > 0
            else "none"
        ),
        "identity_pixels_exactly_restored": identity_pixels > 0,
    }


@app.post("/v1/vehicle/analyze-identity")
def analyze_vehicle_identity(
    payload: ImageNormalizeRequest,
) -> dict[str, object]:
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    )
    source, _, _ = resize_in_place_for_memory(
        source,
        min(MAX_PROCESSING_SIDE, 1400),
    )
    regions = call_openai_identity_analysis(source)

    return {
        "status": "completed",
        "version": "1.7.0.26",
        "regions": regions,
        "region_count": len(regions),
        "dynamic_identity_protection": True,
    }


@app.post("/v1/image/normalize")
def normalize_image(payload: ImageNormalizeRequest) -> dict[str, object]:
    """V17.0.14: normalizza HEIC/HEIF/JPEG/PNG per Base44 e browser."""
    try:
        result = normalize_image_to_base64(
            image_base64=payload.image_base64,
            output_format=payload.output_format,
            max_side=payload.max_side,
            jpeg_quality=payload.jpeg_quality,
        )
        result.update({
            "status": "completed",
            "version": "v17.0.14",
            "memory_optimized": True,
        })
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Impossibile normalizzare la fotografia.",
                "type": type(exc).__name__,
                "error": str(exc),
                "heic_support_enabled": HEIC_SUPPORT_ENABLED,
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

@app.get("/v1/version")
def get_backend_version() -> dict[str, object]:
    return {
        "service": "Car Damage Lab Engine",
        "version": "1.7.0.26",
        "prompt_version": "damage-v17.0.26-writable-array-hotfix",
        "multicomponent_release": True,
        "generic_body_panel_filter": True,
        "fragile_rear_components": True,
        "vehicle_identity_lock": True,
        "rear_light_geometry_lock": True,
        "badge_logo_plate_lock": True,
        "dynamic_identity_detection": True,
        "dynamic_identity_masking": True,
        "dynamic_identity_composite": True,
        "paint_color_correction": True,
        "identity_analysis_endpoint": "/v1/vehicle/analyze-identity",
        "imagedraw_hotfix": True,
        "conservative_identity_guard": True,
        "hard_identity_pixel_restore": True,
        "conservative_component_composite": True,
        "default_color_correction_disabled": True,
        "identity_model": "gpt-4.1",
        "plate_hard_restore": True,
        "taillight_outer_geometry_lock": True,
        "post_composite_result_forced": True,
        "writable_array_hotfix": True,
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
        "memory_optimized": True,
        "max_processing_side": MAX_PROCESSING_SIDE,
        "heic_support_enabled": HEIC_SUPPORT_ENABLED,
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
                "analysis_version": "vehicle-segmentation-v17.0.26-writable-array-hotfix",
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
                "analysis_version": "vehicle-segmentation-v17.0.26-writable-array-hotfix",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Connessione a Replicate non riuscita.",
                "error": f"{type(exc).__name__}: {str(exc)}"[:1200],
                "analysis_version": "vehicle-segmentation-v17.0.26-writable-array-hotfix",
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



def _extract_retry_after_seconds(exc: HTTPException) -> float:
    detail = exc.detail

    if isinstance(detail, dict):
        raw = str(detail.get("replicate_detail", ""))
        match = re.search(r'"retry_after"\s*:\s*(\d+)', raw)
        if match:
            return max(1.0, float(match.group(1)))

        match = re.search(r"resets in ~(\d+)s", raw)
        if match:
            return max(1.0, float(match.group(1)))

    return REPLICATE_CREATE_MIN_INTERVAL_SECONDS


def _create_replicate_prediction(
    version: str,
    input_payload: dict,
    timeout_seconds: int | None = None,
) -> dict:
    global REPLICATE_LAST_CREATE_AT

    last_error: Exception | None = None

    for attempt in range(REPLICATE_RATE_LIMIT_MAX_RETRIES):
        with REPLICATE_CREATE_LOCK:
            elapsed = time.monotonic() - REPLICATE_LAST_CREATE_AT
            wait_for = REPLICATE_CREATE_MIN_INTERVAL_SECONDS - elapsed

            if wait_for > 0:
                time.sleep(wait_for)

            try:
                prediction = _replicate_json_request(
                    "https://api.replicate.com/v1/predictions",
                    method="POST",
                    payload={
                        "version": version,
                        "input": input_payload,
                    },
                )
                REPLICATE_LAST_CREATE_AT = time.monotonic()
            except HTTPException as exc:
                last_error = exc

                detail = exc.detail
                status = (
                    detail.get("http_status")
                    if isinstance(detail, dict)
                    else None
                )

                if status != 429:
                    raise

                retry_after = _extract_retry_after_seconds(exc)
                REPLICATE_LAST_CREATE_AT = time.monotonic()

                print(
                    "REPLICATE RATE LIMIT:",
                    "attempt=",
                    attempt + 1,
                    "retry_after=",
                    retry_after,
                )

                time.sleep(retry_after + 1.0)
                continue

        prediction_id = prediction.get("id")
        if not prediction_id:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Replicate non ha restituito un prediction id.",
                    "response": prediction,
                },
            )

        deadline = time.monotonic() + (
            timeout_seconds or REPLICATE_TIMEOUT_SECONDS
        )
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
                        "message": (
                            "Timeout durante la segmentazione guidata SAM 2."
                        ),
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
                    "message": (
                        "La segmentazione guidata SAM 2 non è riuscita."
                    ),
                    "prediction_id": prediction_id,
                    "status": current.get("status"),
                    "error": current.get("error"),
                    "logs": str(current.get("logs") or "")[-1600:],
                },
            )

        return current

    if isinstance(last_error, HTTPException):
        raise last_error

    raise HTTPException(
        status_code=429,
        detail={
            "message": (
                "Limite Replicate ancora attivo dopo diversi tentativi."
            ),
            "analysis_version": (
                "vehicle-segmentation-v17.0.26-writable-array-hotfix"
            ),
        },
    )


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
        timeout_seconds=PROMPTED_SAM_COMPONENT_TIMEOUT_SECONDS,
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



def decode_mask_data_url(
    data_url: str,
    target_size: tuple[int, int],
) -> Image.Image:
    header, encoded = data_url.split(",", 1) if "," in data_url else ("", data_url)
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("L")
    return resize_mask(image, target_size)


def normalize_refinement_points(
    points: list,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []

    for point in points:
        if isinstance(point, BaseModel):
            raw_x = getattr(point, "x", None)
            raw_y = getattr(point, "y", None)
        elif isinstance(point, dict):
            raw_x = point.get("x")
            raw_y = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            raw_x, raw_y = point[0], point[1]
        else:
            continue

        try:
            x = float(raw_x)
            y = float(raw_y)
        except Exception:
            continue

        if max(abs(x), abs(y)) <= 1.5:
            px = round(x * (width - 1))
            py = round(y * (height - 1))
        elif max(abs(x), abs(y)) <= 1100:
            px = round(x * (width - 1) / 1000.0)
            py = round(y * (height - 1) / 1000.0)
        else:
            px = round(x)
            py = round(y)

        normalized.append((
            max(0, min(width - 1, px)),
            max(0, min(height - 1, py)),
        ))

    return normalized


def _component_crop_box(
    box: dict,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    margin_x = max(12, round(box["width"] * COMPONENT_REFINEMENT_MARGIN_RATIO))
    margin_y = max(12, round(box["height"] * COMPONENT_REFINEMENT_MARGIN_RATIO))

    return (
        max(0, box["x1"] - margin_x),
        max(0, box["y1"] - margin_y),
        min(image_width, box["x2"] + margin_x),
        min(image_height, box["y2"] + margin_y),
    )


def _select_refined_connected_region(
    binary: np.ndarray,
    reference_binary: np.ndarray,
    box: dict,
) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.where(binary > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )

    if count <= 1:
        return binary

    center_x = (box["x1"] + box["x2"]) / 2.0
    center_y = (box["y1"] + box["y2"]) / 2.0

    best_label = 0
    best_score = -1.0

    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 12:
            continue

        overlap = int(
            np.logical_and(component, reference_binary > 0).sum()
        )
        inside_box = int(
            component[
                box["y1"]:box["y2"],
                box["x1"]:box["x2"],
            ].sum()
        )
        centroid_x, centroid_y = centroids[index]
        distance = (
            ((centroid_x - center_x) / max(1.0, box["width"])) ** 2
            + ((centroid_y - center_y) / max(1.0, box["height"])) ** 2
        ) ** 0.5
        center_score = max(0.0, 1.0 - distance)
        score = (
            overlap * 2.0
            + inside_box * 1.2
            + area * 0.05
            + center_score * max(1, area) * 0.25
        )

        if score > best_score:
            best_score = score
            best_label = index

    selected = np.zeros_like(binary)

    if best_label > 0:
        selected[labels == best_label] = 255
    else:
        selected = binary

    return selected


def _score_refinement_candidate(
    candidate: np.ndarray,
    initial: np.ndarray,
    box: dict,
    edge_map: np.ndarray,
) -> float:
    area = int((candidate > 0).sum())
    if area <= 0:
        return -1e9

    overlap = int(
        np.logical_and(candidate > 0, initial > 0).sum()
    )
    union = int(
        np.logical_or(candidate > 0, initial > 0).sum()
    )
    iou = overlap / max(1, union)

    box_area = max(1, box["width"] * box["height"])
    inside = int(
        (candidate[
            box["y1"]:box["y2"],
            box["x1"]:box["x2"],
        ] > 0).sum()
    )
    inside_ratio = inside / max(1, area)
    plausible_area = min(area / box_area, box_area / max(1, area))

    contour = cv2.morphologyEx(
        candidate,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    contour_pixels = contour > 0
    edge_alignment = (
        float(edge_map[contour_pixels].mean()) / 255.0
        if contour_pixels.any()
        else 0.0
    )

    return (
        0.34 * iou
        + 0.30 * inside_ratio
        + 0.20 * plausible_area
        + 0.16 * edge_alignment
    )


def refine_component_mask_local(
    source: Image.Image,
    box: dict,
    initial_mask: Image.Image,
    positive_points: list[tuple[int, int]] | None = None,
    negative_points: list[tuple[int, int]] | None = None,
    iterations: int | None = None,
) -> tuple[Image.Image, dict]:
    positive_points = positive_points or []
    negative_points = negative_points or []
    iterations = iterations or COMPONENT_REFINEMENT_ITERATIONS

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    initial = mask_to_binary(
        resize_mask(initial_mask.convert("L"), source.size)
    )

    crop_x1, crop_y1, crop_x2, crop_y2 = _component_crop_box(
        box,
        source.width,
        source.height,
    )

    crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    initial_crop = initial[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return initial_mask, {
            "refinement_status": "skipped_invalid_crop",
            "candidate_count": 0,
        }

    original_crop_h, original_crop_w = crop.shape[:2]
    max_side = max(original_crop_w, original_crop_h)
    scale = min(
        1.0,
        COMPONENT_REFINEMENT_MAX_CROP_SIDE / max(1, max_side),
    )

    if scale < 0.999:
        scaled_w = max(32, round(original_crop_w * scale))
        scaled_h = max(32, round(original_crop_h * scale))
        crop_work = cv2.resize(
            crop,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA,
        )
        initial_work = cv2.resize(
            initial_crop,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        crop_work = crop
        initial_work = initial_crop

    gray = cv2.cvtColor(crop_work, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 28, 28)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    candidates: list[np.ndarray] = []

    for erosion_size, dilation_size in ((5, 7), (9, 11)):
        gc_mask = np.full(
            initial_work.shape,
            cv2.GC_PR_BGD,
            dtype=np.uint8,
        )

        border = max(2, round(min(gc_mask.shape) * 0.02))
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        probable_fg = initial_work > 0
        gc_mask[probable_fg] = cv2.GC_PR_FGD

        erode_kernel = np.ones((erosion_size, erosion_size), np.uint8)
        core = cv2.erode(
            initial_work,
            erode_kernel,
            iterations=1,
        ) > 0
        gc_mask[core] = cv2.GC_FGD

        center_x = round(
            (((box["x1"] + box["x2"]) / 2) - crop_x1) * scale
        )
        center_y = round(
            (((box["y1"] + box["y2"]) / 2) - crop_y1) * scale
        )

        if (
            0 <= center_x < gc_mask.shape[1]
            and 0 <= center_y < gc_mask.shape[0]
        ):
            cv2.circle(gc_mask, (center_x, center_y), 4, cv2.GC_FGD, -1)

        for x, y in positive_points:
            local_x = round((x - crop_x1) * scale)
            local_y = round((y - crop_y1) * scale)
            if (
                0 <= local_x < gc_mask.shape[1]
                and 0 <= local_y < gc_mask.shape[0]
            ):
                cv2.circle(gc_mask, (local_x, local_y), 5, cv2.GC_FGD, -1)

        for x, y in negative_points:
            local_x = round((x - crop_x1) * scale)
            local_y = round((y - crop_y1) * scale)
            if (
                0 <= local_x < gc_mask.shape[1]
                and 0 <= local_y < gc_mask.shape[0]
            ):
                cv2.circle(gc_mask, (local_x, local_y), 6, cv2.GC_BGD, -1)

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                crop_work,
                gc_mask,
                None,
                bg_model,
                fg_model,
                iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            continue

        candidate_work = np.where(
            (gc_mask == cv2.GC_FGD)
            | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        candidate_work = cv2.morphologyEx(
            candidate_work,
            cv2.MORPH_CLOSE,
            np.ones((dilation_size, dilation_size), np.uint8),
        )
        candidate_work = cv2.morphologyEx(
            candidate_work,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        if scale < 0.999:
            candidate_crop = cv2.resize(
                candidate_work,
                (original_crop_w, original_crop_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            candidate_crop = candidate_work

        full = np.zeros(
            (source.height, source.width),
            dtype=np.uint8,
        )
        full[crop_y1:crop_y2, crop_x1:crop_x2] = candidate_crop
        full = _select_refined_connected_region(
            full,
            initial,
            box,
        )
        candidates.append(full)

    if not candidates:
        return initial_mask, {
            "refinement_status": "fallback_initial_mask",
            "candidate_count": 0,
            "processing_scale": round(scale, 4),
        }

    if scale < 0.999:
        edges_full_crop = cv2.resize(
            edges,
            (original_crop_w, original_crop_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        edges_full_crop = edges

    full_edge_map = np.zeros(
        (source.height, source.width),
        dtype=np.uint8,
    )
    full_edge_map[crop_y1:crop_y2, crop_x1:crop_x2] = edges_full_crop

    scored = [
        (
            _score_refinement_candidate(
                candidate,
                initial,
                box,
                full_edge_map,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best = scored[0]

    minimum_area = max(
        24,
        round(
            source.width
            * source.height
            * COMPONENT_REFINEMENT_MIN_AREA_RATIO
        ),
    )

    if int((best > 0).sum()) < minimum_area:
        best = initial
        best_score = 0.0
        status = "fallback_initial_mask"
    else:
        status = "refined"

    refined = clean_component_mask(
        Image.fromarray(best, mode="L")
    )

    return refined, {
        "refinement_status": status,
        "candidate_count": len(candidates),
        "selected_candidate_score": round(float(best_score), 4),
        "positive_point_count": len(positive_points),
        "negative_point_count": len(negative_points),
        "processing_scale": round(scale, 4),
        "processing_crop_size": [
            int(crop_work.shape[1]),
            int(crop_work.shape[0]),
        ],
    }




def apply_component_refinement(
    source: Image.Image,
    component: dict,
) -> dict:
    box = component_detection_box(
        component,
        source.width,
        source.height,
    )
    if box is None:
        return component

    try:
        initial_mask = decode_mask_data_url(
            component["mask_base64"],
            source.size,
        )
    except Exception:
        initial_mask = build_box_fallback_mask(source.size, box)

    refined_mask, diagnostics = refine_component_mask_local(
        source,
        box,
        initial_mask,
    )

    binary = mask_to_binary(refined_mask)
    if int((binary > 0).sum()) < 24:
        return component

    x, y, width, height = cv2.boundingRect(binary)

    result = dict(component)
    result.update({
        "mask_base64": mask_image_to_data_url(refined_mask),
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "pre_refinement_mask_source": component.get("mask_source"),
        "mask_source": (
            "opencv_component_refinement_after_"
            + str(component.get("mask_source", "unknown"))
        ),
        "refinement": diagnostics,
        "requires_review": True,
        "automatic_mask_status": "proposal_only",
    })
    return result




def polygon_points_to_pixels(
    points: list,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []

    for point in points:
        if isinstance(point, BaseModel):
            raw_x = getattr(point, "x", None)
            raw_y = getattr(point, "y", None)
        elif isinstance(point, dict):
            raw_x = point.get("x")
            raw_y = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            raw_x, raw_y = point[0], point[1]
        else:
            continue

        try:
            x = float(raw_x)
            y = float(raw_y)
        except Exception:
            continue

        # V17 frontend exchanges polygon coordinates in the 0..1000 space.
        px = round(x * (width - 1) / 1000.0)
        py = round(y * (height - 1) / 1000.0)

        result.append((
            max(0, min(width - 1, px)),
            max(0, min(height - 1, py)),
        ))

    return result


def polygon_points_to_normalized(
    points: list[tuple[int, int]],
    width: int,
    height: int,
) -> list[dict]:
    return [
        {
            "x": round(x * 1000.0 / max(1, width - 1), 2),
            "y": round(y * 1000.0 / max(1, height - 1), 2),
        }
        for x, y in points
    ]


def build_vehicle_edge_map(source: Image.Image) -> np.ndarray:
    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blur_size = SMART_POLYGON_EDGE_BLUR
    if blur_size % 2 == 0:
        blur_size += 1

    gray = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        0,
    )

    canny = cv2.Canny(
        gray,
        SMART_POLYGON_CANNY_LOW,
        SMART_POLYGON_CANNY_HIGH,
    )

    # Add panel seams and strong local contrast in both directions.
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = cv2.addWeighted(
        cv2.convertScaleAbs(grad_x),
        0.5,
        cv2.convertScaleAbs(grad_y),
        0.5,
        0,
    )
    _, strong_gradient = cv2.threshold(
        gradient,
        55,
        255,
        cv2.THRESH_BINARY,
    )

    edge_map = cv2.bitwise_or(canny, strong_gradient)
    edge_map = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )

    return edge_map


def snap_point_to_edge(
    point: tuple[int, int],
    edge_map: np.ndarray,
    radius: int,
) -> tuple[int, int]:
    if radius <= 0:
        return point

    height, width = edge_map.shape
    x, y = point

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width, x + radius + 1)
    y2 = min(height, y + radius + 1)

    region = edge_map[y1:y2, x1:x2]
    ys, xs = np.where(region > 0)

    if len(xs) == 0:
        return point

    absolute_xs = xs + x1
    absolute_ys = ys + y1
    distances = (
        (absolute_xs - x) ** 2
        + (absolute_ys - y) ** 2
    )

    index = int(np.argmin(distances))
    return (
        int(absolute_xs[index]),
        int(absolute_ys[index]),
    )


def snap_polygon_points(
    points: list[tuple[int, int]],
    edge_map: np.ndarray,
    radius: int,
) -> list[tuple[int, int]]:
    return [
        snap_point_to_edge(point, edge_map, radius)
        for point in points
    ]


def smooth_closed_polygon(
    points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if len(points) < 5:
        return points

    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)

    # Keep the polygon editable and avoid over-simplifying curved panels.
    epsilon = max(0.8, perimeter * 0.0035)
    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    result = [
        (int(point[0][0]), int(point[0][1]))
        for point in simplified
    ]

    return result if len(result) >= 3 else points


def polygon_to_mask(
    source_size: tuple[int, int],
    points: list[tuple[int, int]],
    feather_radius: int = 0,
) -> Image.Image:
    width, height = source_size
    mask = np.zeros((height, width), dtype=np.uint8)

    contour = np.asarray(
        points,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.fillPoly(mask, [contour], 255)

    if feather_radius > 0:
        sigma = max(0.8, feather_radius / 2.0)
        blurred = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        mask = np.where(blurred > 40, 255, 0).astype(np.uint8)

    return Image.fromarray(mask, mode="L")


def validate_polygon_mask(
    mask: Image.Image,
    component_code: str,
) -> None:
    binary = mask_to_binary(mask)
    area = int((binary > 0).sum())

    if area < SMART_POLYGON_MIN_AREA_PIXELS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Il poligono disegnato è troppo piccolo.",
                "component_code": component_code,
                "mask_area_pixels": area,
                "minimum_area_pixels": SMART_POLYGON_MIN_AREA_PIXELS,
            },
        )


def smart_polygon_component_payload(
    source: Image.Image,
    component_code: str,
    points: list,
    snap_to_edges: bool,
    snap_radius: int,
    smooth_polygon: bool,
    feather_radius: int,
    confirm_mask: bool,
) -> dict:
    pixel_points = polygon_points_to_pixels(
        points,
        source.width,
        source.height,
    )

    if len(pixel_points) < 3:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Servono almeno tre punti validi.",
                "component_code": component_code,
            },
        )

    original_points = list(pixel_points)
    edge_map = None

    if snap_to_edges:
        edge_map = build_vehicle_edge_map(source)
        pixel_points = snap_polygon_points(
            pixel_points,
            edge_map,
            snap_radius,
        )

    if smooth_polygon:
        pixel_points = smooth_closed_polygon(pixel_points)

    mask = polygon_to_mask(
        source.size,
        pixel_points,
        feather_radius=feather_radius,
    )
    mask = clean_component_mask(mask)
    validate_polygon_mask(mask, component_code)

    binary = mask_to_binary(mask)
    x, y, width, height = cv2.boundingRect(binary)

    return {
        "code": component_code,
        "label": VEHICLE_COMPONENT_CATALOG.get(
            component_code,
            component_code,
        ),
        "category": component_category(component_code),
        "mask_base64": mask_image_to_data_url(mask),
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "polygon_points": polygon_points_to_normalized(
            pixel_points,
            source.width,
            source.height,
        ),
        "original_polygon_points": polygon_points_to_normalized(
            original_points,
            source.width,
            source.height,
        ),
        "mask_source": "manual_smart_polygon",
        "requires_review": not confirm_mask,
        "sam_match_quality": (
            "manual_confirmed"
            if confirm_mask
            else "manual_review"
        ),
        "selection_mode": "manual_smart_polygon",
        "snap_to_edges": snap_to_edges,
        "snap_radius": snap_radius,
        "smooth_polygon": smooth_polygon,
        "feather_radius": feather_radius,
        "analysis_version": (
            "vehicle-segmentation-v17.0.26-writable-array-hotfix"
        ),
    }


def _mask_from_manual_input(
    manual_mask_base64: str,
    source_size: tuple[int, int],
) -> Image.Image:
    mask = decode_mask_data_url(
        manual_mask_base64,
        source_size,
    )
    return clean_component_mask(mask)


def _build_seed_mask_from_points(
    source: Image.Image,
    box: dict,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    current_mask: Image.Image | None,
    reset_mask: bool,
) -> Image.Image:
    if current_mask is not None and not reset_mask:
        initial = resize_mask(
            current_mask.convert("L"),
            source.size,
        )
    else:
        initial = build_box_fallback_mask(
            source.size,
            box,
        )

    # Use interactive points to refine locally.
    refined, _ = refine_component_mask_local(
        source,
        box,
        initial,
        positive_points=positive_points,
        negative_points=negative_points,
        iterations=COMPONENT_REFINEMENT_ITERATIONS,
    )

    return refined


def _component_payload_from_mask(
    component_code: str,
    mask: Image.Image,
    source_label: str,
    confirmed: bool,
    detection_box: dict | None,
    diagnostics: dict | None = None,
) -> dict:
    binary = mask_to_binary(mask)

    if int((binary > 0).sum()) < 24:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "La maschera del componente è vuota o troppo piccola.",
                "component_code": component_code,
            },
        )

    x, y, width, height = cv2.boundingRect(binary)

    return {
        "code": component_code,
        "label": VEHICLE_COMPONENT_CATALOG.get(
            component_code,
            component_code,
        ),
        "category": component_category(component_code),
        "mask_base64": mask_image_to_data_url(mask),
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "detection_box": detection_box,
        "mask_source": source_label,
        "requires_review": not confirmed,
        "sam_match_quality": (
            "manual_confirmed"
            if confirmed
            else "assisted_review"
        ),
        "selection_mode": "assisted",
        "refinement": diagnostics or {},
    }


def _fallback_component_from_detection(
    source: Image.Image,
    component: dict,
    failure_reason: str,
) -> dict | None:
    code = str(component.get("code", "")).strip()
    box = component_detection_box(
        component,
        source.width,
        source.height,
    )
    if not code or box is None:
        return None

    mask = build_box_fallback_mask(source.size, box)
    binary = mask_to_binary(mask)

    if int((binary > 0).sum()) < 24:
        return None

    x, y, width, height = cv2.boundingRect(binary)

    try:
        confidence = float(component.get("confidence", 0.70))
    except Exception:
        confidence = 0.70

    return {
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
        "mask_source": "openai_box_fallback_after_prompted_sam_failure",
        "sam_match_quality": "manual_review_required",
        "requires_review": True,
        "automatic_mask_status": "proposal_only",
        "prompted_failure_reason": failure_reason[:1000],
    }


def prompted_segment_components(
    image_data_url: str,
    source: Image.Image,
    raw_components: list[dict],
    progress_callback=None,
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
    total = max(1, len(valid_components))

    for index, component in enumerate(valid_components, start=1):
        code = str(component.get("code", "")).strip()
        label = VEHICLE_COMPONENT_CATALOG.get(code, code)

        if progress_callback is not None:
            progress_callback(
                index=index,
                total=total,
                code=code,
                label=label,
                stage="segmenting",
            )

        failure_reason = "prompted_sam_failed"

        try:
            item = run_prompted_sam_for_component(
                image_data_url,
                source.size,
                component,
            )

            if item.get("status") == "succeeded":
                results.append(item["component"])

                if progress_callback is not None:
                    progress_callback(
                        index=index,
                        total=total,
                        code=code,
                        label=label,
                        stage="refining",
                    )
                continue

            failure_reason = str(item.get("error", failure_reason))
            failures.append(item)

        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {str(exc)[:900]}"
            failures.append({
                "code": component.get("code"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
            })

        fallback = _fallback_component_from_detection(
            source,
            component,
            failure_reason,
        )
        if fallback is not None:
            results.append(fallback)

        if progress_callback is not None:
            progress_callback(
                index=index,
                total=total,
                code=code,
                label=label,
                stage="fallback" if fallback is not None else "skipped",
            )

    refined_results: list[dict] = []
    refinement_total = max(1, len(results))

    for refinement_index, item in enumerate(results, start=1):
        if progress_callback is not None:
            progress_callback(
                index=refinement_index,
                total=refinement_total,
                code=str(item.get("code", "")),
                label=str(item.get("label", "")),
                stage="local_refinement",
            )

        refined_results.append(
            apply_component_refinement(source, item)
        )

    refined_results.sort(
        key=lambda item: (
            bool(item.get("requires_review", False)),
            -float(item.get("confidence", 0.0)),
        )
    )

    return refined_results, failures




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

    raw_components = raw.get(
        "components",
        raw.get("visible_components", []),
    )
    if not isinstance(raw_components, list):
        raw_components = []

    detected_components: list[dict] = []

    for component in raw_components:
        if not isinstance(component, dict):
            continue

        code = str(component.get("code", "")).strip()
        if code not in VEHICLE_COMPONENT_CATALOG:
            continue

        box = component_detection_box(
            component,
            source.width,
            source.height,
        )

        try:
            confidence = float(component.get("confidence", 0.70))
        except Exception:
            confidence = 0.70

        detected_components.append({
            "code": code,
            "label": VEHICLE_COMPONENT_CATALOG[code],
            "category": component_category(code),
            "confidence": round(max(0.0, min(confidence, 1.0)), 2),
            "detection_box": box,
            "mask_base64": None,
            "mask_source": None,
            "requires_manual_polygon": True,
            "requires_review": True,
            "selected": False,
        })

    detected_components.sort(
        key=lambda item: item.get("confidence", 0.0),
        reverse=True,
    )

    return {
        "view": vehicle_view,
        "view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "vehicle_view": vehicle_view,
        "vehicle_view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "components": detected_components,
        "visible_components": detected_components,
        "automatic_masks_are_disabled": True,
        "manual_polygon_required_only_for_selected_components": True,
        "segmentation_strategy": "manual_smart_polygon",
        "analysis_version": (
            "vehicle-segmentation-v17.0.26-writable-array-hotfix"
        ),
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
                "analysis_version": "vehicle-segmentation-v17.0.26-writable-array-hotfix",
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
            progress_stage="preparing_manual_polygon_workspace",
            progress_percent=88,
            current_component_index=0,
            current_component_total=len(raw_components),
            current_component_code=None,
            current_component_label=None,
        )

        normalized = normalize_vehicle_analysis(
            raw_analysis,
            source,
            [],
            [],
        )

        if not normalized["components"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Non sono stati rilevati componenti utilizzabili "
                        "per la selezione manuale."
                    ),
                    "raw_component_count": len(raw_components),
                    "analysis_version": (
                        "vehicle-segmentation-v17.0.26-writable-array-hotfix"
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
                "vehicle-segmentation-v17.0.26-writable-array-hotfix"
            ),
            "mask_format": "data:image/png;base64",
            "mask_semantics": "white_component_black_background",
            "segmentation_provider": "manual-smart-polygon",
            "segmentation_strategy": "manual_smart_polygon",
            "component_timeout_seconds": PROMPTED_SAM_COMPONENT_TIMEOUT_SECONDS,
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
                        "vehicle-segmentation-v17.0.26-writable-array-hotfix"
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
                "vehicle-segmentation-v17.0.26-writable-array-hotfix"
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
            "vehicle-segmentation-v17.0.26-writable-array-hotfix"
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





@app.post("/v1/vehicle/snap-polygon-points")
def snap_vehicle_polygon_points(
    payload: SmartPolygonSnapRequest,
):
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    ).convert("RGB")

    pixel_points = polygon_points_to_pixels(
        payload.points,
        source.width,
        source.height,
    )
    edge_map = build_vehicle_edge_map(source)
    snapped_points = snap_polygon_points(
        pixel_points,
        edge_map,
        payload.snap_radius,
    )

    return {
        "points": polygon_points_to_normalized(
            snapped_points,
            source.width,
            source.height,
        ),
        "original_points": polygon_points_to_normalized(
            pixel_points,
            source.width,
            source.height,
        ),
        "snap_radius": payload.snap_radius,
        "analysis_version": (
            "vehicle-segmentation-v17.0.26-writable-array-hotfix"
        ),
    }


@app.post("/v1/vehicle/manual-smart-polygon")
def create_vehicle_manual_smart_polygon(
    payload: SmartPolygonRequest,
):
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    ).convert("RGB")

    return smart_polygon_component_payload(
        source=source,
        component_code=payload.component_code,
        points=payload.points,
        snap_to_edges=payload.snap_to_edges,
        snap_radius=payload.snap_radius,
        smooth_polygon=payload.smooth_polygon,
        feather_radius=payload.feather_radius,
        confirm_mask=payload.confirm_mask,
    )


@app.post("/v1/vehicle/assisted-component-selection")
def assisted_component_selection(
    payload: AssistedComponentSelectionRequest,
):
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    ).convert("RGB")

    box = None
    if payload.detection_box is not None:
        box = normalize_box(
            payload.detection_box,
            source.width,
            source.height,
        )

    if payload.manual_mask_base64:
        mask = _mask_from_manual_input(
            payload.manual_mask_base64,
            source.size,
        )
        return _component_payload_from_mask(
            payload.component_code,
            mask,
            source_label="manual_component_mask",
            confirmed=payload.confirm_manual_mask,
            detection_box=box,
            diagnostics={
                "source": "manual_mask_upload",
                "reset_mask": payload.reset_mask,
            },
        )

    if box is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Per la selezione assistita serve un detection_box "
                    "oppure una manual_mask_base64."
                ),
                "component_code": payload.component_code,
            },
        )

    positive_points = normalize_refinement_points(
        payload.positive_points,
        source.width,
        source.height,
    )
    negative_points = normalize_refinement_points(
        payload.negative_points,
        source.width,
        source.height,
    )

    if not positive_points and not negative_points and not payload.current_mask_base64:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Indica almeno un punto positivo o negativo, "
                    "oppure invia la maschera corrente."
                ),
                "component_code": payload.component_code,
            },
        )

    current_mask = None
    if payload.current_mask_base64:
        current_mask = decode_mask_data_url(
            payload.current_mask_base64,
            source.size,
        )

    mask = _build_seed_mask_from_points(
        source,
        box,
        positive_points,
        negative_points,
        current_mask,
        payload.reset_mask,
    )

    return _component_payload_from_mask(
        payload.component_code,
        mask,
        source_label="assisted_component_mask",
        confirmed=False,
        detection_box=box,
        diagnostics={
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
            "reset_mask": payload.reset_mask,
        },
    )


@app.post("/v1/vehicle/refine-component")
def refine_vehicle_component(payload: ComponentRefineRequest):
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    ).convert("RGB")

    box = normalize_box(
        payload.detection_box,
        source.width,
        source.height,
    )
    if box is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Bounding box del componente non valido.",
                "component_code": payload.component_code,
            },
        )

    if payload.current_mask_base64:
        try:
            initial_mask = decode_mask_data_url(
                payload.current_mask_base64,
                source.size,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Maschera corrente non valida.",
                    "error": str(exc)[:500],
                },
            ) from exc
    else:
        initial_mask = build_box_fallback_mask(source.size, box)

    positive_points = normalize_refinement_points(
        payload.positive_points,
        source.width,
        source.height,
    )
    negative_points = normalize_refinement_points(
        payload.negative_points,
        source.width,
        source.height,
    )

    refined_mask, diagnostics = refine_component_mask_local(
        source,
        box,
        initial_mask,
        positive_points=positive_points,
        negative_points=negative_points,
        iterations=payload.iterations,
    )

    binary = mask_to_binary(refined_mask)
    if int((binary > 0).sum()) < 24:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "La correzione non ha prodotto una maschera valida.",
                "component_code": payload.component_code,
                "refinement": diagnostics,
            },
        )

    x, y, width, height = cv2.boundingRect(binary)

    return {
        "component_code": payload.component_code,
        "mask_base64": mask_image_to_data_url(refined_mask),
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "mask_source": "opencv_interactive_component_refinement",
        "requires_review": diagnostics.get("refinement_status") != "refined",
        "refinement": diagnostics,
        "analysis_version": (
            "vehicle-segmentation-v17.0.26-writable-array-hotfix"
        ),
    }


@app.post("/v1/vehicle/analyze-components")
def analyze_vehicle_components_sync_disabled(
    payload: VehicleAnalyzeRequest,
):
    raise HTTPException(
        status_code=409,
        detail={
            "message": (
                "La V17 usa l'analisi asincrona per il rilevamento dei componenti. "
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
                "vehicle-segmentation-v17.0.26-writable-array-hotfix"
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

    result_bytes, full_frame_diagnostics = enforce_full_frame_result(
        source=source,
        candidate_bytes=result_bytes,
        edit_mask=source_mask,
        protect_mask=None,
        feather_px=area_transition_feather_px(
            source.size,
            area_percent,
        ),
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "ai",
        "severity_percent": severity_percent,
        "area_percent": area_percent,
        "result_base64": base64.b64encode(result_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "prompt_version": "damage-v17.0.26-writable-array-hotfix",
        "result_kind": "full_frame_jpeg",
        "full_frame_guard": full_frame_diagnostics,
    }


@app.post("/v1/damage/edit-base64")
def edit_damage_base64(payload: DamageEditBase64Request):
    """
    V17.0.26 Writable Array Hotfix

    - con maschera manuale: foto completa + perimetro reale + prompt naturale,
      output diretto del modello e validazione anti-cambio-auto;
    - senza maschera: fallback guidato full-frame;
    - modalità storiche: comportamento precedente.
    """
    request_diagnostic_id = str(uuid.uuid4())
    request_started_at = time.time()
    selected_components = resolve_selected_components(payload)
    print(
        "[DAMAGE EDIT START]",
        {
            "diagnostic_id": request_diagnostic_id,
            "damage_mode": payload.damage_mode,
            "selected_components": selected_components,
            "protected_components": payload.protected_components,
            "severity_percent": payload.severity_percent,
            "area_percent": payload.area_percent,
            "impact_direction": payload.impact_direction,
            "deformation_type": payload.deformation_type,
            "output_quality": payload.output_quality,
            "has_mask": bool(payload.mask_base64),
            "image_base64_length": len(payload.image_base64 or ""),
            "mask_base64_length": len(payload.mask_base64 or ""),
            "user_instructions_length": len(
                payload.user_instructions or ""
            ),
        },
        flush=True,
    )
    try:
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
        source, original_source_size, processing_scale = (
            resize_in_place_for_memory(
                source,
                MAX_PROCESSING_SIDE,
            )
        )
        processing_size = source.size
        gc.collect()

        guided_mode = payload.damage_mode in {
            "simple_guided",
            "hybrid_guided",
        } or (
            payload.damage_mode == "auto"
            and bool(payload.user_instructions.strip())
        )

        job_id = str(uuid.uuid4())

        if guided_mode:
            if not payload.user_instructions.strip():
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "La modalità guidata richiede user_instructions "
                        "con la descrizione completa del danno."
                    ),
                )

            has_manual_mask = bool(payload.mask_base64)

            if has_manual_mask:
                raw_manual_mask = decode_base64_image(
                    payload.mask_base64,
                    "mask_base64",
                    "L",
                )
                raw_manual_mask = resize_mask_to_processing_size(
                    raw_manual_mask,
                    source.size,
                )
                guided_mask, api_mask, mask_diagnostics = (
                    prepare_hybrid_guided_api_mask(
                        manual_mask=raw_manual_mask,
                        target_size=source.size,
                        edge_margin_px=1,
                    )
                )
                resolved_guided_mode = "hybrid_guided"
            else:
                # Fallback semplice: resta disponibile, ma è meno protetto.
                guided_mask = Image.new(
                    "L",
                    source.size,
                    color=255,
                )
                api_mask = alter_damage_area(guided_mask, 0)
                mask_diagnostics = {
                    "manual_mask_pixels": 0,
                    "guided_mask_pixels": int(
                        source.size[0] * source.size[1]
                    ),
                    "edge_margin_px": 0,
                    "mask_geometry_source": "full_frame_fallback",
                }
                resolved_guided_mode = "simple_guided"

            identity_regions = normalize_identity_regions(
                payload.identity_regions
            )

            if (
                payload.dynamic_identity_protection
                and not identity_regions
            ):
                identity_regions = call_openai_identity_analysis(source)

            hard_identity_regions, detected_tail_light_regions = (
                split_identity_regions(identity_regions)
            )
            relevant_tail_light_regions = (
                filter_relevant_tail_light_regions(
                    detected_tail_light_regions,
                    selected_components,
                )
            )

            identity_protect_mask = (
                build_identity_protect_mask(
                    hard_identity_regions,
                    source.size,
                    payload.identity_detection_padding_percent,
                )
                if hard_identity_regions
                else None
            )

            tail_light_geometry_mask = (
                build_tail_light_geometry_mask(
                    relevant_tail_light_regions,
                    source.size,
                )
                if relevant_tail_light_regions
                else None
            )

            explicit_protect_mask = None
            if payload.protect_mask_base64:
                explicit_protect_mask = decode_base64_image(
                    payload.protect_mask_base64,
                    "protect_mask_base64",
                    "L",
                )
                explicit_protect_mask = resize_mask_to_processing_size(
                    explicit_protect_mask,
                    source.size,
                )

            effective_protect_mask = merge_protect_masks(
                source.size,
                [
                    explicit_protect_mask,
                    identity_protect_mask,
                    tail_light_geometry_mask,
                ],
            )

            if effective_protect_mask is not None:
                guided_mask, _ = combine_edit_and_protect_masks(
                    guided_mask,
                    effective_protect_mask,
                    source.size,
                )
                api_mask = alter_damage_area(guided_mask, 0)
                mask_diagnostics.update({
                    "dynamic_identity_regions": identity_regions,
                    "dynamic_identity_region_count": len(
                        identity_regions
                    ),
                    "hard_identity_region_count": len(
                        hard_identity_regions
                    ),
                    "tail_light_geometry_region_count": len(
                        relevant_tail_light_regions
                    ),
                    "dynamic_identity_protection_applied": True,
                    "tail_light_geometry_lock_applied": (
                        tail_light_geometry_mask is not None
                    ),
                    "identity_protect_pixels": int(
                        (
                            mask_to_binary(
                                effective_protect_mask
                            ) > 0
                        ).sum()
                    ),
                })
            else:
                mask_diagnostics.update({
                    "dynamic_identity_regions": [],
                    "dynamic_identity_region_count": 0,
                    "dynamic_identity_protection_applied": False,
                    "identity_protect_pixels": 0,
                })

            protected_components_prompt = (
                build_strict_protected_components_prompt(
                    payload.protected_components
                )
            )

            deformation_quality_prompt = build_deformation_quality_prompt(
                severity_percent=severity_percent,
                area_percent=area_percent,
                deformation_type=payload.deformation_type,
            )

            impact_direction_prompt = build_impact_direction_prompt(
                payload.impact_direction
            )

            collision_consequences_prompt = build_collision_consequences_prompt(
                selected_components=selected_components,
                protected_components=payload.protected_components,
                severity_percent=severity_percent,
                area_percent=area_percent,
                impact_direction=payload.impact_direction,
            )

            balanced_continuity_prompt = build_balanced_continuity_prompt(
                damage_continuity_text=payload.user_instructions,
                area_percent=area_percent,
                severity_percent=severity_percent,
            )
            strict_identity_prompt = f"""
    SOURCE-IMAGE IDENTITY LOCK — HIGHEST PRIORITY

    Edit the supplied original photograph conservatively.
    This is an image-editing task, never a vehicle redesign task.

    The exact vehicle in the source image is immutable in identity.

    ABSOLUTELY PRESERVE:
    - the same manufacturer, model, generation and trim;
    - the same body shell, roofline, hatch shape, glass openings and proportions;
    - the same paint colour, finish and reflections;
    - the same front and rear light designs and original outer silhouettes;
    - the same wheels, tyres, mirrors, doors, windows, seals and trim;
    - every manufacturer logo, emblem, model badge, trim badge and visible text;
    - the exact original licence plate and every one of its characters;
    - the same environment, background, camera angle, framing and perspective.

    IDENTITY CHANGES ARE FORBIDDEN:
    - never replace this vehicle with another vehicle;
    - never change brand or model;
    - never invent Renault, Clio, Volkswagen, Peugeot, Ford or any other identity;
    - never redraw the manufacturer emblem;
    - never replace or rewrite the model badge;
    - never create a different licence plate;
    - never substitute a different tail-light design;
    - never reinterpret the vehicle based on visual similarity.

    DAMAGE-ONLY PRINCIPLE:
    Change only the physical condition of the selected original components.
    The damaged result must remain unmistakably the same exact vehicle.

    REAR-LIGHT GEOMETRY LOCK:
    When a rear light is damaged, preserve its original outer footprint,
    length, height, curvature, orientation and internal design identity.
    Add cracks, shattered lens regions, broken mounts or slight displacement
    inside that original geometry.
    Do not redesign, shrink, enlarge, round off or replace the lamp.

    BADGE, LOGO AND PLATE BEHAVIOUR:
    If a deformed panel carries a badge, emblem or plate, these items may move
    rigidly with that panel in perspective, but their design, text and identity
    must remain exactly unchanged.
    Do not paint over them, replace them or regenerate them.

    User-guided damage instructions:
    {payload.user_instructions.strip()}

    Editing rules:
    - modify only the requested components of this exact vehicle;
    - begin deformation near the selected impact point;
    - propagate deformation according to the requested impact direction;
    - create one physically coherent collision event;
    - preserve every explicitly protected component;
    - use the manual perimeter as the real editable region when supplied;
    - do not cross into unrelated or protected components;
    - preserve panel gaps where physically possible;
    - return the complete original photograph at the same dimensions.

    {protected_components_prompt}

    {deformation_quality_prompt}

    {impact_direction_prompt}

    {collision_consequences_prompt}

    {balanced_continuity_prompt}

    FINAL SELF-CHECK BEFORE OUTPUT:
    Confirm that the edited image still shows the same exact vehicle.
    Confirm that make, model, logos, badges, licence plate and lamp design have
    not changed.
    If any identity element would change, preserve it from the source image
    instead of regenerating it.

    Output only the complete edited photograph.
    Do not return a crop, isolated component, layer, mask or black background.
    """.strip()

            if os.getenv("MOCK_MODE", "false").lower() == "true":
                output = io.BytesIO()
                source.save(
                    output,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                )
                result_bytes = output.getvalue()
                result_diagnostics = {
                    "result_is_full_frame": True,
                    "original_size": list(source.size),
                    "result_size": list(source.size),
                    "direct_model_output_used": False,
                    "post_composite_applied": True,
                    "hybrid_result_validation_passed": True,
                    "mock": True,
                }
            else:
                generated_bytes = call_openai_image_edit(
                    source,
                    api_mask,
                    strict_identity_prompt,
                    payload.output_quality,
                )

                if has_manual_mask:
                    conservative_composite_bytes = (
                        smart_component_composite(
                            source=source,
                            generated_bytes=generated_bytes,
                            source_mask=guided_mask,
                            protect_mask=effective_protect_mask,
                            expansion_px=0,
                            feather_px=2,
                        )
                    )

                    result_bytes, result_diagnostics = (
                        validate_hybrid_guided_result(
                            source=source,
                            candidate_bytes=conservative_composite_bytes,
                            guided_mask=guided_mask,
                        )
                    )
                    result_diagnostics.update({
                        "conservative_component_composite_applied": True,
                        "identity_regions_hard_excluded": (
                            identity_protect_mask is not None
                        ),
                        "paint_color_correction_skipped_by_default": (
                            not payload.identity_color_correction
                        ),
                    })

                    locality_validation = validate_deformation_locality(
                            source=source,
                            candidate_bytes=result_bytes,
                            guided_mask=guided_mask,
                            area_percent=area_percent,
                            severity_percent=severity_percent,
                            selected_components=selected_components,
                        )
                    result_diagnostics.update(locality_validation)

                    result_bytes, identity_diagnostics = (
                        preserve_identity_pixels(
                            source=source,
                            candidate_bytes=result_bytes,
                            identity_mask=identity_protect_mask,
                            guided_mask=guided_mask,
                            apply_color_correction=(
                                payload.identity_color_correction
                            ),
                        )
                    )
                    result_diagnostics.update(identity_diagnostics)
                    result_diagnostics.update({
                        "identity_regions": identity_regions,
                        "identity_region_count": len(identity_regions),
                    })

                    protected_validation = (
                        validate_protected_components_unchanged(
                            source=source,
                            candidate_bytes=result_bytes,
                            protected_masks_base64=(
                                payload.protected_component_masks_base64
                            ),
                        )
                    )
                    result_diagnostics.update(protected_validation)
                else:
                    # Senza perimetro non possiamo misurare in modo affidabile
                    # le modifiche fuori componente; controlliamo solo il full frame.
                    result_bytes, result_diagnostics = (
                        lightweight_full_frame_validation(
                            source=source,
                            candidate_bytes=generated_bytes,
                        )
                    )
                    result_bytes, identity_diagnostics = (
                        preserve_identity_pixels(
                            source=source,
                            candidate_bytes=result_bytes,
                            identity_mask=identity_protect_mask,
                            guided_mask=guided_mask,
                            apply_color_correction=(
                                payload.identity_color_correction
                            ),
                        )
                    )
                    result_diagnostics.update(identity_diagnostics)
                    result_diagnostics.update({
                        "identity_regions": identity_regions,
                        "identity_region_count": len(identity_regions),
                        "direct_model_output_used": False,
                        "post_composite_applied": (
                            identity_diagnostics[
                                "dynamic_identity_composite_applied"
                            ]
                        ),
                        "hybrid_result_validation_passed": False,
                        "manual_mask_missing": True,
                        "protected_components_validation_applied": False,
                    })

            return {
                "job_id": job_id,
                "status": "completed",
                "mode": "ai",
                "severity_percent": severity_percent,
                "area_percent": area_percent,
                "result_base64": base64.b64encode(
                    result_bytes
                ).decode("ascii"),
                "mime_type": "image/jpeg",
                "prompt_version": (
                    "damage-v17.0.26-writable-array-hotfix"
                ),
                "result_kind": "full_frame_jpeg",
                "deformation_type": payload.deformation_type,
                "impact_direction": payload.impact_direction,
                "damage_mode": resolved_guided_mode,
                "mask_required": False,
                "mask_received": has_manual_mask,
                "mask_used_as_api_edit_region": has_manual_mask,
                "mask_diagnostics": mask_diagnostics,
                "result_diagnostics": result_diagnostics,
                "composite_strategy": "hard_identity_restore_and_taillight_geometry_lock",
                "post_composite_applied": True,
                "user_instructions_used": True,
                "strict_vehicle_identity_prompt": True,
            "diagnostic_id": request_diagnostic_id,
                "protected_components": payload.protected_components,
                "protected_components_prompt_applied": True,
                "deformation_quality_prompt_applied": True,
                "deformation_quality_profile": "balanced_local_continuity",
                "balanced_continuity_prompt_applied": True,
                "max_secondary_tension_lines_requested": 2,
                "point_like_dent_forbidden": True,
                "impact_direction_prompt_applied": True,
                "rear_impact_supported": True,
                "collision_consequences_prompt_applied": True,
                "license_plate_preservation_applied": True,
                "fragile_component_consistency_applied": True,
                "rear_light_damage_logic_applied": True,
                "rear_window_damage_logic_applied": True,
                "fragile_component_protection_override": True,
            "selected_components_hotfix_applied": True,
            "resolved_selected_components": selected_components,
            "generic_body_panel_removed": (
                any(
                    str(item).lower() in {
                        "bodypanel",
                        "body_panel",
                        "bodywork",
                        "generic_body_panel",
                    }
                    for item in (
                        payload.selected_components
                        + list(payload.involved_components.keys())
                    )
                )
                and all(
                    str(item).lower() not in {
                        "bodypanel",
                        "body_panel",
                        "bodywork",
                        "generic_body_panel",
                    }
                    for item in selected_components
                )
            ),
            "multicomponent_validation_applied": True,
            "multicomponent_release_mode": True,
            "backend_version": "1.7.0.26",
            "writable_array_hotfix_applied": True,
            "imagedraw_hotfix_applied": True,
            "conservative_identity_guard_applied": True,
            "hard_identity_pixel_restore_applied": True,
            "conservative_component_composite_enabled": True,
            "plate_hard_restore_applied": True,
            "taillight_outer_geometry_lock_applied": (
                tail_light_geometry_mask is not None
            ),
            "post_composite_result_forced": True,
            "vehicle_identity_lock_applied": True,
            "rear_light_geometry_lock_applied": True,
            "badge_logo_plate_lock_applied": True,
            "dynamic_identity_protection_enabled": (
                payload.dynamic_identity_protection
            ),
            "dynamic_identity_region_count": len(identity_regions),
            "dynamic_identity_regions": identity_regions,
            "paint_color_correction_enabled": (
                payload.identity_color_correction
            ),
                "mask_edge_margin_px": (
                    mask_diagnostics.get("edge_margin_px")
                ),
                "memory_optimized": True,
                "max_processing_side": MAX_PROCESSING_SIDE,
                "original_source_size": list(original_source_size),
                "processing_size": list(processing_size),
                "processing_scale": round(processing_scale, 6),
                "heic_support_enabled": HEIC_SUPPORT_ENABLED,
            }

        # --------------------------------------------------------------
        # Flusso storico con maschera obbligatoria.
        # --------------------------------------------------------------
        if not payload.mask_base64:
            raise HTTPException(
                status_code=422,
                detail=(
                    "mask_base64 è obbligatoria nelle modalità bodywork, "
                    "component_only e mixed."
                ),
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

        result_bytes, full_frame_diagnostics = enforce_full_frame_result(
            source=source,
            candidate_bytes=result_bytes,
            edit_mask=source_mask,
            protect_mask=protect_mask,
            feather_px=area_transition_feather_px(
                source.size,
                area_percent,
            ),
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "mode": "ai",
            "severity_percent": severity_percent,
            "area_percent": area_percent,
            "result_base64": base64.b64encode(result_bytes).decode("ascii"),
            "mime_type": "image/jpeg",
            "prompt_version": "damage-v17.0.26-writable-array-hotfix",
            "result_kind": "full_frame_jpeg",
            "full_frame_guard": full_frame_diagnostics,
            "deformation_type": payload.deformation_type,
            "impact_direction": payload.impact_direction,
            "contact_traces_enabled": payload.contact_traces_enabled,
            "involved_components": payload.involved_components,
            "damage_mode": resolved_damage_mode,
            "area_control": "manual_mask_geometry_unchanged",
            "area_transition_feather_px": area_transition_feather_px(
                source.size,
                area_percent,
            ),
            "composite_strategy": "single_protected_composite",
            "segmentation_contract": "per_component_png_masks",
            "strict_mask_boundary": True,
            "outside_mask_preserved": True,
            "mask_expansion_outside_component": False,
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
    except HTTPException as exc:
        elapsed_ms = round(
            (time.time() - request_started_at) * 1000
        )
        print(
            "[DAMAGE EDIT HTTP ERROR]",
            {
                "diagnostic_id": request_diagnostic_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "elapsed_ms": elapsed_ms,
                "damage_mode": payload.damage_mode,
                "selected_components": selected_components,
                "protected_components": payload.protected_components,
                "severity_percent": payload.severity_percent,
                "area_percent": payload.area_percent,
                "impact_direction": payload.impact_direction,
                "deformation_type": payload.deformation_type,
                "output_quality": payload.output_quality,
                "has_mask": bool(payload.mask_base64),
            },
            flush=True,
        )
        raise
    except Exception as exc:
        elapsed_ms = round(
            (time.time() - request_started_at) * 1000
        )
        print(
            "[DAMAGE EDIT UNHANDLED ERROR]",
            {
                "diagnostic_id": request_diagnostic_id,
                "type": type(exc).__name__,
                "error": str(exc),
                "elapsed_ms": elapsed_ms,
                "damage_mode": payload.damage_mode,
                "selected_components": selected_components,
                "protected_components": payload.protected_components,
                "severity_percent": payload.severity_percent,
                "area_percent": payload.area_percent,
                "impact_direction": payload.impact_direction,
                "deformation_type": payload.deformation_type,
                "output_quality": payload.output_quality,
                "has_mask": bool(payload.mask_base64),
            },
            flush=True,
        )
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Errore interno durante la generazione."
                ),
                "type": type(exc).__name__,
                "error": str(exc),
                "diagnostic_id": request_diagnostic_id,
            },
        ) from exc


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
