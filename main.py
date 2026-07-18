import base64
import io
import json
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
    version="1.6.0.2",
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


def normalize_polygon_points(
    raw_polygon,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    if not isinstance(raw_polygon, list):
        return []

    points: list[tuple[int, int]] = []

    for item in raw_polygon:
        if isinstance(item, dict):
            raw_x = item.get("x")
            raw_y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_x, raw_y = item[0], item[1]
        else:
            continue

        try:
            x_value = float(raw_x)
            y_value = float(raw_y)
        except Exception:
            continue

        # Il modello restituisce coordinate normalizzate 0..1000.
        x = round((x_value / SEGMENTATION_POLYGON_SCALE) * (width - 1))
        y = round((y_value / SEGMENTATION_POLYGON_SCALE) * (height - 1))

        points.append(
            (
                max(0, min(width - 1, x)),
                max(0, min(height - 1, y)),
            )
        )

    return points if len(points) >= 3 else []


def polygon_mask(
    image_size: tuple[int, int],
    points: list[tuple[int, int]],
) -> Image.Image:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(points) >= 3:
        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)

    return Image.fromarray(mask, mode="L")


def clean_component_mask(mask: Image.Image) -> Image.Image:
    binary = mask_to_binary(mask)
    height, width = binary.shape

    kernel_size = max(3, round(min(width, height) * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

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

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            selected[labels == index] = 255

    if int((selected > 0).sum()) == 0:
        selected = cleaned

    return Image.fromarray(selected, mode="L")


def refine_mask_with_grabcut(
    source: Image.Image,
    initial_mask: Image.Image,
) -> Image.Image:
    if not SEGMENTATION_GRABCUT_ENABLED:
        return clean_component_mask(initial_mask)

    reduced_source, scale = resize_image_for_processing(
        source.convert("RGB"),
        SEGMENTATION_GRABCUT_MAX_SIDE,
    )

    reduced_mask = initial_mask.convert("L")
    if scale != 1.0:
        reduced_mask = reduced_mask.resize(
            reduced_source.size,
            Image.Resampling.NEAREST,
        )

    rgb = np.asarray(reduced_source, dtype=np.uint8)
    initial = mask_to_binary(reduced_mask)

    if int((initial > 0).sum()) < 64:
        result = clean_component_mask(reduced_mask)
        return upscale_mask_to_original(result, source.size)

    grabcut_mask = np.full(initial.shape, cv2.GC_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(
        initial,
        np.ones((7, 7), np.uint8),
        iterations=1,
    )
    sure_fg = cv2.erode(
        initial,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    grabcut_mask[probable_fg > 0] = cv2.GC_PR_FGD
    grabcut_mask[sure_fg > 0] = cv2.GC_FGD

    probable_bg = cv2.dilate(
        probable_fg,
        np.ones((21, 21), np.uint8),
        iterations=1,
    )
    grabcut_mask[probable_bg == 0] = cv2.GC_BGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.grabCut(
            bgr,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )

        refined = np.where(
            (grabcut_mask == cv2.GC_FGD)
            | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        allowed = cv2.dilate(
            initial,
            np.ones((15, 15), np.uint8),
            iterations=1,
        )
        refined = cv2.bitwise_and(refined, allowed)

        if int((refined > 0).sum()) < max(
            32,
            int((initial > 0).sum() * 0.25),
        ):
            refined = initial

        cleaned = clean_component_mask(
            Image.fromarray(refined, mode="L")
        )
        return upscale_mask_to_original(cleaned, source.size)

    except Exception:
        cleaned = clean_component_mask(reduced_mask)
        return upscale_mask_to_original(cleaned, source.size)

    finally:
        del rgb
        del initial
        del grabcut_mask
        del probable_fg
        del sure_fg
        del probable_bg
        del background_model
        del foreground_model
        if "bgr" in locals():
            del bgr


def build_component_segmentation(
    source: Image.Image,
    component: dict,
) -> dict | None:
    code = str(component.get("code", "")).strip()
    if code not in VEHICLE_COMPONENT_CATALOG:
        return None

    polygon = normalize_polygon_points(
        component.get("polygon", []),
        source.width,
        source.height,
    )

    if not polygon:
        return None

    initial_mask = polygon_mask(source.size, polygon)
    refined_mask = refine_mask_with_grabcut(source, initial_mask)

    binary = mask_to_binary(refined_mask)
    pixel_area = int((binary > 0).sum())

    if pixel_area < max(
        16,
        round(source.width * source.height * SEGMENTATION_MIN_AREA_RATIO),
    ):
        return None

    x, y, width, height = cv2.boundingRect(binary)

    try:
        confidence = float(component.get("confidence", 0.70))
    except Exception:
        confidence = 0.70

    return {
        "code": code,
        "label": VEHICLE_COMPONENT_CATALOG[code],
        "category": component_category(code),
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
        "mask_base64": mask_image_to_data_url(refined_mask),
        "bounding_box": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "polygon": [
            {"x": int(point[0]), "y": int(point[1])}
            for point in polygon
        ],
        "mask_source": "vision_polygon_grabcut",
    }


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


def normalize_vehicle_analysis(
    raw: dict,
    source: Image.Image,
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

    segmented_components: list[dict] = []
    seen: set[str] = set()

    if isinstance(raw_components, list):
        for item in raw_components:
            if not isinstance(item, dict):
                continue

            code = str(item.get("code", "")).strip()
            if code in seen:
                continue

            segmentation = build_component_segmentation(source, item)
            if segmentation is None:
                continue

            seen.add(code)
            segmented_components.append(segmentation)

    segmented_components.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return {
        "view": vehicle_view,
        "view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "vehicle_view": vehicle_view,
        "vehicle_view_label": VEHICLE_VIEW_LABELS[vehicle_view],
        "components": segmented_components,
        # Compatibilità con il vecchio frontend.
        "visible_components": [
            {
                "code": item["code"],
                "label": item["label"],
                "category": item["category"],
                "confidence": item["confidence"],
                "mask_base64": item["mask_base64"],
                "bounding_box": item["bounding_box"],
            }
            for item in segmented_components
        ],
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
      "polygon": [
        {{"x": integer from 0 to 1000, "y": integer from 0 to 1000}}
      ]
    }}
  ]
}}

Polygon rules:
- Coordinates are normalized to the complete image: top-left is 0,0 and bottom-right is 1000,1000.
- Trace the visible outer contour of the physical component, not an ellipse and not a generic bounding box.
- Use 8 to 24 polygon points when possible.
- Follow actual panel seams, lamp edges, glass edges, wheel outline and bumper boundaries.
- Do not include the floor, workshop equipment, shadows, another vehicle or empty background.
- Components may be partially visible; trace only the visible portion.
- Do not overlap unrelated components unnecessarily.
- Return only components genuinely visible in the photograph.
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
                "analysis_version": "vehicle-segmentation-v16.0.2",
            },
        ) from fallback_exc


@app.post("/v1/vehicle/analyze-components")
def analyze_vehicle_components(payload: VehicleAnalyzeRequest):
    source = decode_base64_image(
        payload.image_base64,
        "image_base64",
        "RGB",
    )

    analysis_source, _ = resize_image_for_processing(
        source,
        ANALYSIS_MAX_SIDE,
    )
    image_data_url = image_to_data_url(analysis_source)

    try:
        raw_analysis = call_openai_vehicle_analysis(image_data_url)
    finally:
        del analysis_source
        del image_data_url

    normalized = normalize_vehicle_analysis(raw_analysis, source)

    if not normalized["components"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Non è stato possibile ottenere maschere attendibili. "
                    "Prova con una foto più nitida o aggiungi manualmente "
                    "il componente."
                ),
                "raw_component_count": len(
                    raw_analysis.get(
                        "components",
                        raw_analysis.get("visible_components", []),
                    )
                    if isinstance(raw_analysis, dict)
                    else []
                ),
                "analysis_version": "vehicle-segmentation-v16.0.2",
            },
        )

    return {
        **normalized,
        "model": os.getenv(
            "OPENAI_VISION_MODEL",
            "gpt-4.1-mini",
        ),
        "analysis_version": "vehicle-segmentation-v16.0.2",
        "mask_format": "data:image/png;base64",
        "mask_semantics": "white_component_black_background",
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
        "prompt_version": "damage-v16.0.2-render-free-memory-optimized",
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
        "prompt_version": "damage-v16.0.2-render-free-memory-optimized",
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
