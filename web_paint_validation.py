"""Web paint-tone validation for damage simulation.

A dent necessarily changes local lightness, shadows and reflections. The web
validator therefore compares stable Lab chromaticity instead of HSV hue on
near-neutral paint, while still rejecting recolouring and excessive global
lightness shifts. Legacy/USB requests keep the original validator.
"""
from __future__ import annotations
import io
from typing import Callable
import cv2
import numpy as np
from PIL import Image

PROFILE = 'web-paint-tone-v2'

# Deliberately conservative: "almost the same tone", not pixel identity.
_NEUTRAL_CHROMA = 18.0
_MIN_NEUTRAL_FRACTION = 0.55
_MAX_CAST_SHIFT_AB = 5.5
_MAX_CANDIDATE_CHROMA = 18.0
_MAX_CHROMA_DELTA_P80 = 15.0
_MAX_LIGHTNESS_MEDIAN_DELTA = 18.0
_MAX_LIGHTNESS_P75_DELTA = 24.0
_MAX_OUTSIDE_DELTA_E = 6.0


def validate_web_paint_colour(source: Image.Image, candidate_bytes: bytes,
                              guided_mask: Image.Image | None,
                              legacy_validator: Callable) -> dict[str, object]:
    legacy = legacy_validator(source, candidate_bytes, guided_mask)
    if guided_mask is None or legacy.get('reason') == 'candidate_decode_error':
        return legacy
    try:
        with Image.open(io.BytesIO(candidate_bytes)) as opened:
            candidate = opened.convert('RGB')
            candidate.load()
        original = source.convert('RGB')
        scale = min(1.0, 512.0 / max(original.size))
        size = tuple(max(1, round(value * scale)) for value in original.size)
        src = np.asarray(original.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        dst = np.asarray(candidate.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        editable = np.asarray(guided_mask.convert('L').resize(size, Image.Resampling.NEAREST)) >= 128
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        dst_lab = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)
        src_chroma = np.linalg.norm(src_lab[:, :, 1:3], axis=2)
        midtones = (src_lab[:, :, 0] >= 10.0) & (src_lab[:, :, 0] <= 92.0)
        inside = editable & midtones
        count = int(inside.sum())
        if count < 200:
            return legacy

        neutral_fraction = float(np.mean(src_chroma[inside] <= _NEUTRAL_CHROMA))
        source_chroma_median = float(np.median(src_chroma[inside]))
        # Coloured paint keeps the established V17 validator. The web-specific
        # path is only for neutral/near-neutral bodywork where HSV hue is
        # mathematically unstable and reflections dominate saturation.
        if neutral_fraction < _MIN_NEUTRAL_FRACTION or source_chroma_median > 14.0:
            return legacy

        ab_delta = dst_lab[:, :, 1:3][inside] - src_lab[:, :, 1:3][inside]
        cast_shift = float(np.linalg.norm(np.median(ab_delta, axis=0)))
        chroma_delta_p80 = float(np.percentile(np.linalg.norm(ab_delta, axis=1), 80))
        candidate_chroma_median = float(np.median(np.linalg.norm(dst_lab[:, :, 1:3][inside], axis=1)))
        lightness_delta = float(np.median(dst_lab[:, :, 0][inside]) - np.median(src_lab[:, :, 0][inside]))
        highlight_delta = float(np.percentile(dst_lab[:, :, 0][inside], 75) - np.percentile(src_lab[:, :, 0][inside], 75))

        failures = []
        if (cast_shift > _MAX_CAST_SHIFT_AB or
                candidate_chroma_median > _MAX_CANDIDATE_CHROMA or
                chroma_delta_p80 > _MAX_CHROMA_DELTA_P80):
            failures.append('neutral paint acquired a different colour cast')
        # Geometry may move highlights substantially, but the whole edited
        # paint region must remain close in apparent tone.
        if (abs(lightness_delta) > _MAX_LIGHTNESS_MEDIAN_DELTA or
                abs(highlight_delta) > _MAX_LIGHTNESS_P75_DELTA):
            failures.append('neutral paint lightness changed beyond damage tolerance')

        # Outside the editable region we remain strict: no recolouring or
        # global relighting is justified by the dent.
        outside = (~editable) & midtones & (src_chroma <= _NEUTRAL_CHROMA)
        outside_delta = None
        if int(outside.sum()) >= 200:
            outside_delta = float(np.median(np.linalg.norm(dst_lab[outside] - src_lab[outside], axis=1)))
            if outside_delta > _MAX_OUTSIDE_DELTA_E:
                failures.append('neutral paint outside the editable zone changed')
        for reason in legacy.get('failure_reasons', []):
            if reason == 'paint outside the editable zone changed':
                failures.append(reason)

        return {
            'passed': not failures,
            'applied': True,
            'validation_profile': PROFILE,
            'source_paint_profile': 'near-neutral',
            'inside_paint_pixels': count,
            'source_neutral_fraction': round(neutral_fraction, 4),
            'source_chroma_median': round(source_chroma_median, 3),
            'hue_angle_used': False,
            'neutral_cast_delta_ab': round(cast_shift, 3),
            'chroma_delta_p80': round(chroma_delta_p80, 3),
            'candidate_chroma_median': round(candidate_chroma_median, 3),
            'lightness_median_delta': round(lightness_delta, 3),
            'lightness_p75_delta': round(highlight_delta, 3),
            'outside_neutral_delta_e_median': None if outside_delta is None else round(outside_delta, 3),
            'failure_reasons': list(dict.fromkeys(failures)),
            'legacy_hsv_validation': legacy,
        }
    except Exception:
        return {
            'passed': False, 'applied': False,
            'validation_profile': PROFILE,
            'failure_reasons': ['web paint validation could not be completed'],
        }
