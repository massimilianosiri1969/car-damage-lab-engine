"""Web paint-tone validation for damage simulation.

Damage geometry necessarily changes highlights, shadows and local saturation.
For web simulations we compare chromatic drift in Lab at corresponding pixels
instead of HSV hue, which is unstable on grey/metallic paint. Small rendering
drift is tolerated; recolouring or a major tone shift is not. Legacy/USB
validation is untouched.
"""
from __future__ import annotations
import io
from typing import Callable
import cv2
import numpy as np
from PIL import Image

PROFILE = 'web-paint-tone-v3'

_MAX_MEDIAN_AB_DRIFT = 6.5
_MAX_P80_AB_DRIFT = 18.0
_MAX_LARGE_CHROMA_DRIFT_FRACTION = 0.28
_LARGE_CHROMA_DRIFT = 22.0
_MAX_MEDIAN_LIGHTNESS_DRIFT = 18.0
_MAX_P75_LIGHTNESS_DRIFT = 28.0
_MAX_OUTSIDE_MEDIAN_DELTA_E = 12.0
_MAX_OUTSIDE_P80_DELTA_E = 20.0


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
        if candidate.size != original.size:
            candidate = candidate.resize(original.size, Image.Resampling.LANCZOS)

        scale = min(1.0, 512.0 / max(original.size))
        size = tuple(max(1, round(value * scale)) for value in original.size)
        src = np.asarray(original.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        dst = np.asarray(candidate.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        editable = np.asarray(guided_mask.convert('L').resize(size, Image.Resampling.NEAREST)) >= 128
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        dst_lab = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)
        lightness = src_lab[:, :, 0]

        # Do not select paint by saturation: on grey metallic paint that selects
        # coloured reflections instead of the actual body colour.
        usable = (lightness >= 10.0) & (lightness <= 94.0)
        inside = editable & usable
        outside = (~editable) & usable
        count = int(inside.sum())
        if count < 300:
            return legacy

        ab_delta = dst_lab[:, :, 1:3][inside] - src_lab[:, :, 1:3][inside]
        ab_norm = np.linalg.norm(ab_delta, axis=1)
        median_ab = float(np.linalg.norm(np.median(ab_delta, axis=0)))
        p80_ab = float(np.percentile(ab_norm, 80))
        large_fraction = float(np.mean(ab_norm > _LARGE_CHROMA_DRIFT))

        src_l = src_lab[:, :, 0][inside]
        dst_l = dst_lab[:, :, 0][inside]
        median_l = float(np.median(dst_l) - np.median(src_l))
        p75_l = float(np.percentile(dst_l, 75) - np.percentile(src_l, 75))

        failures = []
        if (median_ab > _MAX_MEDIAN_AB_DRIFT or
                p80_ab > _MAX_P80_AB_DRIFT or
                large_fraction > _MAX_LARGE_CHROMA_DRIFT_FRACTION):
            failures.append('paint tone changed beyond tolerance')
        if (abs(median_l) > _MAX_MEDIAN_LIGHTNESS_DRIFT or
                abs(p75_l) > _MAX_P75_LIGHTNESS_DRIFT):
            failures.append('paint lightness changed beyond damage tolerance')

        outside_median = None
        outside_p80 = None
        if int(outside.sum()) >= 500:
            outside_de = np.linalg.norm(dst_lab[outside] - src_lab[outside], axis=1)
            outside_median = float(np.median(outside_de))
            outside_p80 = float(np.percentile(outside_de, 80))
            if (outside_median > _MAX_OUTSIDE_MEDIAN_DELTA_E or
                    outside_p80 > _MAX_OUTSIDE_P80_DELTA_E):
                failures.append('image tone outside the editable zone changed too much')

        return {
            'passed': not failures, 'applied': True,
            'validation_profile': PROFILE,
            'paint_comparison': 'corresponding-pixel-lab-drift',
            'inside_comparison_pixels': count,
            'hue_angle_used': False,
            'median_chromatic_drift_ab': round(median_ab, 3),
            'p80_chromatic_drift_ab': round(p80_ab, 3),
            'large_chromatic_drift_fraction': round(large_fraction, 4),
            'lightness_median_delta': round(median_l, 3),
            'lightness_p75_delta': round(p75_l, 3),
            'outside_delta_e_median': None if outside_median is None else round(outside_median, 3),
            'outside_delta_e_p80': None if outside_p80 is None else round(outside_p80, 3),
            'failure_reasons': failures,
            'legacy_hsv_validation': legacy,
        }
    except Exception:
        return {
            'passed': False, 'applied': False,
            'validation_profile': PROFILE,
            'failure_reasons': ['web paint validation could not be completed'],
        }
