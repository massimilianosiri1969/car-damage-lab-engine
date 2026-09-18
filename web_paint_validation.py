"""Neutral-paint numeric check for the opt-in web generation endpoint.

The legacy HSV check samples saturated pixels only. On a grey panel these
can be coloured reflections rather than paint. This module measures neutral
paint in Lab, retains the chromatic legacy path and never changes image pixels.
The independent semantic identity and protected-region checks remain in core.
"""
from __future__ import annotations
import io
from typing import Callable
import cv2
import numpy as np
from PIL import Image

PROFILE = 'web-neutral-paint-v1'


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
        # Bounded diagnostic memory; size/locality are checked independently.
        scale = min(1.0, 512.0 / max(original.size))
        size = tuple(max(1, round(value * scale)) for value in original.size)
        src = np.asarray(original.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        dst = np.asarray(candidate.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        editable = np.asarray(guided_mask.convert('L').resize(size, Image.Resampling.NEAREST)) >= 128
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        dst_lab = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)
        source_chroma = np.linalg.norm(src_lab[:, :, 1:3], axis=2)
        midtones = (src_lab[:, :, 0] >= 12.0) & (src_lab[:, :, 0] <= 90.0)
        inside = editable & midtones
        count = int(inside.sum())
        if count < 200:
            return legacy
        neutral_fraction = float(np.mean(source_chroma[inside] <= 12.0))
        if neutral_fraction < 0.80 or float(np.median(source_chroma[inside])) > 10.0:
            return legacy

        ab_delta = dst_lab[:, :, 1:3][inside] - src_lab[:, :, 1:3][inside]
        cast_shift = float(np.linalg.norm(np.median(ab_delta, axis=0)))
        chroma_delta_p80 = float(np.percentile(np.linalg.norm(ab_delta, axis=1), 80))
        candidate_chroma = float(np.median(np.linalg.norm(dst_lab[:, :, 1:3][inside], axis=1)))
        lightness_delta = float(np.median(dst_lab[:, :, 0][inside]) - np.median(src_lab[:, :, 0][inside]))
        highlight_delta = float(np.percentile(dst_lab[:, :, 0][inside], 75) - np.percentile(src_lab[:, :, 0][inside], 75))
        failures = []
        if cast_shift > 4.5 or candidate_chroma > 12.0 or chroma_delta_p80 > 12.0:
            failures.append('neutral paint acquired a different colour cast')
        if abs(lightness_delta) > 12.0 and abs(highlight_delta) > 15.0:
            failures.append('neutral paint brightness changed beyond local deformation tolerance')
        for reason in legacy.get('failure_reasons', []):
            if reason == 'paint outside the editable zone changed':
                failures.append(reason)
        outside = (~editable) & midtones & (source_chroma <= 12.0)
        outside_delta = None
        if int(outside.sum()) >= 200:
            outside_delta = float(np.median(np.linalg.norm(dst_lab[outside] - src_lab[outside], axis=1)))
            if outside_delta > 6.0:
                failures.append('neutral paint outside the editable zone changed')
        return {
            'passed': not failures, 'applied': True,
            'validation_profile': PROFILE,
            'source_paint_profile': 'achromatic',
            'inside_paint_pixels': count,
            'source_neutral_fraction': round(neutral_fraction, 4),
            'hue_angle_used': False,
            'neutral_cast_delta_ab': round(cast_shift, 3),
            'chroma_delta_p80': round(chroma_delta_p80, 3),
            'candidate_chroma_median': round(candidate_chroma, 3),
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
