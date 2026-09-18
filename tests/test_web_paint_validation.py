import ast
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from web_paint_validation import PROFILE, validate_web_paint_colour

# Test the exact production legacy function without importing AI clients.
module = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
names = {'validate_paint_colour_consistency', 'mask_to_binary', 'resize_mask'}
namespace = {'io': io, 'np': np, 'cv2': cv2, 'Image': Image}
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), 'legacy_numeric_check', 'exec'), namespace)
legacy = namespace['validate_paint_colour_consistency']


def png(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format='PNG')
    return out.getvalue()


def fixture(rgb=(90, 90, 90)):
    source = np.empty((256, 256, 3), dtype=np.uint8)
    source[:] = rgb
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[40:210, 40:210] = 255
    return source, mask


def check(source, candidate, mask):
    return validate_web_paint_colour(Image.fromarray(source), png(candidate), Image.fromarray(mask), legacy)


def test_grey_reflection_false_positive_is_reproduced_and_corrected():
    source, mask = fixture()
    source[60:100, 60:100] = [30, 48, 66]
    candidate = source.copy()
    candidate[60:100, 60:100] = [49, 49, 49]
    old = legacy(Image.fromarray(source), png(candidate), Image.fromarray(mask))
    assert old['passed'] is False
    assert 'paint hue changed beyond tolerance' in old['failure_reasons']
    new = check(source, candidate, mask)
    assert new['passed'] is True
    assert new['validation_profile'] == PROFILE
    assert new['legacy_hsv_validation'] == old


@pytest.mark.parametrize('rgb', [(55, 55, 55), (90, 90, 90), (160, 160, 160)])
def test_unchanged_neutral_paint_passes(rgb):
    source, mask = fixture(rgb)
    assert check(source, source, mask)['passed'] is True


def test_local_shadow_and_highlight_keep_the_original_neutral_colour():
    source, mask = fixture()
    candidate = source.copy()
    candidate[70:105, 60:180] = 62
    candidate[106:140, 60:180] = 112
    assert check(source, candidate, mask)['passed'] is True


@pytest.mark.parametrize('changed', [(140, 55, 55), (55, 120, 55), (55, 55, 140), (120, 120, 55)])
def test_recolouring_neutral_panel_is_rejected(changed):
    source, mask = fixture()
    candidate = source.copy()
    candidate[mask > 0] = changed
    result = check(source, candidate, mask)
    assert result['passed'] is False
    assert 'neutral paint acquired a different colour cast' in result['failure_reasons']


@pytest.mark.parametrize('level', [8, 235])
def test_changing_grey_to_black_or_white_is_rejected(level):
    source, mask = fixture()
    candidate = source.copy()
    candidate[mask > 0] = level
    assert check(source, candidate, mask)['passed'] is False


def test_changes_outside_mask_are_still_rejected():
    source, mask = fixture()
    candidate = source.copy()
    candidate[mask == 0] = [150, 150, 150]
    result = check(source, candidate, mask)
    assert result['passed'] is False
    assert 'neutral paint outside the editable zone changed' in result['failure_reasons']


@pytest.mark.parametrize('rgb', [(180, 30, 30), (30, 160, 30), (30, 30, 180)])
def test_chromatic_panels_keep_exact_legacy_behaviour(rgb):
    source, mask = fixture(rgb)
    candidate = source.copy()
    candidate[mask > 0] = [90, 90, 90]
    old = legacy(Image.fromarray(source), png(candidate), Image.fromarray(mask))
    assert old['passed'] is False
    assert check(source, candidate, mask) == old


def test_jpeg_reencoding_does_not_change_neutral_identity():
    source, mask = fixture()
    output = io.BytesIO()
    Image.fromarray(source).save(output, format='JPEG', quality=90)
    result = validate_web_paint_colour(Image.fromarray(source), output.getvalue(), Image.fromarray(mask), legacy)
    assert result['passed'] is True


def test_broken_candidate_is_rejected():
    source, mask = fixture()
    result = validate_web_paint_colour(Image.fromarray(source), b'invalid', Image.fromarray(mask), legacy)
    assert result['passed'] is False
