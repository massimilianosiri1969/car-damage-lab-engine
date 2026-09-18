import io
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import web_entrypoint as web


def images():
    a = np.full((256, 256, 3), 90, dtype=np.uint8)
    a[60:100, 60:100] = [30, 48, 66]
    b = a.copy()
    b[60:100, 60:100] = [49, 49, 49]
    m = np.zeros((256, 256), dtype=np.uint8)
    m[40:210, 40:210] = 255
    buf = io.BytesIO()
    Image.fromarray(b).save(buf, format='PNG')
    return Image.fromarray(a), buf.getvalue(), Image.fromarray(m)


def test_legacy_routes_and_version_are_preserved():
    with TestClient(web.app) as client:
        assert client.get('/v1/version').json()['version'] == '1.7.0.23'
        version = client.get('/v1/web/version').json()
        assert version['version'] == '2.0.0'
        assert version['paint_colour_validation_profile'] == 'web-paint-tone-v3'
        assert version['paint_semantic_policy'] == 'numeric-web-validator-authoritative'
        assert version['legacy_endpoints_unchanged'] is True
        assert client.post('/v1/web/damage/edit-base64/start', json={}).status_code == 422


def test_web_scope_is_reset_even_on_failure():
    def fail():
        assert web._web_validation.get() is True
        raise RuntimeError('test')
    with pytest.raises(RuntimeError):
        web._run_web_task(fail)
    assert web._web_validation.get() is False


def test_legacy_and_web_tasks_can_run_concurrently_without_cross_talk():
    args = images()
    barrier = Barrier(2)
    def evaluate():
        barrier.wait(timeout=10)
        return web.core.validate_paint_colour_consistency(*args)['passed']
    with ThreadPoolExecutor(max_workers=2) as executor:
        web_result = executor.submit(web._run_web_task, evaluate)
        legacy_result = executor.submit(evaluate)
        assert web_result.result(timeout=10) is True
        assert legacy_result.result(timeout=10) is False
    assert web._web_validation.get() is False


def test_web_semantic_identity_ignores_only_paint_subflags(monkeypatch):
    observed = {
        'passed': False,
        'same_make': True, 'same_model': True, 'same_license_plate': True,
        'same_manufacturer_emblem': True, 'same_model_badges': True,
        'same_paint_colour': False, 'same_paint_hue': False,
        'same_paint_saturation': False, 'same_paint_brightness': False,
        'tail_light_outer_geometry_preserved': True,
        'vehicle_identity_changed': False, 'skipped': False,
    }
    monkeypatch.setattr(web, '_legacy_identity', lambda *args: dict(observed))
    args = images()
    result = web._run_web_task(web._identity_dispatch, args[0], args[1], ['hood'])
    assert result['passed'] is True
    assert result['same_paint_colour'] is False  # observation retained for audit
    assert result['paint_tone_deferred_to_numeric_web_validator'] is True


@pytest.mark.parametrize('field', [
    'same_make', 'same_model', 'same_license_plate',
    'same_manufacturer_emblem', 'same_model_badges',
    'tail_light_outer_geometry_preserved',
])
def test_web_semantic_identity_never_relaxes_non_paint_identity(monkeypatch, field):
    observed = {
        'passed': False,
        'same_make': True, 'same_model': True, 'same_license_plate': True,
        'same_manufacturer_emblem': True, 'same_model_badges': True,
        'same_paint_colour': True, 'same_paint_hue': True,
        'same_paint_saturation': True, 'same_paint_brightness': True,
        'tail_light_outer_geometry_preserved': True,
        'vehicle_identity_changed': False, 'skipped': False,
    }
    observed[field] = False
    monkeypatch.setattr(web, '_legacy_identity', lambda *args: dict(observed))
    args = images()
    result = web._run_web_task(web._identity_dispatch, args[0], args[1], ['hood'])
    assert result['passed'] is False


def test_web_semantic_identity_never_allows_vehicle_identity_change(monkeypatch):
    observed = {
        'passed': False,
        'same_make': True, 'same_model': True, 'same_license_plate': True,
        'same_manufacturer_emblem': True, 'same_model_badges': True,
        'same_paint_colour': True, 'same_paint_hue': True,
        'same_paint_saturation': True, 'same_paint_brightness': True,
        'tail_light_outer_geometry_preserved': True,
        'vehicle_identity_changed': True, 'skipped': False,
    }
    monkeypatch.setattr(web, '_legacy_identity', lambda *args: dict(observed))
    args = images()
    assert web._run_web_task(web._identity_dispatch, args[0], args[1], ['hood'])['passed'] is False


def test_background_queue_uses_web_profile_only_for_new_endpoint(monkeypatch):
    args = images()
    def fake_generation(payload):
        return {'paint_colour_validation': web.core.validate_paint_colour_consistency(*args)}
    monkeypatch.setattr(web.core, 'edit_damage_base64', fake_generation)
    payload = {'image_base64': 'synthetic-image-payload', 'severity_percent': 25, 'area_percent': 50}
    with TestClient(web.app) as client:
        for endpoint, expected in [('/v1/damage/edit-base64/start', False), ('/v1/web/damage/edit-base64/start', True)]:
            response = client.post(endpoint, json=payload)
            assert response.status_code == 200
            job_id = response.json()['job_id']
            state = client.get('/v1/damage/edit-base64/status/' + job_id).json()
            assert state['status'] == 'succeeded'
            assert state['result']['paint_colour_validation']['passed'] is expected
            client.delete('/v1/damage/edit-base64/status/' + job_id)
    assert web._web_validation.get() is False
