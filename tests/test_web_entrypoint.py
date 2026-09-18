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
        assert client.get('/v1/web/version').json()['paint_colour_validation_profile'] == 'web-neutral-paint-v1'
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


def test_background_queue_uses_web_profile_only_for_new_endpoint(monkeypatch):
    args = images()
    # A stub returns the result of the actual numeric check. No AI requests.
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
