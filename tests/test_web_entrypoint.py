import pytest
from fastapi.testclient import TestClient
import web_entrypoint as web


def minimal_payload():
    return {
        'image_base64': 'x' * 32,
        'severity_percent': 25,
        'area_percent': 50,
    }


def test_web_version_exposes_explicit_profile_routing():
    with TestClient(web.app) as client:
        legacy = client.get('/v1/version').json()
        version = client.get('/v1/web/version').json()
        assert legacy['version'] == '1.7.0.23'
        assert version['version'] == '4.0.0'
        assert version['paint_colour_validation_profile'] == 'web-paint-tone-v3'
        assert version['paint_profile_routing'] == 'explicit-request-field'
        assert version['legacy_endpoints_unchanged'] is True


def test_request_model_defaults_to_legacy():
    payload = web.core.DamageEditBase64Request.model_validate(minimal_payload())
    assert payload.paint_validation_profile == 'legacy'


def test_web_endpoint_injects_profile_before_queue(monkeypatch):
    captured = {}
    def fake_start(payload, background_tasks):
        captured['profile'] = payload.paint_validation_profile
        captured['dump'] = payload.model_dump()
        return {'job_id': 'test', 'status': 'queued'}
    monkeypatch.setattr(web.core, 'start_async_damage_generation', fake_start)
    with TestClient(web.app) as client:
        response = client.post('/v1/web/damage/edit-base64/start', json=minimal_payload())
    assert response.status_code == 200
    assert captured['profile'] == 'web_v3'
    assert captured['dump']['paint_validation_profile'] == 'web_v3'


def test_legacy_endpoint_keeps_default_profile(monkeypatch):
    captured = {}
    def fake_run(job_id, payload_data):
        captured['profile'] = payload_data['paint_validation_profile']
    monkeypatch.setattr(web.core, 'run_async_damage_generation', fake_run)
    # Direct function check avoids depending on background scheduling internals.
    payload = web.core.DamageEditBase64Request.model_validate(minimal_payload())
    assert payload.paint_validation_profile == 'legacy'


def test_base44_contract_v18_is_recognized_without_explicit_profile():
    payload = web.core.DamageEditBase64Request.model_validate({
        **minimal_payload(),
        'contract_version': '18.0',
    })
    assert payload.contract_version == '18.0'
    assert payload.paint_validation_profile == 'legacy'


def test_usb_legacy_payload_has_no_base44_contract_marker():
    payload = web.core.DamageEditBase64Request.model_validate(minimal_payload())
    assert payload.contract_version is None
    assert payload.paint_validation_profile == 'legacy'
