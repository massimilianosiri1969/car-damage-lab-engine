"""Opt-in web adapter; V17 core and legacy/USB URLs remain unchanged.

Web damage simulation uses a paint validator that tolerates local reflection
changes caused by geometry. Semantic identity remains strict for make, model,
plate, emblems, badges and vehicle architecture. Paint tone is decided by the
numeric web validator so the vision model cannot reject a valid dent merely
because its highlights changed.
"""
from __future__ import annotations
from contextvars import ContextVar
from fastapi import BackgroundTasks
import main as core
from web_paint_validation import PROFILE, validate_web_paint_colour

app = core.app
_web_validation = ContextVar('web_paint_validation', default=False)
_legacy_colour = core.validate_paint_colour_consistency
_legacy_identity = core.call_openai_identity_validation


def _colour_dispatch(source, candidate_bytes, guided_mask):
    if _web_validation.get():
        return validate_web_paint_colour(source, candidate_bytes, guided_mask, _legacy_colour)
    return _legacy_colour(source, candidate_bytes, guided_mask)


def _identity_dispatch(source, candidate_bytes, selected_components):
    result = _legacy_identity(source, candidate_bytes, selected_components)
    if not _web_validation.get() or result.get('skipped') or result.get('reason') == 'candidate_decode_error':
        return result
    # On the web route, paint sub-flags are advisory only. The numeric web
    # validator runs immediately afterwards and is the authoritative paint-tone
    # gate. All non-paint identity checks remain mandatory.
    non_paint_ok = all([
        bool(result.get('same_make')),
        bool(result.get('same_model')),
        bool(result.get('same_license_plate')),
        bool(result.get('same_manufacturer_emblem')),
        bool(result.get('same_model_badges')),
        bool(result.get('tail_light_outer_geometry_preserved')),
        not bool(result.get('vehicle_identity_changed')),
    ])
    original_passed = bool(result.get('passed'))
    result['semantic_identity_passed_without_paint'] = non_paint_ok
    result['semantic_identity_original_passed'] = original_passed
    result['paint_tone_deferred_to_numeric_web_validator'] = True
    result['passed'] = non_paint_ok
    # Do not falsify the model's observations: retain all original paint flags.
    return result


# Register dispatch once. ContextVar keeps concurrent legacy/USB tasks isolated.
core.validate_paint_colour_consistency = _colour_dispatch
core.call_openai_identity_validation = _identity_dispatch


def _run_web_task(function, *args, **kwargs):
    token = _web_validation.set(True)
    try:
        return function(*args, **kwargs)
    finally:
        _web_validation.reset(token)


class _WebBackgroundTasks:
    def __init__(self, tasks: BackgroundTasks):
        self.tasks = tasks

    def add_task(self, function, *args, **kwargs):
        self.tasks.add_task(_run_web_task, function, *args, **kwargs)


@app.get('/v1/web/version')
def web_version():
    return {
        'service': 'Car Damage Lab web adapter',
        'version': '2.0.0',
        'paint_colour_validation_profile': PROFILE,
        'paint_semantic_policy': 'numeric-web-validator-authoritative',
        'legacy_endpoints_unchanged': True,
        'semantic_identity_required': True,
        'async_start_endpoint': '/v1/web/damage/edit-base64/start',
        'async_status_endpoint': '/v1/damage/edit-base64/status/{job_id}',
    }


@app.post('/v1/web/damage/edit-base64/start')
def start_web_generation(payload: core.DamageEditBase64Request, background_tasks: BackgroundTasks):
    return core.start_async_damage_generation(payload, _WebBackgroundTasks(background_tasks))
