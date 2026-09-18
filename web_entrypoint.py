"""Opt-in web adapter; the V17 core and legacy URLs remain unchanged.

Only tasks submitted to /v1/web/ use neutral-paint diagnostics. ContextVar
scopes the validator to each worker, including concurrent legacy/USB requests.
All semantic identity, plate, protected-mask and locality checks remain in core.
"""
from __future__ import annotations
from contextvars import ContextVar
from fastapi import BackgroundTasks
import main as core
from web_paint_validation import PROFILE, validate_web_paint_colour

app = core.app
_web_validation = ContextVar('web_paint_validation', default=False)
_legacy_colour = core.validate_paint_colour_consistency


def _colour_dispatch(source, candidate_bytes, guided_mask):
    if _web_validation.get():
        return validate_web_paint_colour(source, candidate_bytes, guided_mask, _legacy_colour)
    return _legacy_colour(source, candidate_bytes, guided_mask)


# One-time dispatch registration, not a mutable per-request global switch.
core.validate_paint_colour_consistency = _colour_dispatch


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
        'service': 'Car Damage Lab web adapter', 'version': '1.0.0',
        'paint_colour_validation_profile': PROFILE,
        'legacy_endpoints_unchanged': True,
        'semantic_identity_required': True,
        'async_start_endpoint': '/v1/web/damage/edit-base64/start',
        'async_status_endpoint': '/v1/damage/edit-base64/status/{job_id}',
    }


@app.post('/v1/web/damage/edit-base64/start')
def start_web_generation(payload: core.DamageEditBase64Request, background_tasks: BackgroundTasks):
    return core.start_async_damage_generation(payload, _WebBackgroundTasks(background_tasks))
