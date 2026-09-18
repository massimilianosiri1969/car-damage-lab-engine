"""Dedicated web adapter.

The web/legacy distinction is carried explicitly in the validated request
model. This avoids relying on ContextVar propagation through FastAPI/Starlette
background worker threads. Legacy and USB requests default to "legacy".
"""
from __future__ import annotations
from fastapi import BackgroundTasks
import main as core
from web_paint_validation import PROFILE

app = core.app


@app.get('/v1/web/version')
def web_version():
    return {
        'service': 'Car Damage Lab web adapter',
        'version': '5.0.0',
        'paint_colour_validation_profile': PROFILE,
        'paint_profile_routing': 'explicit-request-field',
        'base44_contract_18_bridge': True,
        'legacy_endpoints_unchanged': True,
        'semantic_identity_required': True,
        'async_start_endpoint': '/v1/web/damage/edit-base64/start',
        'async_status_endpoint': '/v1/damage/edit-base64/status/{job_id}',
    }


@app.post('/v1/web/damage/edit-base64/start')
def start_web_generation(payload: core.DamageEditBase64Request, background_tasks: BackgroundTasks):
    web_payload = payload.model_copy(update={'paint_validation_profile': 'web_v3'})
    return core.start_async_damage_generation(web_payload, background_tasks)
