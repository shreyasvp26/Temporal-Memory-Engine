"""Second-Brain Bitemporal Memory & Observability Workbench UI router.

Provides an enterprise-grade internal engineering console for inspecting,
querying, and mutating bitemporal memory graphs, live SSE chat streaming,
and multi-tenant row-level security isolation.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

ui_router = APIRouter(tags=['workbench-ui'])

_HTML_PATH = Path(__file__).parent / 'workbench.html'
WORKBENCH_HTML = _HTML_PATH.read_text(encoding='utf-8')


@ui_router.get('/', response_class=HTMLResponse)
@ui_router.get('/ui', response_class=HTMLResponse)
async def serve_workbench_ui() -> HTMLResponse:
    """Serve the single-page Bitemporal Memory & Observability Workbench UI."""
    return HTMLResponse(content=WORKBENCH_HTML, media_type='text/html')
