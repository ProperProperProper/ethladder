"""Dashboard API for OMLX candidate patterns extracted from the (separate,
external) training-inbox pipeline — see
symbot_python/exchange/inbox_pattern_extractor.py's module docstring for
the full picture of what that pipeline is and isn't connected to.

Promote/reject here are the ONLY way a candidate's status ever changes —
a human decision made through the dashboard. Nothing in
inbox_pattern_extractor_loop (run_everything.py) or this router ever
changes OMLX's actual trading behavior; promoting a candidate just marks
it reviewed, the same as it does for a backtest winner needing a human
to actually wire its params into a deployed config.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from symbot_python.exchange import inbox_pattern_extractor

router = APIRouter()


class ReviewRequest(BaseModel):
    note: str | None = None


@router.get("/api/inbox-candidates")
async def api_inbox_candidates() -> dict:
    return inbox_pattern_extractor.get_status()


@router.post("/api/inbox-candidates/{candidate_id}/promote")
async def api_promote_candidate(candidate_id: str, body: ReviewRequest) -> dict:
    ok = inbox_pattern_extractor.promote_candidate(candidate_id, note=body.note)
    return {"ok": ok}


@router.post("/api/inbox-candidates/{candidate_id}/reject")
async def api_reject_candidate(candidate_id: str, body: ReviewRequest) -> dict:
    ok = inbox_pattern_extractor.reject_candidate(candidate_id, note=body.note)
    return {"ok": ok}
