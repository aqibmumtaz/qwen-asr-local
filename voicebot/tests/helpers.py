from __future__ import annotations

from datetime import date

from app.backend.mock_api import MockBackend
from app.config import Settings
from app.datakit.resolve import DataKit
from app.faq import FaqBook
from app.flow.form import Workflow
from app.flow.types import Session, TurnCtx
from app.persona import persona_vars
from app.config import ROOT

TODAY = date(2026, 10, 21)   # a Wednesday


def make_ctx(llm=None, settings: Settings | None = None, backend=None) -> TurnCtx:
    s = settings or Settings(llm_base_url="")
    return TurnCtx(kit=DataKit(settings=s), llm=llm, backend=backend or MockBackend(), faq=FaqBook(settings=s),
                   settings=s, today=TODAY, persona=persona_vars(s))


def load(name: str) -> Workflow:
    return Workflow(ROOT / "workflows" / name)


async def run_flow(wf: Workflow, ctx: TurnCtx, lines: list[str], session: Session | None = None):
    """Drive one workflow with caller lines; the first line is the utterance that started it.
    Returns (session, run, transcript, last_out)."""
    session = session or Session()
    run = wf.new_run()
    session.active = run
    transcript = []
    out = await wf.start(ctx, session, run, lines[0] if lines else None)
    transcript.append(("user", lines[0] if lines else ""))
    transcript.append(("bot", " ".join(out.replies)))
    for line in lines[1:]:
        if out.done or out.transfer:
            break
        out = await wf.handle(ctx, session, run, line)
        transcript.append(("user", line))
        transcript.append(("bot", " ".join(out.replies)))
    return session, run, transcript, out
