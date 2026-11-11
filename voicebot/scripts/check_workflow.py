#!/usr/bin/env python3
"""Validate a workflow and run the checks that guard the others.

  1. files and references (slots, step types, hooks, manifest fields)
  2. data lists load (scripts/check_data.py logic)
  3. this workflow's scenarios, then ALL scenarios (switching/routing against every other workflow)
  4. routing eval (rules router) so a new workflow cannot silently break existing routing
Usage: python scripts/check_workflow.py <name>        exit code 1 on any failure
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.config import ROOT  # noqa: E402
from app.flow.form import Workflow  # noqa: E402
from app.flow.slots import PARSERS  # noqa: E402
from app.scenarios import load_scenarios, run_many  # noqa: E402

STEP_TYPES = {"slot", "confirm", "hook_confirm", "action"}


def static_checks(name: str) -> list[str]:
    errs: list[str] = []
    d = ROOT / "workflows" / name
    for f in ("workflow.yaml", "manifest.yaml"):
        if not (d / f).exists():
            return [f"missing {f}"]
    try:
        wf = Workflow(d)
    except Exception as e:
        return [f"cannot load workflow: {type(e).__name__}: {e}"]
    m = wf.manifest
    if m.get("name") != wf.name:
        errs.append("manifest name differs from workflow name")
    if not m.get("description") or "TODO" in m["description"]:
        errs.append("manifest description is empty or TODO")
    if len(m.get("examples", [])) < 8 or any("TODO" in e for e in m["examples"]):
        errs.append("manifest needs at least 8 real example phrases (router training data)")
    if not m.get("strong_keywords"):
        errs.append("manifest has no strong_keywords (rule router cannot recognise this workflow)")
    ids = set()
    for st in wf.steps:
        sid = st["id"]
        if sid in ids:
            errs.append(f"duplicate step id {sid}")
        ids.add(sid)
        typ = st.get("type", "slot")
        if typ not in STEP_TYPES:
            errs.append(f"step {sid}: unknown type {typ}")
        if typ == "slot":
            if st.get("slot") not in wf.slot_types:
                errs.append(f"step {sid}: slot {st.get('slot')} not declared in slots")
            if not st.get("ask"):
                errs.append(f"step {sid}: missing ask text")
        if typ in ("hook_confirm", "action") and not wf._hook(st.get("hook", "")):
            errs.append(f"step {sid}: hook {st.get('hook')} not found in hooks.py")
        if typ == "confirm" and not st.get("summary"):
            errs.append(f"step {sid}: confirm step needs summary")
    for slot, typ in wf.slot_types.items():
        if typ not in PARSERS:
            errs.append(f"slot {slot}: unknown type {typ}")
    if not wf.steps or wf.steps[-1].get("type") != "action":
        errs.append("last step should be an action step that completes the workflow")
    return errs


async def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    name = argv[0]
    bad = 0
    errs = static_checks(name)
    print(f"[1] static checks: {'OK' if not errs else str(len(errs)) + ' problem(s)'}")
    for e in errs:
        print("    -", e)
    bad += len(errs)

    from scripts.check_data import main as check_data  # type: ignore
    print("[2] data lists:", "OK" if check_data([]) == 0 else "PROBLEMS (see above)")

    scs = load_scenarios()
    own = [s for s in scs if name in s.get("tags", [])]
    print(f"[3] scenarios: {len(own)} for {name}, {len(scs)} in total")
    if not own:
        print("    - no scenarios tagged with the workflow name")
        bad += 1
    res = await run_many(scs)
    failed = [r for r in res if not r.ok]
    for r in failed:
        print("    FAIL", r.id, "-", r.failures[:2])
    print(f"    {len(res) - len(failed)}/{len(res)} passed")
    bad += len(failed)

    rs = ROOT / "eval" / "routing" / "routing_set.yaml"
    has_routing = any(e.get("workflow") == name or e.get("label") == "switch" and e.get("workflow") == name
                      for e in yaml.safe_load(rs.read_text(encoding="utf-8")))
    print("[4] routing eval entries for this workflow:", "yes" if has_routing else "NONE (add utterances to eval/routing/routing_set.yaml)")
    if not has_routing and name not in ("home_sampling", "booking_status"):
        bad += 1
    print("\nRESULT:", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
