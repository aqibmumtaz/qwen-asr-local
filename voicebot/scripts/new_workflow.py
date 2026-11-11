#!/usr/bin/env python3
"""Scaffold a new workflow folder from a template.  Usage: python scripts/new_workflow.py <name> "<title>" """
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORKFLOW = '''# Workflow: {title}. Templates are spoken Roman Urdu; {{sakti}}/{{leti}} come from the persona.
name: {name}
spoken: "{title}"   # how the bot refers to this task out loud (Roman Urdu)
title: {title}
intro: "Theek hai, shuru karte hain."
slots:
  # slot_key: {{type: tests|name|age|gender|address|datetime|phone|booking_ref|action|text}}
  patient_name: {{type: name}}
steps:
  - id: patient_name
    slot: patient_name
    ask: "Mareez ka naam kya hai?"
    retry: "Naam dobara bata dein."
  - id: confirm
    type: confirm
    non_interruptible: true
    summary: "Confirm kar {{leti}} hoon: {{patient_name}}. Sab theek hai?"
  - id: submit
    type: action
    hook: submit
'''
MANIFEST = '''name: {name}
description: "TODO: one sentence on what the caller wants to get done in this workflow."
verification: none        # none | booking_ref_and_phone | ... (add verification before exposing patient data)
keywords: []
strong_keywords: []       # words that clearly signal THIS workflow (used by the rule router)
examples:                 # 10-20 ways callers ask for this, in Roman Urdu with spelling variants
  - "TODO example phrase"
'''
HOOKS = '''from app.flow.form import HookResult


async def submit(ctx, session, run):
    # call the backend here (ctx.backend), then speak the result
    return HookResult(say=["Aap ki request note kar li gayi hai. Shukriya."], done=True)
'''
SCENARIO = '''- id: {name}_happy_path
  title: {title}, everything clear
  tags: [{name}]
  steps:
    - say: "TODO first caller line"
      expect: {{router: switch, workflow: {name}}}
'''


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    name, title = argv[0], argv[1]
    d = ROOT / "workflows" / name
    if d.exists():
        print(f"{d} already exists")
        return 1
    d.mkdir(parents=True)
    (d / "workflow.yaml").write_text(WORKFLOW.format(name=name, title=title))
    (d / "manifest.yaml").write_text(MANIFEST.format(name=name))
    (d / "hooks.py").write_text(HOOKS)
    (ROOT / "eval" / "scenarios" / f"10_{name}.yaml").write_text(SCENARIO.format(name=name, title=title))
    print(f"created {d} and eval/scenarios/10_{name}.yaml\nnext: edit the files, then python scripts/check_workflow.py {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
