"""Code-only parts of the home sampling workflow."""
from app.flow.form import HookResult
from app.nlp import normalizers as nz


def veto_entity(ctx, entity, rec_id):
    """Imaging, ECG, biopsy etc. need a lab visit, so they cannot be part of a home sampling booking."""
    if entity != "test":
        return False
    return ctx.kit.facts("test", rec_id, ["home_collection"]).get("home_collection", "yes").lower() == "no"


def fasting_notice(ctx, session, run):
    need = []
    for t in run.slots.get("tests", []):
        f = ctx.kit.facts("test", t["id"], ["fasting", "fasting_hours"])
        if f.get("fasting", "").lower() == "yes":
            need.append((t["name"], f.get("fasting_hours") or "8"))
    if not need:
        return None
    names = ", ".join(n for n, _ in need)
    hours = max(int(h) for _, h in need if str(h).isdigit()) if any(str(h).isdigit() for _, h in need) else 8
    return f"{names} ke liye {hours} ghante kuch khana peena mana hai, pani pee sakte hain. Kya aap yeh kar payenge?"


async def submit(ctx, session, run):
    s = run.slots
    payload = {
        "tests": [{"id": t["id"], "name": t["name"]} for t in s.get("tests", [])],
        "patient_name": s.get("patient_name"),
        "age": s.get("age"),
        "gender": s.get("gender"),
        "address": s.get("address"),
        "date": s.get("date"),
        "phone": s.get("phone"),
        "channel": "voice_bot",
        "needs_human_verification": True,     # v1: a person verifies by callback
    }
    if ctx.settings.shadow_mode:
        ctx.emit("tool", {"name": "create_home_sampling_request", "shadow": True, "args": payload})
        return HookResult(say=["Aap ki request note kar li gayi hai. Hamara saathi aap se rabta karega. Shukriya."], done=True)
    res = ctx.backend.create_home_sampling_request(payload)
    ctx.emit("tool", {"name": "create_home_sampling_request", "args": payload, "ok": res.ok, "result": res.data or res.error})
    if not res.ok:
        return HookResult(
            say=["Maaf kijiye, system mein masla aa gaya hai. Main aap ki details hamare saathi ko de deti hoon, wo aap ko call karenge."],
            ok=False, transfer="backend_error")
    ref = res.data["ref"]
    digits = nz.speak_digits(ref.split("-")[-1])
    return HookResult(say=[f"Aap ki booking ho gayi hai. Reference number H S {digits} hai. Shukriya."], done=True)
