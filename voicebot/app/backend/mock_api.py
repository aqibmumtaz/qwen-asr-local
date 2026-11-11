"""Mock booking backend with the same interface a real API client will expose.

Swap `MockBackend` for an HTTP client later (BOOKING_API_MODE=live) without touching workflows.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class BackendResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


SLOT_WINDOWS = {
    "morning": ("08:00", "11:00", "subah aath se gyarah baje"),
    "afternoon": ("12:00", "15:00", "dopahar barah se teen baje"),
    "evening": ("16:00", "19:00", "shaam chaar se saat baje"),
    "night": ("19:00", "21:00", "raat saat se nau baje"),
}


class MockBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self._seq = itertools.count(1001)
        self.bookings: dict[str, dict] = {}
        self.fail_next: str | None = None          # UI switch: simulate backend failure
        self.blocked_slots: set[tuple[str, str]] = set()   # (iso date, part) unavailable
        self._seed()

    def set_today(self, today: date) -> None:
        """Seeded demo booking HS-1000 is dated 'today' (the session's date, which scenarios pin for determinism)."""
        self._seed(today)

    def _seed(self, day: date | None = None) -> None:
        today = (day or date.today()).isoformat()
        self.bookings["HS-1000"] = {
            "ref": "HS-1000", "patient_name": "Ahmed Raza", "phone": "03001234567", "tests": ["Complete Blood Count"],
            "address": "Gulshan-e-Iqbal Block 13 D", "date": today, "part": "morning", "status": "confirmed",
        }

    # ── slots ──
    def get_slots(self, day: date) -> BackendResult:
        out = []
        for part, (a, b, label) in SLOT_WINDOWS.items():
            if (day.isoformat(), part) not in self.blocked_slots:
                out.append({"part": part, "from": a, "to": b, "label": label})
        return BackendResult(True, {"date": day.isoformat(), "slots": out})

    def slot_available(self, day: date, part: str) -> bool:
        return (day.isoformat(), part) not in self.blocked_slots and part in SLOT_WINDOWS

    # ── booking ──
    def _maybe_fail(self) -> BackendResult | None:
        if self.fail_next:
            msg, self.fail_next = self.fail_next, None
            return BackendResult(False, error=msg)
        return None

    def create_home_sampling_request(self, payload: dict) -> BackendResult:
        if (f := self._maybe_fail()):
            return f
        with self._lock:
            ref = f"HS-{next(self._seq)}"
            self.bookings[ref] = {"ref": ref, "status": "confirmed", "created": datetime.now().isoformat(), **payload}
        return BackendResult(True, {"ref": ref})

    def get_booking(self, ref: str) -> BackendResult:
        if (f := self._maybe_fail()):
            return f
        b = self.bookings.get(ref.upper())
        return BackendResult(True, dict(b)) if b else BackendResult(False, error="not_found")

    def reschedule(self, ref: str, day: date, part: str) -> BackendResult:
        if (f := self._maybe_fail()):
            return f
        b = self.bookings.get(ref.upper())
        if not b or b["status"] == "cancelled":
            return BackendResult(False, error="not_found")
        if not self.slot_available(day, part):
            return BackendResult(False, error="slot_unavailable")
        b["date"], b["part"] = day.isoformat(), part
        return BackendResult(True, dict(b))

    def cancel(self, ref: str) -> BackendResult:
        if (f := self._maybe_fail()):
            return f
        b = self.bookings.get(ref.upper())
        if not b:
            return BackendResult(False, error="not_found")
        b["status"] = "cancelled"
        return BackendResult(True, dict(b))


def get_backend():
    return MockBackend()
