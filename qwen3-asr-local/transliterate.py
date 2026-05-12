#!/usr/bin/env python3
"""
Hindi Devanagari → Roman Urdu transliteration — Python wrapper.

Per project convention, Python wrappers shell out to the .sh entry point
(transliterate.sh) so there's a single source of truth for the CLI shape.

Public API:
    from transliterate import transliterate
    roman = transliterate("मेरा नाम अकीब है")

CLI:
    python3 transliterate.py "मेरा नाम"
    echo "मेरा नाम" | python3 transliterate.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR       = Path(__file__).resolve().parent
TRANSLITERATE_SH = SCRIPT_DIR / "transliterate.sh"


def transliterate(text: str) -> str:
    """Hindi Devanagari → Roman Urdu. Calls transliterate.sh internally."""
    if not text:
        return ""
    result = subprocess.run(
        ["bash", str(TRANSLITERATE_SH), text],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip('\n')


def main():
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = sys.stdin.read().rstrip('\n')
    print(transliterate(text))


if __name__ == "__main__":
    main()
