#!/bin/bash
# Hindi Devanagari → Roman Urdu transliteration (shell CLI).
#
# Wraps the hindi_to_roman_urdu engine module (the actual algorithm is Python
# because it requires Unicode normalisation + complex regex that bash can't do).
# This script is the single shell-facing entry point — all .sh scripts call it.
#
# Usage:
#   bash transliterate.sh "मेरा नाम अकीब है"        # arg form
#   echo "मेरा नाम" | bash transliterate.sh         # stdin form
#   ROMAN=$(bash transliterate.sh "मेरा नाम")        # capture
#
# Output: Roman Urdu text to stdout (single line)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Read input from positional args (joined with spaces) or stdin
if [[ $# -gt 0 ]]; then
    INPUT="$*"
else
    INPUT=$(cat)
fi

cd "$SCRIPT_DIR"
python3 -c "
import sys, os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
from hindi_to_roman_urdu import transliterate
print(transliterate(sys.argv[1]))
" "$INPUT"
