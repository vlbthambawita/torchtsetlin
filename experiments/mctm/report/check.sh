#!/usr/bin/env bash
# Fail loudly if pdflatex silently dropped a macro or a reference.
set -u
log="${1:-main.log}"
bad=0
n=$(grep -c "Undefined control sequence" "$log" || true)
if [ "$n" -gt 0 ]; then
  echo "FAIL: $n undefined control sequence(s):"
  grep -A1 "Undefined control sequence" "$log" | grep -oE '\\[A-Za-z]+ *$' | sort -u | head -20
  bad=1
fi
if grep -q "There were undefined references" "$log"; then
  echo "FAIL: undefined references"; bad=1
fi
[ "$bad" -eq 0 ] && echo "OK: no undefined macros or references"
exit $bad
