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
# A duplicate \newcommand is not an undefined macro: pdflatex keeps the FIRST definition and
# carries on, so the wrong number reaches the page with no visible sign.
d=$(grep -c "already defined" "$log" || true)
if [ "$d" -gt 0 ]; then
  echo "FAIL: $d duplicate macro definition(s) (first definition silently wins):"
  grep -oE "Command .[A-Za-z]+ already defined" "$log" | sort -u | head -20
  bad=1
fi
# A table or plot whose data file is missing is dropped with only a warning buried in the
# log, so the page loses a figure and the build still "succeeds".
for f in $(grep -ohE "data/[A-Za-z0-9_.-]+\.csv" sections/*.tex | sort -u); do
  if [ ! -f "$f" ]; then echo "FAIL: missing data file $f"; bad=1; fi
done
if grep -q "LaTeX Error" "$log"; then
  echo "FAIL: LaTeX error(s):"; grep -m5 "LaTeX Error" "$log"; bad=1
fi
python audit_macros.py || bad=1
python check_spacing.py || bad=1
[ "$bad" -eq 0 ] && echo "OK: no undefined macros, references, duplicates or missing data"
exit $bad
