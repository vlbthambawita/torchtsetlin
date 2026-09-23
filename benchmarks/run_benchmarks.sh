#!/usr/bin/env bash
#
# Run the torchtsetlin benchmark suites and regenerate the documentation figures.
#
#   ./benchmarks/run_benchmarks.sh                  # every suite, cpu + cuda:0, docs updated
#   ./benchmarks/run_benchmarks.sh --quick          # ~1 min smoke test, docs untouched
#   ./benchmarks/run_benchmarks.sh -s batch models  # just those suites
#   ./benchmarks/run_benchmarks.sh --plots-only     # re-plot the results already on disk
#
# Suites are run one at a time so that each one is merged into the results JSON as soon as
# it finishes: a crash (or a Ctrl-C) never costs more than the suite that was running.
# The full run takes roughly 20-25 minutes on a 16-thread CPU with an RTX 3090.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONUNBUFFERED=1   # stdout is a pipe into tee; without this progress appears in bursts

ALL_SUITES=(batch small clauses features models feedback threads transfer phases segmentation mnist)
QUICK_SUITES=(small phases)

# ------------------------------------------------------------------------------- defaults
suites=()
devices=()
results=""
figures=""
tables=""
page=""
epochs=3
data_root="./data"
quick=0
do_bench=1
do_plots=1
do_page=1
fresh=0
fail_fast=0
dry_run=0

bold=""; dim=""; red=""; green=""; reset=""
if [[ -t 1 ]]; then
    bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; green=$'\033[32m'; reset=$'\033[0m'
fi

usage() {
    cat <<EOF
${bold}usage:${reset} benchmarks/run_benchmarks.sh [options]

Runs benchmarks/bench_device.py over the requested suites, then benchmarks/plot_benchmarks.py
to render the figures, tables and the "CPU vs GPU" documentation page.

${bold}suites${reset} (--suites, default: all)
  batch      throughput vs mini-batch size, MNIST-shaped flat classifier
  small      the same for a Noisy-XOR scale machine (12 features, 20 clauses)
  clauses    throughput and memory vs clause budget
  features   throughput vs number of Boolean features
  models     the model zoo (flat / coalesced / conv / conv-coalesced / regression)
  feedback   feedback_mode="batch" vs "sequential"
  threads    CPU thread scaling (GPU shown as a reference line)
  transfer   dataset resident on the device vs copied from the host per batch
  phases     where the time of one update() goes
  mnist      end-to-end MNIST epochs, wall clock + test accuracy (needs torchvision)

${bold}options${reset}
  -s, --suites LIST     suites to run, space or comma separated, or "all" (default: all)
  -d, --devices LIST    devices, e.g. "cpu cuda:0" (default: cpu, plus cuda:0 when available)
  -o, --results FILE    results JSON (default: benchmarks/results/cpu_vs_gpu.json)
  -F, --figures DIR     figure output directory (default: docs/assets/benchmarks)
  -T, --tables FILE     markdown tables (default: benchmarks/results/tables.md)
  -P, --page FILE       documentation page to write (default: docs/cpu-vs-gpu.md)
      --epochs N        epochs for the mnist suite (default: ${epochs})
      --data-root DIR   dataset root for the mnist suite (default: ${data_root})
      --quick           cheap subset (${QUICK_SUITES[*]}) into benchmarks/results/quick.json,
                        figures beside it, documentation left alone - use it to smoke-test
      --fresh           move an existing results JSON aside instead of merging into it
      --plots-only      skip benchmarking, just re-render from the results JSON
      --bench-only      benchmark only, no figures, tables or page
      --no-page         render figures and tables but do not rewrite the documentation page
      --fail-fast       stop at the first failing suite (default: keep going, report at the end)
  -n, --dry-run         print the commands instead of running them
  -h, --help            this message

${bold}environment${reset}
  PYTHON                interpreter to use (default: python)
  CUDA_VISIBLE_DEVICES  restrict which GPUs "cuda:0" refers to
EOF
}

die() { printf '%s\n' "${red}error:${reset} $*" >&2; exit 2; }

hms() { local s=$1; if [[ $s -ge 60 ]]; then printf '%dm%02ds' $((s / 60)) $((s % 60));
        else printf '%ds' "$s"; fi; }

# "a,b c" -> the items of $1 in the global _SPLIT array
_SPLIT=()
split_list() {
    local IFS=', ' item
    _SPLIT=()
    set -f                      # the split below must not glob
    for item in $1; do
        [[ -n "$item" ]] && _SPLIT+=("$item")
    done
    set +f
}

# --------------------------------------------------------------------------------- parse
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--suites)
            shift; [[ $# -gt 0 ]] || die "--suites needs a value"
            while [[ $# -gt 0 && "$1" != -* ]]; do
                split_list "$1"
                if [[ ${#_SPLIT[@]} -gt 0 ]]; then suites+=("${_SPLIT[@]}"); fi
                shift
            done ;;
        -d|--devices)
            shift; [[ $# -gt 0 ]] || die "--devices needs a value"
            while [[ $# -gt 0 && "$1" != -* ]]; do
                split_list "$1"
                if [[ ${#_SPLIT[@]} -gt 0 ]]; then devices+=("${_SPLIT[@]}"); fi
                shift
            done ;;
        -o|--results)  results="${2:?--results needs a path}"; shift 2 ;;
        -F|--figures)  figures="${2:?--figures needs a path}"; shift 2 ;;
        -T|--tables)   tables="${2:?--tables needs a path}"; shift 2 ;;
        -P|--page)     page="${2:?--page needs a path}"; shift 2 ;;
        --epochs)      epochs="${2:?--epochs needs a number}"; shift 2 ;;
        --data-root)   data_root="${2:?--data-root needs a path}"; shift 2 ;;
        --quick)       quick=1; shift ;;
        --fresh)       fresh=1; shift ;;
        --plots-only)  do_bench=0; shift ;;
        --bench-only)  do_plots=0; do_page=0; shift ;;
        --no-page)     do_page=0; shift ;;
        --fail-fast)   fail_fast=1; shift ;;
        -n|--dry-run)  dry_run=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             die "unknown option '$1' (try --help)" ;;
    esac
done

cd "$REPO_ROOT"

# ------------------------------------------------------------------------------- defaults
if [[ ${#suites[@]} -eq 0 ]]; then
    if [[ $quick -eq 1 ]]; then suites=("${QUICK_SUITES[@]}"); else suites=(all); fi
fi
if [[ " ${suites[*]} " == *" all "* ]]; then suites=("${ALL_SUITES[@]}"); fi
for s in "${suites[@]}"; do
    [[ " ${ALL_SUITES[*]} " == *" $s "* ]] || die "unknown suite '$s' (try --help)"
done

if [[ $quick -eq 1 ]]; then
    results="${results:-benchmarks/results/quick.json}"
    figures="${figures:-benchmarks/results/quick-figures}"
    tables="${tables:-benchmarks/results/quick-tables.md}"
    epochs=1
    [[ -n "$page" ]] || do_page=0
fi
results="${results:-benchmarks/results/cpu_vs_gpu.json}"
figures="${figures:-docs/assets/benchmarks}"
tables="${tables:-benchmarks/results/tables.md}"
page="${page:-docs/cpu-vs-gpu.md}"
log="${results%.json}.log"

# ------------------------------------------------------------------------- sanity checks
"$PYTHON" - <<'PY' || die "the interpreter cannot import torch and torchtsetlin — install with: pip install -e '.[dev]'"
import importlib, sys
for mod in ("torch", "torchtsetlin"):
    try:
        importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover - user environment
        print(f"cannot import {mod}: {exc}", file=sys.stderr)
        sys.exit(1)
PY

if [[ $do_plots -eq 1 ]] && ! "$PYTHON" -c "import matplotlib" 2>/dev/null; then
    die "plotting needs matplotlib — install with: pip install matplotlib (or pip install -e '.[dev]')"
fi

if [[ ${#devices[@]} -eq 0 ]]; then
    while IFS= read -r dev; do devices+=("$dev"); done < <("$PYTHON" -c "
import torch
print('cpu')
if torch.cuda.is_available():
    print('cuda:0')
")
fi

if [[ " ${suites[*]} " == *" mnist "* ]] && ! "$PYTHON" -c "import torchvision" 2>/dev/null; then
    printf '%s\n' "${red}note:${reset} the mnist suite needs torchvision — dropping it from this run" >&2
    filtered=(); for s in "${suites[@]}"; do [[ "$s" == mnist ]] || filtered+=("$s"); done
    suites=("${filtered[@]}")
    [[ ${#suites[@]} -gt 0 ]] || die "nothing left to run"
fi

if [[ $dry_run -eq 0 ]]; then
    mkdir -p "$(dirname "$results")" "$figures" "$(dirname "$tables")"
fi

run() {  # echo the command, then run it unless --dry-run
    printf '%s\n' "${dim}\$ $*${reset}"
    [[ $dry_run -eq 1 ]] && return 0
    "$@"
}

# ----------------------------------------------------------------------------------- go
if [[ $dry_run -eq 0 ]]; then
    exec > >(tee -a "$log") 2>&1
fi

printf '\n%s\n' "${bold}torchtsetlin benchmarks${reset}  $(date '+%Y-%m-%d %H:%M:%S')"
printf '  repo      %s\n' "$REPO_ROOT"
printf '  python    %s (%s)\n' "$("$PYTHON" -c 'import sys;print(sys.version.split()[0])')" \
                               "$(command -v "$PYTHON")"
"$PYTHON" -c "
import torch, torchtsetlin as tt
print(f'  torch     {torch.__version__}  (cuda {torch.version.cuda or \"n/a\"}, {torch.get_num_threads()} threads)')
print(f'  library   torchtsetlin {tt.__version__}')
for i in range(torch.cuda.device_count()):
    print(f'  cuda:{i}    {torch.cuda.get_device_name(i)}')
"
printf '  suites    %s\n' "${suites[*]}"
printf '  devices   %s\n' "${devices[*]}"
printf '  results   %s\n' "$results"
[[ $do_plots -eq 1 ]] && printf '  figures   %s\n' "$figures"
[[ $do_plots -eq 1 ]] && printf '  tables    %s\n' "$tables"
[[ $do_page  -eq 1 ]] && printf '  page      %s\n' "$page"
printf '  log       %s\n\n' "$log"

if [[ $fresh -eq 1 && -f "$results" && $dry_run -eq 0 ]]; then
    mv -v "$results" "$results.bak"
fi

t_start=$SECONDS
failed=()

if [[ $do_bench -eq 1 ]]; then
    i_suite=0
    for suite in "${suites[@]}"; do
        i_suite=$((i_suite + 1))
        printf '\n%s %s\n' "${bold}=== suite $i_suite/${#suites[@]}: $suite ===${reset}" \
               "${dim}($(hms $((SECONDS - t_start))) into the run)${reset}"
        t_suite=$SECONDS
        args=(benchmarks/bench_device.py --suites "$suite" --devices "${devices[@]}"
              --out "$results")
        [[ "$suite" == mnist ]] && args+=(--epochs "$epochs" --root "$data_root")
        if run "$PYTHON" "${args[@]}"; then
            [[ $dry_run -eq 0 ]] &&
                printf '%s\n' "${green}ok${reset} $suite in $(hms $((SECONDS - t_suite)))"
        else
            printf '%s\n' "${red}FAILED${reset} $suite after $(hms $((SECONDS - t_suite)))"
            failed+=("$suite")
            [[ $fail_fast -eq 1 ]] && break
        fi
    done
fi

if [[ $do_plots -eq 1 ]]; then
    if [[ $dry_run -eq 0 && ! -f "$results" ]]; then
        die "no results at $results — run without --plots-only first"
    fi
    printf '\n%s\n' "${bold}=== figures and tables ===${reset}"
    plot_args=(benchmarks/plot_benchmarks.py --results "$results" --out "$figures"
               --tables "$tables")
    if [[ $do_page -eq 1 ]]; then plot_args+=(--page "$page"); else plot_args+=(--no-page); fi
    run "$PYTHON" "${plot_args[@]}" || { failed+=(plots); }
fi

# ------------------------------------------------------------------------------- summary
elapsed=$((SECONDS - t_start))
printf '\n%s  %dm %02ds\n' "${bold}done${reset}" $((elapsed / 60)) $((elapsed % 60))
printf '  results  %s\n' "$results"
if [[ $do_plots -eq 1 && $dry_run -eq 0 ]]; then
    printf '  figures  %s (%s png)\n' "$figures" "$(find "$figures" -name '*.png' | wc -l)"
    printf '  tables   %s\n' "$tables"
    [[ $do_page -eq 1 ]] && printf '  page     %s\n' "$page"
    printf '\n  preview with: mkdocs serve\n'
fi

if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\n%s %s\n' "${red}failed:${reset}" "${failed[*]}"
    printf '  (everything that did finish is in %s — re-run just the failures with -s %s)\n' \
           "$results" "${failed[*]}"
    exit 1
fi
