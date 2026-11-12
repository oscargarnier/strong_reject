#!/usr/bin/env bash
#
# Restructure results/ from:
#   results/{model}_{dataset}_{attack}_{evaluator}/{number}/responses_and_evaluations.csv
# to:
#   results/{model}_{dataset}_{attack}/{number}/strongreject_{evaluator}_evaluations.csv
#
# Only evaluators "finetuned" and "rubric" are handled.
# Top folders that differ only by evaluator are merged into one.
#
# Usage:
#   ./restructure_results.sh [-n] [results_dir]
#     -n   dry run: print what would happen without changing anything
#     results_dir defaults to "results"
 
set -euo pipefail
 
DRY_RUN=0
if [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=1
    shift
fi
 
ROOT="${1:-results}"
EVALUATORS=("finetuned" "rubric")
SRC_NAME="responses_and_evaluations.csv"
 
if [[ ! -d "$ROOT" ]]; then
    echo "Error: directory '$ROOT' not found." >&2
    exit 1
fi
 
run() {
    if (( DRY_RUN )); then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}
 
moved=0
skipped=0
 
for evaluator in "${EVALUATORS[@]}"; do
    suffix="_${evaluator}"
 
    for top in "$ROOT"/*"$suffix"/; do
        [[ -d "$top" ]] || continue
        top="${top%/}"
        name="$(basename "$top")"
        base="${name%"$suffix"}"
        new_top="$ROOT/$base"
 
        # Guard against a name that is only the suffix
        if [[ -z "$base" ]]; then
            echo "Skipping '$top' (empty base name)" >&2
            continue
        fi
 
        for sub in "$top"/*/; do
            [[ -d "$sub" ]] || continue
            sub="${sub%/}"
            number="$(basename "$sub")"
            src="$sub/$SRC_NAME"
 
            if [[ ! -f "$src" ]]; then
                echo "Warning: '$src' not found, skipping." >&2
                skipped=$((skipped + 1))
                continue
            fi
 
            dest_dir="$new_top/$number"
            dest="$dest_dir/strongreject_${evaluator}_evaluations.csv"
 
            if [[ -e "$dest" ]]; then
                echo "Warning: '$dest' already exists, skipping '$src'." >&2
                skipped=$((skipped + 1))
                continue
            fi
 
            run mkdir -p "$dest_dir"
            run mv -- "$src" "$dest"
            echo "$src -> $dest"
            moved=$((moved + 1))
 
            # Remove the old numbered folder if it is now empty
            if (( ! DRY_RUN )); then
                rmdir "$sub" 2>/dev/null || echo "Note: '$sub' not empty, left in place." >&2
            fi
        done
 
        # Remove the old top folder if it is now empty
        if (( ! DRY_RUN )); then
            rmdir "$top" 2>/dev/null || echo "Note: '$top' not empty, left in place." >&2
        fi
    done
done
 
echo
echo "Done. Moved: $moved, skipped: $skipped."
(( DRY_RUN )) && echo "(dry run, nothing was changed)"
exit 0

