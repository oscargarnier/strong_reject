#!/usr/bin/env bash
#
# Second pass of the results/ restructuring.
#
# Leftover state:
#   results/{base}_{evaluator}/{number}/{jailbroken_prompts.csv,responses.csv,...}
# Target state:
#   results/{base}/{number}/{jailbroken_prompts.csv,responses.csv,...}
#
# Since both evaluator folders (finetuned, rubric) hold copies of the same files,
# the second copy hits an existing destination:
#   - identical content -> the leftover duplicate is deleted
#   - different content -> left in place and a warning is printed
# Empty old folders are removed at the end.
#
# Usage:
#   ./finish_restructure.sh [-n] [results_dir]
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
deduped=0
conflicts=0
removed_dirs=0

for evaluator in "${EVALUATORS[@]}"; do
    suffix="_${evaluator}"

    for top in "$ROOT"/*"$suffix"/; do
        [[ -d "$top" ]] || continue
        top="${top%/}"
        name="$(basename "$top")"
        base="${name%"$suffix"}"
        [[ -n "$base" ]] || continue
        new_top="$ROOT/$base"

        for sub in "$top"/*/; do
            [[ -d "$sub" ]] || continue
            sub="${sub%/}"
            number="$(basename "$sub")"
            dest_dir="$new_top/$number"

            for src in "$sub"/*; do
                [[ -f "$src" ]] || continue
                fname="$(basename "$src")"
                dest="$dest_dir/$fname"

                if [[ -e "$dest" ]]; then
                    if cmp -s -- "$src" "$dest"; then
                        run rm -- "$src"
                        echo "Duplicate removed: $src"
                        deduped=$((deduped + 1))
                    else
                        echo "Warning: '$dest' exists with different content, leaving '$src'." >&2
                        conflicts=$((conflicts + 1))
                    fi
                else
                    run mkdir -p "$dest_dir"
                    run mv -- "$src" "$dest"
                    echo "$src -> $dest"
                    moved=$((moved + 1))
                fi
            done

            # Remove the numbered folder if empty
            if (( ! DRY_RUN )); then
                if rmdir "$sub" 2>/dev/null; then
                    removed_dirs=$((removed_dirs + 1))
                else
                    echo "Note: '$sub' not empty, left in place." >&2
                fi
            fi
        done

        # Remove the old top folder if empty
        if (( ! DRY_RUN )); then
            if rmdir "$top" 2>/dev/null; then
                removed_dirs=$((removed_dirs + 1))
            else
                echo "Note: '$top' not empty, left in place." >&2
            fi
        fi
    done
done

echo
echo "Done. Moved: $moved, duplicates removed: $deduped, conflicts: $conflicts, folders removed: $removed_dirs."
(( DRY_RUN )) && echo "(dry run, nothing was changed; folder removals are not simulated)"
exit 0
