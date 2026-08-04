#!/bin/bash
# Pull the Amarel copies of the stochastic-pendulum arms back to the iLab
# filesystem so the report can score them alongside the iLab/direct-box runs.
# Only the small per-epoch artifacts are fetched -- checkpoints stay on Amarel.
#
# The two destinations are NOT symmetric, and getting this wrong destroys data:
#
#   stoch_compare -> stoch_compare_amarel   is a pure MIRROR. Nothing else writes
#     there, so --delete is right: it removes the leftovers of a preempted run
#     that was cleared on Amarel, which would otherwise stitch a resubmitted
#     run's early epochs onto the dead run's later ones.
#
#   stoch_compare_seeds -> stoch_compare_seeds   is SHARED. Seed replicates run
#     both on Amarel and locally (westeros CPU), so a run present here but absent
#     on Amarel is a local run, NOT a stale leftover. --delete here deletes live
#     local results. It did exactly that on 2026-08-03.
REMOTE=amarel.rutgers.edu:/scratch/st1122/adaptive_roa/experiments
LOCAL=/common/users/shared/pracsys/adaptive_roa_experiments

sync_one() {  # remote_subdir local_subdir delete_flag
    mkdir -p "$LOCAL/$2"
    rsync -a $3 --info=stats1 \
          --include='*/' \
          --include='full_roa_per_point.npz' \
          --include='artifacts_v2.json' \
          --include='results.json' \
          --include='final_results.json' \
          --exclude='*' \
          "$REMOTE/$1/" "$LOCAL/$2/" >/dev/null 2>&1
    n=$(ls -d "$LOCAL/$2"/*/ 2>/dev/null | wc -l)
    # Tag ONLY on --delete. This line is the operator's signal that a destructive
    # mirror ran; flagging it for any rsync option (e.g. an --exclude) would make a
    # harmless filter look like a delete.
    case "$3" in *--delete*) tag=" [mirror, --delete]";; *) tag="";; esac
    echo "synced $1 -> $LOCAL/$2 ($n runs)$tag"
}

sync_one stoch_compare       stoch_compare_amarel --delete
sync_one stoch_compare_seeds stoch_compare_seeds  ""

#   ensemble_epistemic -> ensemble_epistemic   is SHARED, so NO --delete. The
#     Amarel half (clf_{low,med,xhigh,det}_*) and the locally-written half
#     (fm_high_* on iLab, clf_high_* on arrakis) live in the same directory and
#     write to it concurrently. --delete would erase every locally-run arm on the
#     first sync, because none of them exist on Amarel. Run names do not collide,
#     so a plain merge is correct.
#   fm_xhigh_* EXISTS TWICE. Those arms run on arrakis (writing straight into the
#     local tree) AND are queued on Amarel under the same run names. Merging the
#     Amarel copies into the same local directories would interleave two
#     independent runs inside one arm's curve -- the exact corruption --delete
#     exists to prevent, arriving by a different route. They are kept apart here
#     rather than by cancelling either copy, which is the user's call to make.
sync_one ensemble_epistemic  ensemble_epistemic   "--exclude=fm_xhigh_*"
sync_one ensemble_epistemic  ensemble_epistemic_amarel_fmxhigh "--include=fm_xhigh_*/*** --exclude=*"
