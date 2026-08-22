#!/usr/bin/env bash
# Operator front door for the experiment — run on jobe from any cwd. This
# wrapper changes to the repository root before doing anything, and every
# model-facing command is CUDA + bf16 + flash-attn only.
#
# QUICK USAGE
#   scripts/lab.sh gate                     tests + ruff + identity + gradient gates
#   scripts/lab.sh probe  [TRAIN FLAGS]     four-step foreground config smoke
#   scripts/lab.sh train  TAG [TRAIN FLAGS] probe, train, score; detached one-run queue
#   scripts/lab.sh queue  FILE              same for every line of FILE, in order
#   scripts/lab.sh score  TAG... [SCORE FLAGS]
#                                            score existing checkpoints in foreground
#   scripts/lab.sh status                   marker plus recent training summaries
#   scripts/lab.sh watch                    follow filtered queue/eval/error output
#   scripts/lab.sh kill                     stop this queue and matching training Python
#
# COMMAND BEHAVIOR
#   gate
#     Runs `pytest tests -q`, `ruff check .`, the alpha=0 HF identity gate,
#     and the BPTT-vs-reference gradient gate, in that order. `&&` makes the
#     first failure stop the chain. It deliberately does not run the slower
#     untrained-perplexity repro gate.
#   probe [TRAIN FLAGS]
#     Runs train.py in the foreground after appending `--steps 4` and
#     `--out /tmp/probe-lab.pt`; those two wrapper values therefore win over
#     duplicates supplied by the caller. This includes full shape warmup and
#     the zero-recompile/frozen-base audits, then performs four optimizer steps.
#   train TAG [TRAIN FLAGS]
#     Writes a one-line queue file at logs/queue-TAG.txt and launches it with
#     nohup. Refuses to stack on a live queue worker or any matching train.py
#     run. TAG becomes a filename stem, so use a whitespace-free, path-safe
#     label. To attach SCORE FLAGS, pass a literal/quoted `|` followed by them;
#     for anything elaborate, prefer an explicit queue file.
#   queue FILE
#     Launches FILE under nohup, serially. The worker continues past a failed
#     probe, train, or score and records the failure; DONE means the file was
#     exhausted, not that every entry succeeded.
#   score TAG... [SCORE FLAGS]
#     Runs scripts/score.py in the foreground and tees the combined output to
#     logs/score-YYYYMMDD-HHMMSS.log. Every TAG names data/train/TAG.pt.
#   status
#     Prints the last eight lines of logs/queue-STATUS, reports whether any
#     matching training Python is active, and shows the last three step/eval
#     lines from each of the two newest non-score/probe/queue logs.
#   watch
#     Follows only newly appended lines from logs/queue.log and queue-STATUS,
#     filtered to run banners, evals, warmup, exits, DONE/FAILED, tracebacks,
#     OOMs, kills, and interrupts. It waits until interrupted.
#   kill
#     Stops the PID in logs/queue.pid (if live), removes that PID file, then
#     pkill's every process whose command matches the experiment's train `run`
#     invocation and appends a hand-killed marker. This is host-wide matching,
#     not TAG-specific.
#   _run-queue FILE
#     Internal foreground worker used by nohup; not an operator entry point.
#
# TRAIN FLAGS
# These are forwarded to `python -m recirculated_dot_experiment.train run`.
# Defaults below are train.py defaults. Comma-separated values contain no
# spaces. The selected task and k are homogeneous within each optimizer batch.
#
#   --model ID_OR_PATH       Frozen pretrained model (default
#                            google/gemma-3-1b-pt). An HF id or local path.
#   --source N               Zero-indexed layer whose activation is recirculated
#                            (default 11).
#   --dest N                 Zero-indexed layer input receiving the norm-matched
#                            source activation through the learned gate (default 4).
#   --condition ARM          Training arm: `dots+wire` trains through the
#                            think-scope recirculation wire; `dots` is the
#                            dots-alone parallel control (default dots+wire).
#   --tasks T1,T2,...        Training/eval task mixture, sampled uniformly by
#                            task per step (default parity). Names: s5_chain,
#                            parity, reachability, threesum.
#   --k K1,K2,...            Periodic-eval dot budgets (default 1,2,4,8,16,32).
#                            The max-k span runs once and smaller k values are
#                            causal-prefix readouts. This does not choose train k.
#   --train-k K1,K2,...      Per-step training dot budgets (default 4,8,16,32).
#                            k<=2 is eval-only by default because no refreshed
#                            column reaches a supervised logit before t=3.
#   --k-gamma G              Samples train k with P(k) proportional to k**G
#                            (default 1); 0 makes the listed k values uniform.
#   --batch N                Effective optimizer batch (default 512). It must be
#                            positive; the execution policy may split it into
#                            equal-shape CUDA microbatches without changing the
#                            batch-mean objective.
#   --steps N                Final within-run step number (default 2000), not an
#                            additional-step count: resume continues at saved+1
#                            and stops at N.
#   --lr X                   AdamW peak learning rate (default 1e-3; weight decay
#                            is always zero).
#   --warmup N               Linear-warmup steps to --lr (default 100).
#   --cosine N               Cosine period after warmup (default 2000), independent
#                            of --steps; after N steps it stays at --lr-floor.
#                            0 disables cosine and holds --lr after warmup.
#   --lr-floor X             Post-cosine learning-rate floor (default 1e-4;
#                            unused when --cosine 0).
#   --lam X                  Weight of mean emission-span CE in
#                            CE(answer)+X*mean(CE(emission)) (default 0.125).
#   --seed N                 Training RNG seed (default 0): controls initialization,
#                            task/k choice, and addressable online examples; the
#                            checkpoint also preserves Python/Torch/CUDA RNG state.
#   --log-every N            Print synchronized loss/time/peak-memory status every
#                            N steps (default 20). Must be positive.
#   --eval-every N           Run the paired max-k forced-choice sweep every N
#                            steps (default 500); 0 disables periodic eval.
#   --eval-n N               Fixed seed-0 examples per task in each periodic eval
#                            (default 256).
#   --prefetch N             Requested depth of the deterministic one-worker,
#                            pinned-host batch pipeline (default 2; <=1 acts as 1).
#   --save-every N           Atomically overwrite the resumable --out checkpoint
#                            every N steps (default 100); 0 disables periodic saves.
#                            Clean completion and KeyboardInterrupt save regardless.
#   --snapshot-every N       Also write immutable OUT.N trajectory checkpoints
#                            every N steps (default 0, disabled). score.py discovers
#                            these for trajectory readout.
#   --knobs K=V,...          Integer task-generator overrides recorded in the
#                            checkpoint, e.g. length=8 (default none). Every key
#                            must exist for every selected task. Pinned defaults:
#                            s5_chain length=8; parity length=32; reachability
#                            nodes=10,edges=18; threesum has no knobs.
#   --out PATH               Main checkpoint path (train.py default
#                            data/train/surface.pt). `probe` and queued `train`
#                            append their own --out, forcing /tmp/probe-*.pt and
#                            data/train/TAG.pt respectively, so this flag cannot
#                            redirect lab.sh-managed artifacts.
#   --resume [PATH]          Restore surface, optimizer, and RNG state from PATH;
#                            with no PATH, restore from --out. Resume is exact when
#                            the original config is re-supplied; saved CLI args are
#                            not automatically loaded. Any queue line containing
#                            `--resume` skips its four-step probe.
#
# SCORE FLAGS
# score.py takes its task(s), task knobs, wire pair, home arm, and train batch
# (capped by score n) from each checkpoint. It always reports untrained nulls
# in both arms plus the trained final surface in its home and cross arm.
#
#   TAG...                   One or more checkpoint stems under data/train/.
#   --n N                    Paired evaluation examples per task/knob set
#                            (default 512).
#   --seed N                 Evaluation-instance seed (default 0).
#   --k K1,K2,...            Forced-choice causal-prefix budgets and max free-run
#                            budget (default 1,2,4,8,16,32).
#   --no-trajectory          Skip OUT.N snapshot scoring; final, null, cross-arm,
#                            and requested knob-transfer cells still run.
#   --transfer K=V1,V2,...   Re-score the trained home arm and both untrained null
#                            arms under each listed value, layered over checkpoint
#                            knobs (default none). K must be valid for each task.
#   --model ID_OR_PATH       Base model used to score all TAGs (default
#                            google/gemma-3-1b-pt); score.py does not read the saved
#                            model id for this choice.
#
# QUEUE FILE GRAMMAR AND ARTIFACTS
# One run per line:
#
#   TAG <TRAIN FLAGS> [| <SCORE FLAGS>]
#
# Blank lines and text from the first `#` onward are ignored. The first
# whitespace-delimited word is TAG; the first `|` separates post-hoc score args.
# Lines are expanded as plain shell words by the worker: quoting inside FILE is
# not parsed, so values must not rely on embedded spaces or shell quoting.
#
# For each non-resume entry the worker first forces `--steps 4` and writes
# /tmp/probe-TAG.pt, logging only to logs/TAG.probe.log. A successful entry then
# forces --out data/train/TAG.pt, tees training to logs/TAG.log and queue.log,
# and tees post-hoc scoring to logs/TAG.score.log and queue.log. Progress is
# appended to logs/queue-STATUS (`DONE` is the durable completion marker).
# logs/queue.log is replaced when a new queue launches; per-TAG logs and STATUS
# are not cleared. PYTHONUNBUFFERED=1 keeps detached logs current.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs data/train
STATUS=logs/queue-STATUS
FILTER='==>|eval|compile/warmup|exit|DONE|FAILED|Error|Traceback|OOM|Killed|interrupted'

running() { pgrep -f "recirculated_dot_experiment.train run" >/dev/null; }

run_queue() {  # foreground: the detached worker
  local file=$1 raw line tag rest score_args rc
  echo "started $(date -Is) at $(git rev-parse --short HEAD) from $file" >> "$STATUS"
  while IFS= read -r raw || [[ -n $raw ]]; do
    line=${raw%%#*}
    [[ -z ${line// /} ]] && continue
    score_args=""
    if [[ $line == *"|"* ]]; then score_args=${line#*|}; line=${line%%|*}; fi
    read -r tag rest <<< "$line"
    if [[ $rest != *--resume* ]]; then
      # shellcheck disable=SC2086
      python -m recirculated_dot_experiment.train run $rest --steps 4 \
        --out "/tmp/probe-$tag.pt" > "logs/$tag.probe.log" 2>&1 </dev/null
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "$tag PROBE FAILED exit $rc $(date -Is) (logs/$tag.probe.log)" >> "$STATUS"
        continue
      fi
    fi
    # shellcheck disable=SC2086
    python -m recirculated_dot_experiment.train run $rest \
      --out "data/train/$tag.pt" 2>&1 </dev/null | tee "logs/$tag.log"
    rc=$?
    echo "$tag exit $rc $(date -Is)" >> "$STATUS"
    [[ $rc -ne 0 ]] && continue
    # shellcheck disable=SC2086
    python scripts/score.py "$tag" $score_args 2>&1 </dev/null | tee "logs/$tag.score.log" \
      || echo "$tag SCORE FAILED $(date -Is) (logs/$tag.score.log)" >> "$STATUS"
  done < "$file"
  echo "DONE $(date -Is)" >> "$STATUS"
}

launch_queue() {  # detached
  local file=$1
  if running; then echo "a training run is already active; refusing to stack" >&2; exit 1; fi
  if [[ -f logs/queue.pid ]] && kill -0 "$(cat logs/queue.pid)" 2>/dev/null; then
    echo "a queue worker is already running (pid $(cat logs/queue.pid))" >&2; exit 1
  fi
  nohup "$0" _run-queue "$file" > logs/queue.log 2>&1 &
  echo $! > logs/queue.pid
  echo "queue $file started, worker pid $!; follow with: scripts/lab.sh watch"
}

cmd=${1:-help}; shift || true
case $cmd in
  gate)
    python -m pytest tests -q && ruff check . \
      && python -m recirculated_dot_experiment.g0 identity \
      && python -m recirculated_dot_experiment.train gate ;;
  probe)
    # shellcheck disable=SC2086
    python -m recirculated_dot_experiment.train run "$@" --steps 4 --out /tmp/probe-lab.pt ;;
  train)
    [[ $# -ge 1 ]] || { echo "usage: lab.sh train TAG [train args]" >&2; exit 2; }
    tag=$1; shift
    file=logs/queue-$tag.txt
    echo "$tag $*" > "$file"
    launch_queue "$file" ;;
  queue)
    [[ $# -eq 1 && -f $1 ]] || { echo "usage: lab.sh queue FILE" >&2; exit 2; }
    launch_queue "$1" ;;
  _run-queue)
    run_queue "$1" ;;
  score)
    [[ $# -ge 1 ]] || { echo "usage: lab.sh score TAG... [score args]" >&2; exit 2; }
    python scripts/score.py "$@" 2>&1 | tee "logs/score-$(date +%Y%m%d-%H%M%S).log" ;;
  status)
    [[ -f $STATUS ]] && tail -n 8 "$STATUS" || echo "no queue marker yet"
    running && echo "-- training active --" || echo "-- no training active --"
    for f in $(ls -t logs/*.log 2>/dev/null | grep -vE 'score|probe|queue' | head -2); do
      echo "== $f"; grep -E "^step|eval" "$f" | tail -3 || true
    done
    exit 0 ;;
  watch)
    tail -n 0 -F logs/queue.log "$STATUS" 2>/dev/null | grep --line-buffered -E "$FILTER" ;;
  kill)
    if [[ -f logs/queue.pid ]]; then kill "$(cat logs/queue.pid)" 2>/dev/null && echo "queue worker stopped"; rm -f logs/queue.pid; fi
    pkill -f "recirculated_dot_experiment.train run" && echo "training python stopped" || echo "no training python running"
    echo "killed by hand $(date -Is)" >> "$STATUS" ;;
  *)
    sed -n '2,20p' "$0" ;;
esac
