#!/usr/bin/env bash
# Operator front door for the experiment — run on jobe from any cwd.
#
#   scripts/lab.sh gate                     tests + ruff + identity + gradient gates
#   scripts/lab.sh probe  [train args]      4-step smoke of a config (warm + audit)
#   scripts/lab.sh train  TAG [train args]  probe, run, score — detached, one run
#   scripts/lab.sh queue  FILE              same, one run per line of FILE, in order
#   scripts/lab.sh score  TAG... [args]     post-hoc scorer (scripts/score.py)
#   scripts/lab.sh status                   queue marker + last lines of live logs
#   scripts/lab.sh watch                    filtered live tail (evals, exits, errors)
#   scripts/lab.sh kill                     stop the queue and the running python
#
# Queue file: one run per line, `TAG <train.py run args> [| <score.py args>]`;
# blank lines and `#` comments ignored. Each run is probed with --steps 4
# first (skipped for --resume lines), then trained to data/train/TAG.pt
# with logs/TAG.log, then scored to logs/TAG.score.log; everything also
# streams through logs/queue.log (what `watch` follows). Progress marker:
# logs/queue-STATUS (ends with DONE). Remember: --cosine is a period, not
# the run length — pass --cosine N to anneal across an N-step run.
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
