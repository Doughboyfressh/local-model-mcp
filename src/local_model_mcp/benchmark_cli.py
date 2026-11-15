"""Command-line entry point for the same isolated benchmark jobs exposed over MCP."""

import argparse
import json
import sys
import time

from .jobs import TERMINAL, Jobs
from .runtime import default_root
from .schemas import BenchmarkRequest
from .workspace import Workspace


def main():
    parser = argparse.ArgumentParser(
        description="Compare two local Qwen checkpoints without training"
    )
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--dataset")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-eval-tokens", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3, dest="measured_runs")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--min-token-agreement", type=float, default=1)
    parser.add_argument("--max-loss-increase", type=float, default=0.01)
    parser.add_argument("--include-generated-text", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = vars(parser.parse_args())
    root, no_wait = args.pop("root"), args.pop("no_wait")
    if args["prompts"] is None:
        del args["prompts"]
    try:
        request = BenchmarkRequest.model_validate(args)
        jobs = Jobs(Workspace(root))
        job = jobs.start("benchmark", request)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if no_wait:
        print(json.dumps(job))
        return
    previous = None
    try:
        while True:
            status = jobs.status(job["job_id"], log_lines=0)
            if status["status"] in TERMINAL:
                if status["status"] == "completed":
                    print(json.dumps({"job_id": job["job_id"], **status["result"]}, indent=2))
                    if not status["result"]["passed"]:
                        raise SystemExit(1)
                    return
                print(status.get("error", status["status"]), file=sys.stderr)
                raise SystemExit(2)
            progress = status.get("benchmark_progress", {}).get(
                "phase", status.get("phase", status["status"])
            )
            if progress != previous:
                print(f"Benchmark: {progress}", file=sys.stderr)
                previous = progress
            time.sleep(0.2)
    except KeyboardInterrupt:
        jobs.cancel(job["job_id"])
        print("Benchmark cancellation requested", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
