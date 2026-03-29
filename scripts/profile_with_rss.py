#!/usr/bin/env python3
import resource
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: profile_with_rss.py <command> [args...]", file=sys.stderr)
        return 2

    start = time.perf_counter()
    completed = subprocess.run(sys.argv[1:], check=False)
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"max_rss_kb={usage.ru_maxrss}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
