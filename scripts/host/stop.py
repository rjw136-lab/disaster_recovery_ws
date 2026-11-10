#!/usr/bin/env python3
"""
Convenience script to stop all Disaster Recovery robot and host processes.
"""
import sys
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = ["/home/lab/start.py", "--stop"] + args
    sys.exit(subprocess.call(cmd))
