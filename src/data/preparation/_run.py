"""Run an existing CLI from the repo root using the IDE's selected Python."""
import subprocess
import sys
from src.data.preparation.settings import ROOT


def run(module, arguments):
    # --help can inspect any numbered entry without starting its workload.
    args = ['--help'] if '--help' in sys.argv[1:] else [str(arg) for arg in arguments] + sys.argv[1:]
    result = subprocess.run([sys.executable, '-m', module, *args], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
