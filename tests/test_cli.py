import json
import subprocess
import sys
from pathlib import Path


def test_unknown_sample_returns_structured_failure():
    result = subprocess.run([sys.executable, 'cli.py', 'describe', '--sample-id', '../../private'],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 2
    parsed = json.loads(result.stdout)
    assert parsed['ok'] is False
    assert 'sample' in parsed['error'].lower()


def test_inventory_reads_frozen_samples_and_exclusions():
    result = subprocess.run([sys.executable, 'cli.py', 'inventory'],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed['ok'] is True
    assert parsed['samples'] == 77
    assert parsed['evaluation_samples'] == 73
    assert parsed['evaluation_identities'] == 46
