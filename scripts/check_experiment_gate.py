"""Check a saved experiment gate, evidence fingerprints, and time budget."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REQUIRED = ('correctness', 'measurement', 'execution_path', 'artifact_integrity', 'resource_estimate')


def check(path):
    path = Path(path)
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
            raise ValueError('schema_version must be 1')
        checks, fingerprints = record['checks'], record['fingerprints']
        if not isinstance(checks, dict) or not isinstance(fingerprints, dict) or not fingerprints:
            raise ValueError('checks and nonempty fingerprints must be mappings')
        for name in REQUIRED:
            item = checks.get(name)
            if not isinstance(item, dict) or item.get('status') != 'pass':
                raise ValueError(f'{name} has not passed')
            evidence = item.get('evidence')
            if not isinstance(evidence, str) or evidence not in fingerprints:
                raise ValueError(f'{name} evidence must be fingerprinted')
        for filename, expected in fingerprints.items():
            if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
                raise ValueError('fingerprint paths must be nonempty relative paths')
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError(f'invalid SHA-256 for {filename}')
            digest = hashlib.sha256()
            with (path.parent / filename).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(f'changed file: {filename}')
        run = record['proposed_run']
        estimate, budget = run['estimated_seconds'], run['budget_seconds']
        for value in (estimate, budget):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('time values must be finite numbers')
        if estimate <= 0 or budget <= 0 or estimate > budget:
            raise ValueError('invalid or over-budget run estimate')
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        return False, str(error)
    return True, 'Gate record valid: required passes, fingerprints and time budget checked; scientific evidence still requires review.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate')
    args = parser.parse_args()
    ok, message = check(args.gate)
    print(('PASS: ' if ok else 'BLOCK: ') + message)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
