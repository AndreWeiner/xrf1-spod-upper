"""Reproducible, lossless PLT conversion through FlowTorch's public API."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
import torch
from flowtorch.data import PLT2HDF5

ROOT = Path(__file__).resolve().parents[1]


def configured_path(key):
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Set {key} in {ROOT / '.env'} (see .env.example)")
    path = Path(value).expanduser()
    return (ROOT / path).absolute() if not path.is_absolute() else path.absolute()


def project_path(path):
    """Use portable paths for locations within the repository."""
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zones', nargs='+', default=['wing-upper'],
                        help='Exact zone names; default: wing-upper')
    parser.add_argument('--fields', nargs='+', default=['cp', 'cfx', 'cfy', 'cfz'],
                        help='Exact fields; default: cp cfx cfy cfz')
    parser.add_argument('--limit', type=int, help='First N times for a separate smoke test')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    load_dotenv(ROOT / '.env', override=False)
    source = configured_path('PLT_DATA_DIR')
    if not source.is_dir():
        raise FileNotFoundError(f'Dataset directory is unavailable: {source}')
    converter = PLT2HDF5(source, pattern='*.plt', backend='native')
    times = list(converter.files)[:args.limit] if args.limit else list(converter.files)
    label = f'surface_smoke_{args.limit}' if args.limit else 'surface'
    output = configured_path('OUTPUT_DIR') / label
    # A failed or finished run is never silently overwritten or mistaken for a new run.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'command': [project_path(sys.argv[0]), *sys.argv[1:]],
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'settings': {'backend': 'native', 'fields': args.fields, 'zones': args.zones,
                     'field_dtype': 'float64', 'geometry_dtype': 'float64',
                     'compression': 'zstd', 'compression_level': 3, 'shuffle': True,
                     'spatial_chunk': 65536, 'association': 'point'},
        'source_directory': project_path(source), 'output_directory': project_path(output),
        'script_sha256': sha256(Path(__file__)), 'sources': [],
    }
    record = output / 'conversion.json'
    def save():
        temporary = record.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(record)
    save()
    try:
        for index, time in enumerate(times, start=1):
            path = Path(converter.files[time])
            stat = path.stat()
            manifest['sources'].append({'time': time, 'name': path.name,
                'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)})
            if index % 100 == 0 or index == len(times):
                print(f'Hashed {index}/{len(times)} source snapshots', flush=True)
        save()
        print(f'Converting {len(times)} snapshots to {project_path(output)}', flush=True)
        descriptor = converter.convert(output, fields=args.fields, zones=args.zones,
            times=times, field_dtype=torch.float64, compression='zstd',
            compression_level=3, shuffle=True, geometry_compression='zstd',
            geometry_compression_level=3, geometry_shuffle=True, spatial_chunk=65536)
        # Detect source modifications during the run without rereading the full series.
        for item in manifest['sources']:
            stat = (source / item['name']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
                raise RuntimeError(f"Source changed during conversion: {item['name']}")
        manifest['outputs'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                               for p in output.iterdir() if p.suffix in ('.h5', '.xdmf')}
        manifest['status'] = 'complete'
        manifest['descriptor'] = project_path(descriptor)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = repr(error)
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(project_path(descriptor), flush=True)


if __name__ == '__main__':
    main()
