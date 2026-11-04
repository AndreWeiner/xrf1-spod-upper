"""Check storage filters/precision and compare first/last frames with native PLT input."""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # Registers the Zstd filter before any HDF5 dataset is read.
import numpy as np
import torch
from flowtorch.data import TecplotDataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    record = json.loads((args.directory / 'conversion.json').read_text())
    if record['status'] != 'complete':
        raise ValueError('Conversion is incomplete')
    source = Path(record['source_directory'])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    sources = record['sources']
    files = {s['time']: s['name'] for s in sources}
    checks = 0
    with h5py.File(args.directory / 'geometry.h5') as geometry, \
         h5py.File(args.directory / 'fields.h5') as fields, \
         TecplotDataloader(str(source), files, dtype=torch.float64,
                           geometry_dtype=torch.float64, backend='native') as loader:
        for handle in (geometry, fields):
            if not handle.attrs.get('complete', False):
                raise ValueError('HDF5 completion marker is missing')
            def check_dataset(name, item):
                if isinstance(item, h5py.Dataset):
                    if item.dtype.kind == 'f' and item.dtype != np.dtype('float64'):
                        raise ValueError(f'Non-Float64 dataset: {name}')
                    filters = item.id.get_create_plist()
                    ids = [filters.get_filter(i)[0] for i in range(filters.get_nfilters())]
                    if 32015 not in ids or not item.shuffle:
                        raise ValueError(f'Missing Zstd/shuffle: {name}')
            handle.visititems(check_dataset)
        frames = list(fields['frames'].values())
        if len(frames) != len(sources):
            raise ValueError('Snapshot count mismatch')
        if [frame.attrs['label'] for frame in frames] != list(files):
            raise ValueError('Time order mismatch')
        for zone_key, zone in geometry['zones'].items():
            loader.zone = zone.attrs['name']
            reference = loader.geometry()
            for key in ('points', 'topology', 'locations', 'weights'):
                np.testing.assert_array_equal(zone[key][:], getattr(reference, key))
                checks += 1
            for index in sorted({0, len(frames) - 1}):
                frame = frames[index]
                for dataset in frame[zone_key].values():
                    original = loader.load_snapshot(dataset.attrs['name'], frame.attrs['label']).numpy()
                    np.testing.assert_array_equal(dataset[:], original)
                    checks += 1
    report = {'status': 'passed', 'snapshots': len(sources),
              'exact_array_comparisons': checks,
              'scope': 'all dataset filters/dtypes; all zone geometry; first/last frame fields'}
    (args.directory / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
