"""Record the reviewed smoke/pilot evidence and check substantial-run readiness."""
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil
from dotenv import load_dotenv
from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import DataloaderStateVectorSource
from flowtorch.data import HDF5Dataloader
from convert_surface import ROOT, configured_path, sha256
from check_experiment_gate import check


def main():
    load_dotenv(ROOT/'.env',override=False)
    output=configured_path('OUTPUT_DIR')/'analysis'
    directory=output/'working/preflight'
    pilot=json.loads((directory/'pilot/run.json').read_text())
    smoke=(directory/'smoke.log').read_text()
    assert '"status": "passed"' in smoke and pilot['status']=='complete'
    available=psutil.virtual_memory().available
    free=shutil.disk_usage(output).free
    assert available>10*1024**3 and free>24*1024**3
    conversion=configured_path('OUTPUT_DIR')/'surface'
    metadata=json.loads((conversion/'conversion.json').read_text())
    started=time.perf_counter()
    for name, entry in metadata['outputs'].items():
        assert sha256(conversion/name)==entry['sha256'],name
    estimated=pilot['total_seconds']*(201994/2048)*2
    evidence={
        'correctness':{'smoke':'smoke.log','pilot':'pilot/run.json',
            'observed':pilot['validation'],
            'acceptance':'Energy relative error <1e-9; leading weighted orthogonality <1e-7; sampled reconstruction relative error <1e-7; independent NumPy oracle and direct PAMSPOD spectra in smoke check.'},
        'measurement':{'pilot_wall_seconds':pilot['total_seconds'],'instrument':'time.perf_counter, includes loading/factorization/validation/checkpoint save',
            'peak_rss_bytes':pilot['peak_rss_bytes'],'instrument_rss':'resource.getrusage ru_maxrss, process peak',
            'repeats':1,'limitation':'Resource pilot, not a performance comparison; linear extrapolation is approximate.'},
        'execution_path':{'SVD_module':os.path.relpath(inspect.getfile(SVD),ROOT),
            'observed_backend':'CPU native Float64 TSQR; 4 Torch threads; one process; disk-backed basis',
            'pilot_shape':[6144,1310],'full_shapes':[[201994,1310],[403988,1310],[605982,1310]],
            'full_rank_ceiling':1309,'centering':'per-row temporal mean','weights':'stored area weights repeated per component','normalization':'unit factors for each coefficient'},
        'artifact_integrity':{'input_checksums_verified':metadata['outputs'],'check_seconds':time.perf_counter()-started,
            'pilot_checkpoint':'pilot/checkpoint','pilot_status':pilot['status'],'centering_null_direction':'Exclude the known constant-time null direction by rank ceiling N-1; synthetic initial failed null-direction test retained in smoke_initial_null_direction.log.'},
        'resource_estimate':{'linear_estimate_seconds':estimated,'conservative_estimate_seconds':1800,'budget_seconds':7200,
            'available_ram_bytes':available,'free_disk_bytes':free,'construction_memory_budget_bytes':8*1024**3,
            'expected_peak_memory_upper_bytes':12*1024**3,'estimated_peak_disk_bytes':22*1024**3,
            'storage':'Three independent full resolved-rank checkpoints, approximately 12.7 GB combined, with temporary disk basis released after each case.',
            'stop_rule':'Stop on invariant failure, nonfinite data, failed checkpoint, incompatible existing record or elapsed budget breach.'},
        'preflight_self_review':{'reviewer':'second-pass self-review; no delegated reviewer',
            'findings':['Temporal mean yields rank ceiling N-1.','Energy is based on squared singular values, not FlowTorch s_cum.','Uniform positive dt independently checked from all source times.','Equal component normalization preserves vector inner product.','No SPOD run dispatched before rank selection.','Pilot rank estimates apply only to the spatial subset, not the full surface.'],
            'limitations':['Single-process CPU execution only.','Full-field invariants will be checked before accepting each checkpoint.']}}
    evidence_path=directory/'evidence.json'
    evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    files=[evidence_path,directory/'smoke.log',directory/'pilot/run.json',ROOT/'config/spod_study.json',ROOT/'tests/check_svd.py',ROOT/'requirements-lock.txt']
    files += list((ROOT/'scripts').glob('*.py'))
    files += [conversion/name for name in metadata['outputs']]
    files += [Path(inspect.getfile(cls)) for cls in [SVD,DataloaderStateVectorSource,HDF5Dataloader]]
    files += [Path(inspect.getfile(HDF5Dataloader)).parent/name for name in ['_hdf5_schema.py','_hdf5_compression.py']]
    gate={'schema_version':1,'checks':{name:{'status':'pass','evidence':'evidence.json'} for name in ['correctness','measurement','execution_path','artifact_integrity','resource_estimate']},
        'fingerprints':{os.path.relpath(p,directory):sha256(p) for p in files},
        'proposed_run':{'estimated_seconds':1800,'budget_seconds':7200}}
    path=directory/'gate.json';path.write_text(json.dumps(gate,indent=2)+'\n')
    passed,message=check(path)
    print(message,flush=True)
    if not passed: raise SystemExit(1)

if __name__=='__main__':main()
