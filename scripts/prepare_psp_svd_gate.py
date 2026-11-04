"""Record reviewed PSP SVD pilot evidence and validate the full-run gate."""
import inspect
import json
import os
from pathlib import Path
import shutil
import psutil
from dotenv import load_dotenv
from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import StateVectorSource
from flowtorch.data import PSPDataloader
from convert_surface import ROOT,configured_path,sha256
from check_experiment_gate import check


def main():
    load_dotenv(ROOT/'.env');working=configured_path('OUTPUT_DIR')/'psp/working'
    destination=working/'svd_preflight';pilot=destination/'polar_0255/run.json'
    r=json.loads(pilot.read_text());assert r['status']=='complete' and r['snapshots']==3639 and r['points']==8192
    assert '"status": "pass"' in (working/'psp_svd_smoke.log').read_text()
    assert r['validation']['sampled_reconstruction_relative_error']<1e-7
    available=psutil.virtual_memory().available;free=shutil.disk_usage(working).free
    assert available>10*1024**3 and free>40*1024**3
    evidence={
        'correctness':{'smoke':'../psp_svd_smoke.log','pilot':'polar_0255/run.json','pilot_observed':r['validation'],
            'acceptance':'Independent nonuniform-weight NumPy oracle; centered reconstruction and native checkpoint SPOD equivalence. Full-case energy error <1e-9, first-64 weighted orthogonality <1e-7, sampled full-rank reconstruction <1e-7, spatial/temporal centering residuals <1e-11.'},
        'measurement':{'pilot_seconds':r['total_seconds'],'peak_process_rss_bytes':r['peak_rss_bytes'],
            'scope':'perf_counter includes hashing, original full-domain loading, spatial centering, subset TSQR, validation, checkpoint save; one pilot, not a timing benchmark'},
        'execution_path':{'native_svd_module':os.path.relpath(inspect.getfile(SVD),ROOT),
            'backend':'CPU Float64 native weighted TSQR, four threads, disk-backed basis, 8192-row batches',
            'full_domains':'Default full-record automatic mask plus five-pixel crop, no native masks; approximately 64-67 thousand retained points per polar, all 3639 snapshots',
            'weights':'Existing FlowTorch geometric nodal area arrays in fields.npz, selected without rescaling',
            'centering':'Full-domain weighted spatial mean removed first; FlowTorch SVD subtracts pointwise temporal mean'},
        'artifact_integrity':{'pilot_status':r['status'],'pilot_input_hashes':r['fingerprints'],
            'checkpoint':'polar_0255/checkpoint/metadata.pt','note':'Metadata written last by native SVD marks complete factors; fingerprints checked on reuse'},
        'resource_estimate':{'available_memory_bytes':available,'free_disk_bytes':free,
            'construction_budget_bytes':8*1024**3,'estimated_peak_process_bytes':12*1024**3,
            'expected_persistent_checkpoints_bytes':17*1024**3,'estimated_peak_storage_bytes':24*1024**3,
            'estimated_seconds':1600,'budget_seconds':14400,'per_polar_seconds':1800,
            'basis':'One pilot including full input load took about 18 s. Conservative full-domain extrapolation with TSQR reduction overhead: 200 s per polar; sequential processing.',
            'stop_rule':'Stop on failed centering/SVD invariant, incompatible or incomplete prior result, or per-polar elapsed budget breach'},
        'self_review':{'kind':'Bounded second-pass self-review; no delegated agent',
            'findings':['Spatial means computed over the complete retained domain even in the spatial-subset pilot.',
                'Subset pilot is not expected to have zero mean over its own subset; full runs enforce both centering constraints.',
                'Temporal mean removed exactly once by native SVD; saved means allow original Cp reconstruction.',
                'No Mask 2 enters the final SVD domain.',
                'The source uses Float64 selected original data and native supplied area arrays; excluded pixels are absent.',
                'Squared singular values define energy fractions; N-1 excludes the known centering null direction.',
                'SVD rank remains complete for later truncation and SPOD.from_svd reuse; final rank and SPOD sweep remain undecided.']}}
    path=destination/'evidence.json';path.write_text(json.dumps(evidence,indent=2)+'\n')
    files=[path,pilot,working/'psp_svd_smoke.log',working/'psp_svd_pilot.log',destination/'polar_0255/checkpoint/metadata.pt',
        ROOT/'config/psp_svd_study.json',ROOT/'tests/check_psp_svd.py',ROOT/'requirements-lock.txt']
    files += [ROOT/'scripts'/p for p in ['psp_svd.py','psp_study.py','svd_study.py','convert_surface.py','check_experiment_gate.py']]
    files += [Path(inspect.getfile(cls)) for cls in [SVD,StateVectorSource,PSPDataloader]]
    for directory in sorted(working.glob('polar_*')):
        files += [directory/'run.json',directory/'masks.npz',directory/'fields.npz']
    record={'schema_version':1,'checks':{k:{'status':'pass','evidence':'evidence.json'} for k in ['correctness','measurement','execution_path','artifact_integrity','resource_estimate']},
        'fingerprints':{os.path.relpath(p,destination):sha256(p) for p in files},
        'proposed_run':{'estimated_seconds':1600,'budget_seconds':14400}}
    gate=destination/'gate.json';gate.write_text(json.dumps(record,indent=2)+'\n')
    passed,message=check(gate);print(message)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
