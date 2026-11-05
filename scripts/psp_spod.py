"""Reuse a saved PSP SVD for the configured rank/adaptive-tolerance comparison."""
import argparse
import inspect
import json
from pathlib import Path
import time
import subprocess
import sys
import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.analysis import SVD
from convert_surface import ROOT,configured_path,sha256
from surface_spod import RankView,calculate,compatible_result,atomic_json
from check_experiment_gate import check

CONFIG=ROOT/'config/psp_spod_study.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--polar',help='PSP polar ID; defaults to configured primary record')
    parser.add_argument('--rank',type=int);parser.add_argument('--tolerance',type=float)
    parser.add_argument('--gate',type=Path,help='Optional reviewed experiment gate')
    args=parser.parse_args();load_dotenv(ROOT/'.env');config=json.loads(CONFIG.read_text())
    torch.set_default_dtype(torch.float64);torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    if args.gate:
        passed,message=check(args.gate)
        if not passed:raise ValueError(message)
    base=configured_path('OUTPUT_DIR')/'psp';polar=args.polar or config['polar']
    if polar not in config['polars']:raise ValueError('Polar is not included in the study configuration')
    saved=base/'working/svd'/f'polar_{polar}';source=json.loads((saved/'run.json').read_text())
    if not (saved/'verification.json').is_file():
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_psp_svd.py'),'--polar',polar],check=True,cwd=ROOT)
    verification=json.loads((saved/'verification.json').read_text())
    assert source['status']=='complete' and verification['status']=='pass' and source['polar']==polar
    assert source['angle_deg']==config['polars'][polar] and source['snapshots']==3639 and source['dtype']=='float64'
    assert source['subtract_snapshot_spatial_mean'] and source['subtract_pointwise_temporal_mean']
    assert source['config']['domain']=={'automatic_mask':'p5_t10','patch_size':5,'threshold':10.,'temporal_reduction':'max','crop_pixels_each_edge':5,'native_mask':None}
    assert sha256(saved/'checkpoint/metadata.pt')==source['checkpoint_metadata_sha256']
    ranks=[args.rank] if args.rank is not None else config['ranks']
    tolerances=[args.tolerance] if args.tolerance is not None else config['adaptive_tolerances']
    if not set(ranks)<=set(config['ranks']) or not set(tolerances)<=set(config['adaptive_tolerances']):raise ValueError('Requested variant not in study configuration')
    destination=base/'working/spod'/f'polar_{polar}';destination.mkdir(parents=True,exist_ok=True)
    settings=[(rank,(f'adaptive_{tol:g}',True,config['adaptive_max_tapers'],tol)) for rank in ranks for tol in tolerances]
    pending=[]
    for rank,setting in settings:
        path=destination/f'rank_{rank}'/setting[0]
        if compatible_result(path,saved,rank,setting):
            print(f'Reusing {polar}, rank {rank}, {setting[0]}',flush=True)
        elif path.exists():raise FileExistsError(f'Incompatible or incomplete variant: {path}')
        else:pending.append((rank,setting,path))
    if not pending:return
    # The previous checkpoint validation supplied complete per-file fingerprints.
    for relative,digest in verification['manifest_sha256'].items():
        if relative.startswith('checkpoint/') and sha256(saved/relative)!=digest:raise ValueError(f'Changed SVD factor: {relative}')
    original=SVD.load(saved/'checkpoint',basis_directory=destination)
    try:
        assert original.V.dtype==original.s.dtype==torch.float64 and len(original.V)==3639
        for rank,setting,path in pending:
            view=RankView(original,rank)
            assert view.s.shape==(rank,) and view.V.shape==(3639,rank)
            start=time.perf_counter()
            calculate(view,source['dt_seconds'],setting,path)
            record=json.loads((path/'run.json').read_text())
            if time.perf_counter()-start>config['maximum_variant_seconds']:raise TimeoutError('Per-variant runtime budget exceeded')
            with np.load(path/'spectra.npz') as arrays:
                assert arrays['eigvals'].dtype==np.float64 and arrays['reduced_modes'].dtype==np.complex128
                assert len(arrays['frequency'])==1820 and np.all(np.diff(arrays['frequency'])>0)
                assert arrays['n_tapers'].max()<=200 and arrays['n_tapers'].min()>=1
            record.update(polar=polar,angle_deg=source['angle_deg'],snapshots=source['snapshots'],
                svd_metadata_sha256=sha256(saved/'checkpoint/metadata.pt'),
                svd_run_sha256=sha256(saved/'run.json'),preprocessing_sha256=sha256(saved/'preprocessing.npz'),
                study_config_sha256=sha256(CONFIG),psp_entrypoint_sha256=sha256(Path(__file__)),
                domain=source['config']['domain'],subtract_snapshot_spatial_mean=True,
                subtract_pointwise_temporal_mean=True,weights='existing FlowTorch geometric nodal areas, inherited from SVD')
            atomic_json(path/'run.json',record)
    finally:original.close()

if __name__=='__main__':main()
