"""Reuse the numerical Cp SVD to extend its adaptive rank/tolerance study."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.analysis import SVD
from convert_surface import ROOT,configured_path,sha256
from surface_spod import RankView,calculate,compatible_result,atomic_json
from check_experiment_gate import check

CONFIG=ROOT/'config/numerical_cp_spod_sensitivity.json'


def result_path(output,rank,tolerance):
    if rank==400:return output/'analysis/working/spod/cp'/f'adaptive_{tolerance:g}'
    return output/'analysis/working/spod_rank_sensitivity/cp'/f'rank_{rank}'/f'adaptive_{tolerance:g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rank',type=int);parser.add_argument('--tolerance',type=float)
    parser.add_argument('--gate',type=Path,help='Optional reviewed experiment gate')
    args=parser.parse_args();load_dotenv(ROOT/'.env');config=json.loads(CONFIG.read_text())
    torch.set_default_dtype(torch.float64);torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    if args.gate:
        passed,message=check(args.gate)
        if not passed:raise ValueError(message)
    output=configured_path('OUTPUT_DIR');saved=output/'analysis/working/svd/cp'
    source=json.loads((saved/'run.json').read_text())
    assert source['status']=='complete' and source['fields']==['cp'] and source['snapshots']==1310
    assert source['subtract_temporal_mean'] and not source['subtract_spatial_mean'] and source['dtype']=='float64'
    ranks=[args.rank] if args.rank is not None else config['ranks']
    tolerances=[args.tolerance] if args.tolerance is not None else config['adaptive_tolerances']
    if not set(ranks)<=set(config['ranks']) or not set(tolerances)<=set(config['adaptive_tolerances']):raise ValueError('Variant not configured')
    pending=[]
    for rank in ranks:
        for tolerance in tolerances:
            setting=(f'adaptive_{tolerance:g}',True,config['adaptive_max_tapers'],tolerance)
            path=result_path(output,rank,tolerance)
            if compatible_result(path,saved,rank,setting):print(f'Reusing numerical cp rank {rank}, tolerance {tolerance:g}',flush=True)
            elif path.exists():raise FileExistsError(f'Incompatible or incomplete result: {path}')
            else:pending.append((rank,setting,path))
    if not pending:return
    destination=output/'analysis/working/spod_rank_sensitivity/cp';destination.mkdir(parents=True,exist_ok=True)
    original=SVD.load(saved/'checkpoint',basis_directory=destination)
    try:
        assert original.V.shape==(1310,1309) and original.V.dtype==original.s.dtype==torch.float64
        for rank,setting,path in pending:
            view=RankView(original,rank);assert view.V.shape==(1310,rank) and view.s.shape==(rank,)
            start=time.perf_counter();calculate(view,source['dt'],setting,path)
            if time.perf_counter()-start>config['maximum_variant_seconds']:raise TimeoutError('Per-variant runtime budget exceeded')
            record=json.loads((path/'run.json').read_text())
            with np.load(path/'spectra.npz') as data:
                assert data['eigvals'].dtype==data['frequency'].dtype==np.float64 and data['reduced_modes'].dtype==np.complex128
                assert data['reduced_modes'].shape==(656,rank,3)
                assert data['n_tapers'].max()<=200 and data['n_tapers'].min()>0
            record.update(case='cp',snapshots=1310,svd_metadata_sha256=sha256(saved/'checkpoint/metadata.pt'),
                svd_run_sha256=sha256(saved/'run.json'),study_config_sha256=sha256(CONFIG),
                entrypoint_sha256=sha256(Path(__file__)),subtract_temporal_mean=True,subtract_spatial_mean=False,
                weights='existing FlowTorch surface-area weights inherited from SVD')
            atomic_json(path/'run.json',record)
    finally:original.close()

if __name__=='__main__':main()
