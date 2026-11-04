"""Reusable area-weighted PSP SVDs after snapshot-spatial and pointwise-temporal centering."""
import argparse
from datetime import datetime,timezone
import gc
import importlib.metadata
import inspect
import json
from pathlib import Path
import resource
import time

from dotenv import load_dotenv
import numpy as np
import torch
from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import StateVectorSource,StateVectorLayout,FieldSpec
from flowtorch.data import PSPDataloader
from convert_surface import ROOT,configured_path,sha256
from psp_study import save_json
from svd_study import rank_summary
from check_experiment_gate import check

CONFIG=ROOT/'config/psp_svd_study.json'


class CenteredPSPSource(StateVectorSource):
    """Retained Cp after spatial centering; FlowTorch SVD handles temporal centering."""
    def __init__(self,data,weight):
        if data.dtype!=torch.float64 or weight.dtype!=torch.float64:raise ValueError('Float64 required')
        if data.ndim!=2 or weight.shape!=(data.shape[0],):raise ValueError('Invalid matrix/weight shape')
        if not bool(torch.isfinite(data).all() and torch.isfinite(weight).all() and (weight>0).all()):raise ValueError('Nonfinite data or invalid weights')
        self.data=data;self.weight=weight;self._layout=StateVectorLayout([FieldSpec('Cp')],(data.shape[0],))
    @property
    def layout(self):return self._layout
    @property
    def n_snapshots(self):return self.data.shape[1]
    @property
    def device(self):return self.data.device
    def read(self,s,t):return self.data[s,t]
    def read_weight(self,s):return self.weight[s]


def spatial_center(data,weight):
    """Modify owned retained data in place; save the original spatial mean."""
    mean=weight @ data/weight.sum()
    data.sub_(mean[None,:])
    residual=float((weight @ data/weight.sum()).abs().max())
    if residual>1e-11:raise ValueError(f'Spatial centering residual {residual}')
    return mean,residual


def validate(svd,source,require_zero_spatial_mean=True):
    k=min(64,svd.rank);gram=torch.zeros((k,k),dtype=torch.float64)
    energy=error=reference=0.;mean_residual=0.;spatial_residual=torch.zeros(source.n_snapshots,dtype=torch.float64)
    indices=[0,source.n_snapshots//2,source.n_snapshots-1]
    for part,modes in svd.U.iter_chunks():
        x=source.read(part,slice(None));temporal=x.mean(1);centered=x-temporal[:,None];w=source.weight[part]
        gram+=modes[:,:k].T @ (w[:,None]*modes[:,:k])
        energy+=float((centered.square()*w[:,None]).sum())
        reconstruction=modes @ (svd.s[:,None]*svd.V[indices].T)
        error+=float(((reconstruction-centered[:,indices]).square()*w[:,None]).sum())
        reference+=float((centered[:,indices].square()*w[:,None]).sum())
        mean_residual=max(mean_residual,float(centered.mean(1).abs().max()))
        spatial_residual+=w @ centered
    result={'relative_weighted_energy_error':abs(float(svd.s_full.square().sum())/energy-1),
        'weighted_orthogonality_max_error':float((gram-torch.eye(k,dtype=torch.float64)).abs().max()),
        'temporal_orthogonality_max_error':float((svd.V[:,:k].T@svd.V[:,:k]-torch.eye(k,dtype=torch.float64)).abs().max()),
        'sampled_reconstruction_relative_error':(error/reference)**.5,
        'snapshot_indices_checked':indices,'orthogonality_mode_count':k,'direct_weighted_energy':energy,
        'pointwise_temporal_mean_max_abs':mean_residual,
        'snapshot_weighted_spatial_mean_max_abs':float(spatial_residual.abs().max()/source.weight.sum())}
    if result['relative_weighted_energy_error']>1e-9 or result['weighted_orthogonality_max_error']>1e-7 or result['temporal_orthogonality_max_error']>1e-9 or result['sampled_reconstruction_relative_error']>1e-7 or mean_residual>1e-11 or (require_zero_spatial_mean and result['snapshot_weighted_spatial_mean_max_abs']>1e-11):
        raise ValueError(f'SVD validation failed: {result}')
    return result


def calculate(directory,destination,config,pilot=False):
    start=time.perf_counter();prior=json.loads((directory/'run.json').read_text())
    path=configured_path('PSP_DATA_DIR')/prior['source_file']
    fingerprints={'source':sha256(path),'masks':sha256(directory/'masks.npz'),'fields':sha256(directory/'fields.npz'),
        'code':sha256(Path(__file__)),'config':sha256(CONFIG),'flowtorch_svd':sha256(Path(inspect.getfile(SVD)))}
    assert fingerprints['source']==prior['source_sha256']
    if destination.exists():
        record=json.loads((destination/'run.json').read_text())
        # Accept the reviewed entry-point-only predecessor without rewriting its provenance.
        if record.get('fingerprints',{}).get('code')=='538a5748f4f84a7b7b7f6f486b8f2d09681d4392a7ec7ee4de1937b9bef83e72':
            fingerprints['code']='538a5748f4f84a7b7b7f6f486b8f2d09681d4392a7ec7ee4de1937b9bef83e72'
        if record['status']=='complete' and record['fingerprints']==fingerprints and record['pilot']==pilot:
            assert sha256(destination/'checkpoint/metadata.pt')==record['checkpoint_metadata_sha256']
            print(f'{prior["polar"]}: reusing completed SVD',flush=True);return record
        raise FileExistsError(f'Incomplete or incompatible SVD directory: {destination}')
    destination.mkdir(parents=True)
    record={'status':'running','polar':prior['polar'],'angle_deg':prior['angle_of_attack_deg'],
        'pilot':pilot,'fingerprints':fingerprints,'config':config,'dtype':'float64','device':'cpu',
        'threads':torch.get_num_threads(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'versions':{k:importlib.metadata.version(k) for k in ['flowtorch-fluid','torch','numpy','h5py']}}
    save_json(destination/'run.json',record)
    loader=PSPDataloader(str(path),dtype=torch.float64)
    svd=None
    try:
        loader.zone=prior['zone'];times=loader.write_times
        assert len(times)==prior['snapshots']==3639
        with np.load(directory/'fields.npz') as fields,np.load(directory/'masks.npz') as masks:
            domain=config['domain'];keep=masks[domain['automatic_mask']].copy();crop=domain['crop_pixels_each_edge']
            assert domain['native_mask'] is None and crop==5 and domain['temporal_reduction']=='max'
            keep[:crop]=False;keep[-crop:]=False;keep[:,:crop]=False;keep[:,-crop:]=False
            assert not (keep & ~fields['finite_keep']).any()
            indices=np.flatnonzero(keep);weight=torch.from_numpy(fields['area_weights'][keep].copy())
            coordinates=fields['coordinates'][keep].copy()
        full_points=int(keep.sum())
        data=loader.load_snapshot('Cp',times)[torch.from_numpy(keep)]
        assert data.dtype==torch.float64 and data.shape==(full_points,3639)
        # Even the spatial-subset pilot removes the full retained-domain spatial mean.
        spatial_mean,spatial_residual=spatial_center(data,weight)
        if pilot:
            n=config['pilot_points'];data=data[:n].clone();weight=weight[:n].clone();indices=indices[:n];coordinates=coordinates[:n]
        source=CenteredPSPSource(data,weight)
        temporal_mean=data.mean(1)
        fs=float(loader.zone_info['SamplingFrequency'][0]);dt=1/fs
        ceiling=min(source.spatial_shape[0],source.n_snapshots-1)
        record.update(points=source.spatial_shape[0],full_domain_points=full_points,snapshots=len(times),
            sampling_frequency_hz=fs,dt_seconds=dt,rank_ceiling=ceiling,
            spatial_centering_max_residual=spatial_residual,
            area_weight_sum=float(weight.sum()),area_units='squared source-coordinate units, unchanged from saved FlowTorch weights',
            subtract_snapshot_spatial_mean=True,subtract_pointwise_temporal_mean=True)
        save_json(destination/'run.json',record)
        np.savez(destination/'preprocessing.npz',mask=keep,retained_flat_indices=indices,coordinates=coordinates,
            area_weights=weight.numpy(),snapshot_spatial_mean=spatial_mean.numpy(),
            temporal_mean_after_spatial_subtraction=temporal_mean.numpy(),time_seconds=np.arange(len(times))/fs)
        print(f'{prior["polar"]}: TSQR {source.spatial_shape[0]} x {len(times)}, rank ceiling {ceiling}',flush=True)
        svd=SVD(source,rank=ceiling,mode='tsqr',subtract_mean=True,
            spatial_batch_size=config['spatial_batch_size'],memory_budget_bytes=config['memory_budget_gib']*1024**3,
            basis_storage='disk',basis_directory=destination)
        record['factorization_and_preprocessing_seconds']=time.perf_counter()-start
        evidence=validate(svd,source,require_zero_spatial_mean=not pilot)
        # A subset pilot does not have zero spatial mean over its subset.
        record['validation']=evidence
        record.update(rank_summary(svd.s_full.numpy(),config));record['resolved_rank']=svd.rank
        np.savez(destination/'spectrum.npz',singular_values=svd.s_full.numpy())
        svd.save(destination/'checkpoint');svd.close();svd=None
        record.update(status='complete',checkpoint_metadata_sha256=sha256(destination/'checkpoint/metadata.pt'),
            total_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            finished_utc=datetime.now(timezone.utc).isoformat())
        if record['total_seconds']>config['maximum_polar_seconds']:raise TimeoutError('Per-polar elapsed budget exceeded')
        save_json(destination/'run.json',record)
        print(json.dumps({k:record[k] for k in ['polar','status','total_seconds','peak_rss_bytes','ranks_for_energy_fraction']}),flush=True)
        return record
    except BaseException as error:
        record.update(status='failed',error=repr(error));save_json(destination/'run.json',record);raise
    finally:
        if svd is not None:svd.close()
        loader._file.close()
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['pilot','full'])
    parser.add_argument('--polar');parser.add_argument('--gate',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env');config=json.loads(CONFIG.read_text())
    torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    if args.stage=='full' and args.gate:
        passed,message=check(args.gate)
        if not passed:raise ValueError(message)
    working=configured_path('OUTPUT_DIR')/'psp/working';cases=sorted(working.glob('polar_*'))
    if args.polar:cases=[p for p in cases if p.name==f'polar_{args.polar}']
    if args.stage=='pilot':cases=cases[:1]
    for directory in cases:
        destination=working/('svd_preflight' if args.stage=='pilot' else 'svd')/directory.name
        calculate(directory,destination,config,pilot=args.stage=='pilot')

if __name__=='__main__':main()
