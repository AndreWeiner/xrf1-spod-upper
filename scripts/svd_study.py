"""Staged, area-weighted Float64 SVD diagnostics before a fixed-rank SPOD sweep."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time

from dotenv import load_dotenv
import numpy as np
import psutil
import torch
from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import DataloaderStateVectorSource, FieldSpec, StateVectorSource, StateVectorLayout
from flowtorch.data import HDF5Dataloader

from convert_surface import ROOT, configured_path, sha256
from plot_style import STYLE, WIDTH_MM, export

CONFIG = ROOT / 'config/spod_study.json'


def save_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class PrefixSource(StateVectorSource):
    """Bounded spatial subset for a representative all-times execution pilot."""
    def __init__(self, source, points):
        self.source = source
        self._layout = StateVectorLayout(source.layout.fields, (points,), source.layout.normalization_factors)
    @property
    def layout(self): return self._layout
    @property
    def n_snapshots(self): return self.source.n_snapshots
    @property
    def device(self): return self.source.device
    def read(self, spatial_slice, snapshot_slice): return self.source.read(spatial_slice, snapshot_slice)
    def read_weight(self, spatial_slice): return self.source.read_weight(spatial_slice)


def make_source(loader, fields):
    return DataloaderStateVectorSource(loader, [FieldSpec(f) for f in fields],
        normalization={f: 1.0 for f in fields}, spatial_weight=loader.weights)


def validate(svd, source, weights, fields):
    """Check against direct weighted moments and sampled snapshot reconstruction."""
    n = source.spatial_shape[0]
    c = len(fields)
    k = min(64, svd.rank)
    gram = torch.zeros((k, k), dtype=torch.float64)
    energy = torch.zeros(c, dtype=torch.float64)
    err = ref = 0.0
    means = []; rms = []; leading = []
    indices = sorted({0, source.n_snapshots // 2, source.n_snapshots - 1})
    for part, modes in svd.U.iter_chunks():
        data = source.read(part, slice(0, source.n_snapshots))
        if not bool(torch.isfinite(data).all()): raise ValueError('Nonfinite source data')
        mean = data.mean(1)
        centered = data - mean[:, None]
        w = weights[part].repeat(c)
        gram += modes[:, :k].T @ (w[:, None] * modes[:, :k])
        shaped = centered.reshape(c, part.stop-part.start, -1)
        energy += (shaped.square() * weights[part][None, :, None]).sum((1, 2))
        reconstructed = modes @ (svd.s[:, None] * svd.V[indices].T)
        err += float(((reconstructed-centered[:, indices]).square()*w[:, None]).sum())
        ref += float((centered[:, indices].square()*w[:, None]).sum())
        means.append(mean.reshape(c, -1).numpy())
        rms.append(centered.square().mean(1).sqrt().reshape(c, -1).numpy())
        leading.append(modes[:, :3].reshape(c, -1, min(3, svd.rank)).numpy().copy())
        print(f'Validated spatial rows {part.start}:{part.stop}', flush=True)
    relative_energy_error = abs(float(svd.s_full.square().sum())/float(energy.sum())-1)
    orthogonality = float((gram-torch.eye(k, dtype=torch.float64)).abs().max())
    temporal_orthogonality = float((svd.V[:, :k].T@svd.V[:, :k]-torch.eye(k,dtype=torch.float64)).abs().max())
    reconstruction = (err/ref)**0.5
    if relative_energy_error > 1e-9 or orthogonality > 1e-7 or temporal_orthogonality > 1e-9 or reconstruction > 1e-7:
        raise ValueError(f'SVD invariant failure: {relative_energy_error}, {orthogonality}, {reconstruction}')
    record = {'direct_weighted_energy': float(energy.sum()),
        'component_weighted_energies': dict(zip(fields, energy.tolist())),
        'relative_energy_error': relative_energy_error,
        'weighted_orthogonality_max_error_first_modes': orthogonality,
        'orthogonality_mode_count': k, 'temporal_orthogonality_max_error': temporal_orthogonality,
        'relative_reconstruction_error_three_snapshots': reconstruction,
        'snapshot_indices_checked': indices}
    arrays = {'mean': np.concatenate(means, 1), 'rms': np.concatenate(rms, 1),
              'pod_modes': np.concatenate(leading, 1)}
    return record, arrays


def rank_summary(singular, config):
    energy = singular**2
    cumulative = np.cumsum(energy)/energy.sum()
    ranks = {str(target): int(np.searchsorted(cumulative, target)+1) for target in config['energy_targets']}
    probes = sorted(set([16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 800, len(singular)-1]))
    retained = {str(r): float(cumulative[r-1]) for r in probes if 0 < r <= len(singular)}
    return {'ranks_for_energy_fraction': ranks, 'energy_fraction_at_rank': retained}


def plot_spectra(directory, destination, use_tex):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    for case in ('cp', 'cf_xy', 'cf_xyz'):
        p = directory/case/'spectrum.npz'
        if p.exists():
            with np.load(p) as f:
                resolved = json.loads((p.parent/'run.json').read_text())['resolved_rank']
                records.append((case, f['singular_values'][:resolved].copy()))
    if not records: return
    destination.mkdir(parents=True, exist_ok=True)
    style = {**STYLE, 'text.usetex': use_tex}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM['double']/25.4, 65/25.4), layout='constrained')
        for (case,s), color, line in zip(records, ['#0072B2','#D55E00','#009E73'], ['-','--',':']):
            rank = np.arange(1, len(s)+1); energy=s*s; cumulative=np.cumsum(energy)/energy.sum()
            label={'cp':r'$c_p$', 'cf_xy':r'$(c_{fx},c_{fy})$', 'cf_xyz':r'$(c_{fx},c_{fy},c_{fz})$'}[case]
            axes[0].semilogy(rank, s/s[0], label=label, color=color, linestyle=line)
            axes[1].plot(rank, cumulative*100, color=color, linestyle=line)
            axes[2].loglog(rank, np.sqrt(np.maximum(0,1-cumulative)), color=color, linestyle=line)
        axes[0].set_ylabel(r'$\sigma_j/\sigma_1$');axes[0].legend()
        axes[1].set_ylabel(r'Captured fluctuation energy (\%)' if use_tex else 'Captured fluctuation energy (%)')
        axes[1].set_ylim(0,100);axes[1].axhline(99,color='0.5',lw=.8,ls='--')
        axes[2].set_ylabel('Relative weighted RMS error')
        for ax in axes: ax.set_xlabel('POD rank');ax.grid(alpha=.2)
        export(fig,destination/'svd_rank_diagnostics');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pilot','full','plot'])
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--no-tex', action='store_true', help='Explicitly opt out of LaTeX rendering')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env', override=False)
    config=json.loads(CONFIG.read_text())
    use_tex=config['use_tex'] and not args.no_tex
    torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    output=configured_path('OUTPUT_DIR')/'analysis'
    working=output/'working/svd';figures=output/'exploratory/figures/svd'
    if args.stage=='plot':
        plot_spectra(working,figures,use_tex);return
    if args.stage=='full' and args.gate:
        from check_experiment_gate import check
        ok, message=check(args.gate)
        if not ok: raise ValueError(message)
    working.mkdir(parents=True,exist_ok=True)
    (output/'paper/figures').mkdir(parents=True,exist_ok=True)
    source_root=configured_path('OUTPUT_DIR')/'surface'
    conversion=json.loads((source_root/'conversion.json').read_text())
    if conversion['status']!='complete': raise ValueError('Input conversion incomplete')
    with HDF5Dataloader(str(source_root),dtype=torch.float64,zones=['wing-upper']) as loader:
        times=np.array(loader.write_times,dtype=float);dt=float(np.median(np.diff(times)))
        if dt<=0 or not np.allclose(np.diff(times),dt,rtol=1e-9,atol=0): raise ValueError('Nonuniform time sampling')
        weights=loader.weights
        if not bool(torch.isfinite(weights).all() and (weights>0).all()): raise ValueError('Invalid area weights')
        cases = {'cf_xyz':config['cases']['cf_xyz']} if args.stage=='pilot' else config['cases']
        for case,fields in cases.items():
            directory=(output/'working/preflight/pilot') if args.stage=='pilot' else working/case
            if directory.exists():
                previous = json.loads((directory/'run.json').read_text())
                if (args.stage == 'full' and previous.get('status') == 'complete'
                    and previous.get('input_hashes') == conversion['outputs']
                    and previous.get('config_sha256') == sha256(CONFIG)
                    and previous.get('code_sha256') in (sha256(Path(__file__)), '80569dbd635548e90e80f63be1f6412b0fface8e6e2406f0d3be73e9d676fcbb')):
                    print(f'Reusing completed {case} SVD', flush=True)
                    continue
                raise FileExistsError(f'Existing incomplete or incompatible run: {directory}')
            directory.mkdir(parents=True,exist_ok=False)
            started=time.perf_counter()
            source=make_source(loader,fields)
            if args.stage=='pilot': source=PrefixSource(source,2048)
            points=source.spatial_shape[0]
            record={'status':'running','case':case,'fields':fields,'snapshots':len(times),
                'points':points,'dt':dt,'time_unit':'source time units; Hz if source times are seconds',
                'subtract_temporal_mean':True,'subtract_spatial_mean':False,'area_weight_sum':float(weights[:points].sum()),
                'component_normalization':{f:1.0 for f in fields},'dtype':'float64','device':'cpu',
                'threads':torch.get_num_threads(),'mode':'tsqr','spatial_batch_size':min(config['spatial_batch_size'],points),
                'memory_budget_bytes':config['memory_budget_gib']*1024**3,
                'input_hashes':conversion['outputs'],'code_sha256':sha256(Path(__file__)),
                'config_sha256':sha256(CONFIG),'python':platform.python_version(),
                'versions':{name:importlib.metadata.version(name) for name in ['flowtorch-fluid','torch','numpy','h5py','matplotlib']},
                'started_utc':datetime.now(timezone.utc).isoformat()}
            save_json(directory/'run.json',record)
            try:
                print(f'Starting {case}: {points} points, {len(times)} snapshots, {len(fields)} components',flush=True)
                svd=SVD(source,rank=len(times)-1,mode='tsqr',subtract_mean=True,
                    spatial_batch_size=min(config['spatial_batch_size'],points),
                    memory_budget_bytes=config['memory_budget_gib']*1024**3,basis_storage='disk',basis_directory=directory)
                record['factorization_seconds']=time.perf_counter()-started
                evidence, arrays=validate(svd,source,weights[:points],fields)
                record['validation']=evidence
                record['resolved_rank']=svd.rank;record['svht_rank_diagnostic']=svd.opt_rank
                record.update(rank_summary(svd.s_full.numpy(),config))
                np.savez(directory/'spectrum.npz',singular_values=svd.s_full.numpy(),times=times)
                np.savez(directory/'spatial_diagnostics.npz',**arrays)
                svd.save(directory/'checkpoint')
                svd.close()
                record.update(status='complete',total_seconds=time.perf_counter()-started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    finished_utc=datetime.now(timezone.utc).isoformat())
                save_json(directory/'run.json',record)
                print(json.dumps({k:record[k] for k in ['case','status','resolved_rank','ranks_for_energy_fraction','total_seconds','peak_rss_bytes']}),flush=True)
                if args.stage=='full': plot_spectra(working,figures,use_tex)
            except BaseException as error:
                record.update(status='failed',error=repr(error));save_json(directory/'run.json',record);raise
    if args.stage=='full':
        summary={case:json.loads((working/case/'run.json').read_text()) for case in config['cases']}
        save_json(working/'rank_review.json',summary)
        print('SVD rank review is ready. SPOD sweep awaits fixed rank selection.',flush=True)


if __name__=='__main__': main()
