"""Reuse saved full SVD factors for the fixed-rank surface PAMSPOD study."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import resource
import time

from dotenv import load_dotenv
import numpy as np
import torch
from flowtorch.analysis import SVD, PAMSPOD, AMSPOD
from convert_surface import ROOT, configured_path, sha256
from check_experiment_gate import check
from plot_style import STYLE, WIDTH_MM, export
from reference_scales import strouhal


ALGORITHM_VERSION = 'surface-pamspod-from-svd-v1'


class RankView(SVD):
    """Non-owning rank view: FlowTorch's rank setter alone does not slice s/V."""
    def __init__(self, original, rank):
        if not 1 <= rank <= original.s.numel(): raise ValueError('Invalid retained rank')
        self.__dict__ = original.__dict__.copy()
        self._parent = original
        self.rank = rank
    @property
    def s(self): return self._parent.s[:self.rank]
    @property
    def V(self): return self._parent.V[:, :self.rank]
    def close(self): pass  # The original object owns its disk-backed basis.


def atomic_json(path, record):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)


def variants(config):
    return [(f'fixed_{k}',False,k,1e-5) for k in config['fixed_tapers']] + [
        (f'adaptive_{tol:g}',True,config['adaptive_max_tapers'],tol) for tol in config['adaptive_tolerances']]


def calculate(svd, dt, settings, directory):
    label,adaptive,cap,tol=settings
    started=time.perf_counter()
    record={'status':'running','label':label,'rank':svd.rank,'adaptive':adaptive,
        'requested_max_tapers':cap,'tolerance':tol,'dt':dt,'nfft':len(svd.V),
        'dtype':'float64/complex128','device':'cpu','subtract_mean':'in cached SVD only',
        'source_sha256':sha256(Path(__file__)),
        'algorithm_version':ALGORITHM_VERSION,
        'flowtorch_spod_sha256':sha256(Path(inspect.getfile(PAMSPOD)))}
    directory.mkdir(parents=True,exist_ok=False);atomic_json(directory/'run.json',record)
    try:
        spod=PAMSPOD.from_svd(svd,dt,adaptive=adaptive,max_tapers=cap,tolerance=tol,keep_n_modes=3,verbose=True)
        vals=spod.eigvals.detach().cpu();modes=AMSPOD.modes.fget(spod).detach().cpu()
        if not bool(torch.isfinite(vals).all() and torch.isfinite(modes).all()):raise ValueError('Nonfinite SPOD result')
        norms=modes[:,:,0].abs().square().sum(1)
        max_norm_error=float((norms-1).abs().max())
        if max_norm_error>1e-7:raise ValueError(f'Leading-mode normalization failure: {max_norm_error}')
        if float(vals.min()) < -1e-10*float(vals.abs().max()):raise ValueError('Negative spectral energy')
        logs=spod.log
        counts=logs['n_tapers'].numpy()
        convergence={str(k):[float(v) for v in values] for k,values in logs.get('convergence',{}).items()}
        residual=np.array([1-convergence[str(i)][-1] if str(i) in convergence and convergence[str(i)] else np.nan for i in range(len(counts))])
        width=torch.as_tensor(spod.half_bandwidth).numpy()
        np.savez(directory/'spectra.npz',frequency=spod.frequency.numpy(),eigvals=vals.numpy(),
            reduced_modes=modes.numpy(),n_tapers=counts,half_bandwidth=width,final_residual=residual)
        effective=min(cap,svd.rank,len(svd.V))
        record['spectra_sha256']=sha256(directory/'spectra.npz')
        record.update(status='complete',seconds=time.perf_counter()-started,effective_max_tapers=effective,
            observed_max_tapers=int(counts.max()),leading_mode_norm_max_error=max_norm_error,
            cap_hit_fraction=float(np.mean(counts==effective)),
            tolerance_met_fraction=float(np.mean(residual<=tol)) if adaptive else None,
            convergence=convergence,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        atomic_json(directory/'run.json',record)
        print(json.dumps({k:record[k] for k in ['label','seconds','observed_max_tapers','cap_hit_fraction','tolerance_met_fraction']}),flush=True)
    except BaseException as error:
        record.update(status='failed',error=repr(error));atomic_json(directory/'run.json',record);raise


def plot(output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=output/'exploratory/figures/spod';destination.mkdir(parents=True,exist_ok=True)
    for case in config['cases']:
        paths=[output/'working/spod'/case/label for label,*_ in variants(config)]
        if not any((p/'spectra.npz').exists() for p in paths):continue
        with plt.rc_context(STYLE):
            fig,axs=plt.subplots(3,1,figsize=(WIDTH_MM['double']/25.4,155/25.4),layout='constrained',sharex=True)
            for i,p in enumerate(paths):
                if not (p/'spectra.npz').exists():continue
                with np.load(p/'spectra.npz') as data:
                    f=strouhal(data['frequency'],'numerical');selection=f>0
                    label=p.name.replace('_',r'\_');line='-' if p.name.startswith('fixed') else '--'
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
                    axs[0].semilogy(f[selection],data['eigvals'][selection,0],label=label,ls=line,color=color)
                    axs[1].plot(f[selection],data['n_tapers'][selection],ls=line,color=color)
                    if p.name.startswith('adaptive'):
                        axs[2].semilogy(f[selection],data['final_residual'][selection],label=label,ls=line,color=color)
            axs[0].set_ylabel(r'Leading SPOD eigenvalue');axs[0].legend(ncol=3)
            axs[1].set_ylabel('Number of tapers');axs[1].set_ylim(bottom=0)
            axs[2].set_ylabel('Final leading-mode residual');axs[2].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
            axs[-1].set_xscale('log')
            for ax in axs:ax.grid(alpha=.2)
            export(fig,destination/case);plt.close(fig)


def compatible_result(p, saved, selected, setting):
    if not (p/'run.json').is_file() or not (p/'spectra.npz').is_file(): return False
    r=json.loads((p/'run.json').read_text())
    return r['status']=='complete' and r['rank']==selected and r['requested_max_tapers']==setting[2] and r['tolerance']==setting[3] and r.get('algorithm_version')==ALGORITHM_VERSION and r.get('flowtorch_spod_sha256')==sha256(Path(inspect.getfile(PAMSPOD))) and r.get('svd_metadata_sha256')==sha256(saved/'checkpoint/metadata.pt') and r.get('spectra_sha256')==sha256(p/'spectra.npz')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['pilot','full','plot']);parser.add_argument('--gate',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env',override=False)
    config=json.loads((ROOT/'config/spod_study.json').read_text())
    if config['selected_ranks'] is None:raise ValueError('Choose fixed ranks first')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    output=configured_path('OUTPUT_DIR')/'analysis'
    if args.stage=='plot':plot(output,config);return
    if args.stage=='full' and args.gate:
        valid,message=check(args.gate)
        if not valid:raise ValueError(message)
    cases=['cp'] if args.stage=='pilot' else list(config['cases'])
    for case in cases:
        saved=output/'working/svd'/case
        source=json.loads((saved/'run.json').read_text())
        assert source['status']=='complete' and source['fields']==config['cases'][case]
        selected=config['selected_ranks'][case]
        destination=output/'working/spod'/case;destination.mkdir(parents=True,exist_ok=True)
        settings=[('fixed_16',False,16,1e-5),('adaptive_0.0001',True,config['adaptive_max_tapers'],1e-4)] if args.stage=='pilot' else variants(config)
        if all(compatible_result(destination/setting[0],saved,selected,setting) for setting in settings):
            print(f'Reusing all completed {case} variants without loading the physical SVD basis',flush=True)
            continue
        original=SVD.load(saved/'checkpoint',basis_directory=destination)
        view=RankView(original,selected)
        try:
            for setting in settings:
                p=destination/setting[0]
                if p.exists():
                    r=json.loads((p/'run.json').read_text())
                    if compatible_result(p,saved,selected,setting):
                        print(f'Reusing {case}/{setting[0]}',flush=True);continue
                    raise FileExistsError(f'Incompatible result: {p}')
                print(f'Starting {case}/{setting[0]} from cached SVD at rank {selected}',flush=True)
                calculate(view,source['dt'],setting,p)
                r=json.loads((p/'run.json').read_text());r['svd_metadata_sha256']=sha256(saved/'checkpoint/metadata.pt');atomic_json(p/'run.json',r)
        finally:
            original.close();del original,view;gc.collect()
        plot(output,config)
    print(f'{args.stage} surface SPOD stage completed.',flush=True)

if __name__=='__main__':main()
