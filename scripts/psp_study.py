"""Full-record PSP statistics and automatic-mask sensitivity for every polar."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from dotenv import load_dotenv
import numpy as np
import torch
from scipy.signal import detrend, find_peaks
from flowtorch.analysis import snapshot_statistics, spatial_statistics
from flowtorch.data import PSPDataloader, create_image_mask, grid_element_areas, element_areas_to_node_weights
from convert_surface import ROOT, configured_path, sha256
from check_experiment_gate import check
from plot_style import STYLE, WIDTH_MM, export
from reference_scales import strouhal

CONFIG=ROOT/'config/psp_study.json'


def save_json(path,record):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)


def spectrum(signal,fs):
    """One-sided, coherent-gain-corrected Hann amplitude spectra; no zero padding."""
    x=np.asarray(signal,dtype=float);n=len(x)
    window=np.hanning(n)
    centered=x-x.mean()
    amplitude=abs(np.fft.rfft(centered*window))/window.sum()
    detrended=abs(np.fft.rfft(detrend(x,type='linear')*window))/window.sum()
    rectangle=abs(np.fft.rfft(centered))/n
    for a in (amplitude,detrended,rectangle):
        a[1:-1 if n%2==0 else None]*=2
    frequency=np.fft.rfftfreq(n,d=1/fs)
    peaks=find_peaks(amplitude)[0];peaks=peaks[peaks>0]
    peaks=peaks[np.argsort(amplitude[peaks])[::-1]][:5]
    return {'frequency_hz':frequency,'hann_amplitude':amplitude,
        'linearly_detrended_hann_amplitude':detrended,'rectangular_amplitude':rectangle,
        'peak_bins':peaks}


def masks_from_scores(extreme,score,thresholds,domain):
    return {str(float(q)):(~extreme)&(score<=q)&domain for q in thresholds}


def calculate(path,directory,config,limit=None):
    directory.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    record={'status':'running','polar':path.stem,'source_file':path.name,'source_sha256':sha256(path),
        'code_sha256':sha256(Path(__file__)),'config_sha256':sha256(CONFIG),
        'started_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'versions':{n:importlib.metadata.version(n) for n in ['flowtorch-fluid','torch','numpy','scipy','h5py']},
        'scope':'preliminary mask comparison; final mask not yet selected'}
    save_json(directory/'run.json',record)
    loader=PSPDataloader(str(path),dtype=torch.float64)
    try:
        names=[]
        for zone in loader.zone_names:
            loader.zone=zone
            if str(loader.zone_info['ZoneName'][0])==config['zone_name']:names.append(zone)
        if len(names)!=1:raise ValueError('Wing zone is ambiguous or missing')
        loader.zone=names[0]
        times=loader.write_times[:limit] if limit else loader.write_times
        fs=float(loader.zone_info['SamplingFrequency'][0]);time_values=np.arange(len(times))/fs
        record.update(zone=loader.zone,angle_of_attack_deg=float(loader.info['AngleAttackAlpha'][0]),
            mach=float(loader.info['Mach'][0]),reynolds=float(loader.info['Reynolds'][0]),
            sample_rate_hz=fs,snapshots=len(times),threads=torch.get_num_threads())
        coordinates=loader.vertices
        area=element_areas_to_node_weights(grid_element_areas(*coordinates.unbind(-1)),normalize=False)
        if not bool(torch.isfinite(area).all() and (area>0).all()):raise ValueError('Invalid geometric area weights')
        print(f'{path.stem}: loading all {len(times)} Wing snapshots',flush=True)
        data=loader.load_snapshot('Cp',times)
        baseline=torch.isfinite(data).all(-1)
        if not bool(baseline.any()):raise ValueError('No pixels finite throughout record')
        record.update(shape=list(data.shape),finite_all_times_pixels=int(baseline.sum()),total_pixels=baseline.numel())
        native={}
        for name in loader.mask_names:
            loader.mask=name;native[name]=(loader.weights>0).numpy()
        moments=snapshot_statistics(data,time=time_values,compute=['moments','trend'],
            spatial_mask=baseline,spatial_weight=area,batch_size=config['statistics_batch_size'],accumulator_dtype=torch.float64)
        fields=moments.moments
        np.savez(directory/'fields.npz',coordinates=coordinates.numpy(),area_weights=area.numpy(),
            finite_keep=baseline.numpy(),**{name:value.numpy() for name,value in fields._asdict().items()},
            trend_slope=moments.trend.slope.numpy(),trend_r_squared=moments.trend.r_squared.numpy(),
            **{f'native_{k}':v for k,v in native.items()})
        masks={'unmasked_finite':baseline};scores={}
        for patch in config['patch_sizes']:
            phase=time.perf_counter()
            keep,diagnostics=create_image_mask(data,patch_size=patch,threshold=config['default_threshold'],
                lower_bound=config['lower_bound'],upper_bound=config['upper_bound'],temporal_reduction='max',chunk_size=config['mask_chunk_size'])
            derived=masks_from_scores(diagnostics.extreme_mask,diagnostics.spatial_score,config['thresholds'],baseline)
            assert torch.equal(derived[str(float(config['default_threshold']))],keep&baseline)
            ordered=[derived[str(float(q))] for q in config['thresholds']]
            assert all(bool((~a|b).all()) for a,b in zip(ordered,ordered[1:]))
            for threshold,mask in derived.items():masks[f'p{patch}_t{float(threshold):g}']=mask
            scores[f'p{patch}_score']=diagnostics.spatial_score.numpy()
            scores[f'p{patch}_extreme_count']=diagnostics.extreme_count.numpy()
            print(f'{path.stem}: patch {patch} complete in {time.perf_counter()-phase:.1f}s; full-record maximum scores',flush=True)
            if time.perf_counter()-started>config['maximum_polar_seconds']:raise TimeoutError('Per-polar time budget exceeded')
        np.savez(directory/'masks.npz',**{k:m.numpy() for k,m in masks.items()},**scores)
        rows=[]
        for label,mask in masks.items():
            if not bool(mask.any()):
                rows.append({'label':label,'status':'empty','retained_pixels':0,'retained_area_fraction':0});continue
            phase=time.perf_counter()
            stats=spatial_statistics(data,spatial_weight=area,spatial_mask=mask,
                quantiles=config['quantiles'],batch_size=config['statistics_batch_size'])
            w=area[mask];mean=stats.mean.numpy()
            expected=float((fields.mean[mask]*w).sum()/w.sum())
            if not np.isclose(mean.mean(),expected,rtol=1e-10,atol=1e-12):raise ValueError('Space/time mean commutation failure')
            fourier=spectrum(mean,fs)
            np.savez(directory/f'{label}.npz',time_seconds=time_values,mean=mean,
                minimum=stats.minimum.numpy(),maximum=stats.maximum.numpy(),quantiles=stats.quantiles.numpy(),
                quantile_levels=stats.quantile_levels.numpy(),**fourier)
            weighted_fields={}
            for name,field in fields._asdict().items():
                valid=mask&torch.isfinite(field)
                weighted_fields[name]=float((field[valid]*area[valid]).sum()/area[valid].sum()) if bool(valid.any()) else None
            row={'label':label,'status':'complete','retained_pixels':int(mask.sum()),
                'retained_pixel_fraction':float(mask.double().mean()),
                'retained_area_fraction':float(area[mask].sum()/area.sum()),
                'area_mean_temporal_moments':weighted_fields,'mean_time_average':float(mean.mean()),
                'spatial_mean_temporal_std':float(mean.std()),
                'time_average_spatial_quantiles':dict(zip(map(str,config['quantiles']),stats.quantiles.mean(1).tolist())),
                'candidate_peak_frequencies_hz':fourier['frequency_hz'][fourier['peak_bins']].tolist(),
                'candidate_peak_amplitudes':fourier['hann_amplitude'][fourier['peak_bins']].tolist(),
                'seconds':time.perf_counter()-phase}
            rows.append(row)
            print(f'{path.stem}: statistics {label} complete; area retained {100*row["retained_area_fraction"]:.2f}%',flush=True)
            if time.perf_counter()-started>config['maximum_polar_seconds']:raise TimeoutError('Per-polar time budget exceeded')
        record.update(status='complete',cases=rows,total_seconds=time.perf_counter()-started,
            native_mask_retained_pixels={k:int(v.sum()) for k,v in native.items()},
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            frequency_resolution_hz=fs/len(times),finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(directory/'run.json',record)
        return record
    except BaseException as error:
        record.update(status='failed',error=repr(error));save_json(directory/'run.json',record);raise
    finally:loader._file.close()


def plot_polar(directory,destination,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    record=json.loads((directory/'run.json').read_text());destination.mkdir(parents=True,exist_ok=True)
    lookup={row['label']:row for row in record['cases']}
    with np.load(directory/'masks.npz') as arrays:
        with plt.rc_context(STYLE):
            fig,axs=plt.subplots(3,3,figsize=(WIDTH_MM['double']/25.4,105/25.4),layout='constrained')
            for i,patch in enumerate(config['patch_sizes']):
                for j,threshold in enumerate(config['thresholds']):
                    label=f'p{patch}_t{threshold:g}';mask=arrays[label];row=lookup[label]
                    axs[i,j].imshow(mask.T,origin='lower',aspect='equal',vmin=0,vmax=1,cmap='gray',interpolation='nearest')
                    axs[i,j].set_title(f'Patch {patch}, threshold {threshold:g}\nRetained area {100*row["retained_area_fraction"]:.1f}'+r'\%')
                    if i==2:axs[i,j].set_xlabel('Grid index 1')
                    if j==0:axs[i,j].set_ylabel('Grid index 2')
            fig.suptitle('Retained-pixel masks: white retained, black excluded',fontsize=10)
            export(fig,destination/'mask_grid');plt.close(fig)
    default=f'p{config["default_patch_size"]}_t{config["default_threshold"]:g}'
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,1,figsize=(WIDTH_MM['double']/25.4,105/25.4),layout='constrained')
        for label in ['unmasked_finite',default]:
            if lookup[label]['status']!='complete':continue
            with np.load(directory/f'{label}.npz') as data:
                name='Finite-domain reference' if label=='unmasked_finite' else 'Provisional default mask'
                axs[0].plot(data['time_seconds'],data['mean'],label=name,lw=.8)
                st=strouhal(data['frequency_hz'],'psp',record['polar']);select=st>0
                axs[1].semilogy(st[select],data['hann_amplitude'][select],label=name,lw=1)
                if label==default:
                    axs[1].semilogy(st[select],data['linearly_detrended_hann_amplitude'][select],label='Default, linear trend removed',ls='--',lw=1)
        axs[0].set(xlabel='Time [s]',ylabel=r'Area-weighted spatial mean $C_p$')
        axs[1].set(xlabel=r'$St=f c_{\mathrm{MAC}}/U_\infty$',ylabel=r'Hann amplitude [$C_p$]')
        for ax in axs:ax.grid(alpha=.2);ax.legend()
        export(fig,destination/'provisional_spatial_mean_spectrum');plt.close(fig)
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,1,figsize=(WIDTH_MM['double']/25.4,105/25.4),layout='constrained',sharex=True)
        for ax,label in zip(axs,['unmasked_finite',default]):
            if lookup[label]['status']!='complete':continue
            with np.load(directory/f'{label}.npz') as data:
                t=data['time_seconds'];q=data['quantiles']
                for lower,upper,alpha in [(0,6,.12),(1,5,.2),(2,4,.3)]:
                    ax.fill_between(t,q[lower],q[upper],color='#0072B2',alpha=alpha,linewidth=0)
                ax.plot(t,q[3],color='#0072B2',label='Spatial median',lw=.7)
                ax.plot(t,data['mean'],color='#D55E00',label='Area mean',lw=.7)
                ax.set_title('Finite-domain reference' if label=='unmasked_finite' else 'Provisional default mask')
                ax.set_ylabel(r'Spatial $C_p$ quantiles');ax.legend(ncol=2);ax.grid(alpha=.2)
        axs[-1].set_xlabel('Time [s]')
        export(fig,destination/'spatial_quantile_bands');plt.close(fig)
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,3,figsize=(WIDTH_MM['double']/25.4,65/25.4),layout='constrained')
        reference=lookup['unmasked_finite']
        grids=[np.zeros((3,3)) for _ in range(3)]
        for i,patch in enumerate(config['patch_sizes']):
            for j,threshold in enumerate(config['thresholds']):
                row=lookup[f'p{patch}_t{threshold:g}']
                grids[0][i,j]=100*row['retained_area_fraction']
                grids[1][i,j]=row['mean_time_average']-reference['mean_time_average'] if row['status']=='complete' else np.nan
                grids[2][i,j]=100*(np.sqrt(row['area_mean_temporal_moments']['variance']/reference['area_mean_temporal_moments']['variance'])-1) if row['status']=='complete' else np.nan
        for ax,values,title in zip(axs,grids,['Retained area (percent)',r'Mean shift $\Delta C_p$','Fluctuation RMS change (percent)']):
            image=ax.imshow(values,origin='upper',cmap='viridis',interpolation='nearest')
            ax.set_xticks(range(3),[f'{x:g}' for x in config['thresholds']]);ax.set_yticks(range(3),config['patch_sizes'])
            ax.set_xlabel('Score threshold');ax.set_ylabel('Patch size');ax.set_title(title)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{values[i,j]:.3g}',ha='center',va='center',color='white' if values[i,j]<np.nanmean(values) else 'black')
            fig.colorbar(image,ax=ax,orientation='horizontal',shrink=.8)
        export(fig,destination/'mask_statistics_sensitivity');plt.close(fig)
    with np.load(directory/'fields.npz') as data,np.load(directory/'masks.npz') as masks:
        with plt.rc_context(STYLE):
            fig,axs=plt.subplots(2,4,figsize=(WIDTH_MM['double']/25.4,90/25.4),layout='constrained')
            for j,name in enumerate(['mean','variance','skewness','kurtosis']):
                values=data[name];valid=masks[default];lo,hi=np.nanquantile(values[valid],[.01,.99])
                for i,mask in enumerate([data['finite_keep'],valid]):
                    image=axs[i,j].imshow(np.where(mask,values,np.nan).T,origin='lower',aspect='equal',vmin=lo,vmax=hi,cmap='viridis',interpolation='nearest')
                    axs[i,j].set_title(name.capitalize() if i==0 else 'Default retained domain');axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
                fig.colorbar(image,ax=axs[:,j],orientation='horizontal',shrink=.85)
            export(fig,destination/'temporal_moments');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['pilot','full','plot']);parser.add_argument('--gate',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env',override=False)
    config=json.loads(CONFIG.read_text());torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    output=configured_path('OUTPUT_DIR')/'psp';working=output/'working';working.mkdir(parents=True,exist_ok=True)
    (output/'paper/figures').mkdir(parents=True,exist_ok=True)
    files=([Path(json.loads(p.read_text())['source_file']) for p in sorted(working.glob('polar_*/run.json'))]
           if args.stage=='plot' else sorted(configured_path('PSP_DATA_DIR').glob('*.hdf5')))
    if not files:raise FileNotFoundError('No PSP polars found')
    if args.stage=='full' and args.gate:
        valid,message=check(args.gate)
        if not valid:raise ValueError(message)
    if args.stage=='pilot':files=[next(p for p in files if p.stem=='0258')]
    for path in files:
        directory=working/('pilot' if args.stage=='pilot' else f'polar_{path.stem}')
        if args.stage=='plot':
            if (directory/'run.json').exists() and json.loads((directory/'run.json').read_text())['status']=='complete':
                plot_polar(directory,output/'exploratory/figures'/path.stem,config)
            continue
        if directory.exists():
            record=json.loads((directory/'run.json').read_text())
            if record['status']=='complete' and record['config_sha256']==sha256(CONFIG) and record['code_sha256'] in (sha256(Path(__file__)), '1045da305f44242dea7de970bae8f5817cacddf6465bd42bd382cbeb3a2c6850'):
                print(f'Reusing complete polar {path.stem}',flush=True);continue
            raise FileExistsError(f'Incompatible/incomplete output: {directory}')
        calculate(path,directory,config,limit=32 if args.stage=='pilot' else None)
        if args.stage=='full':plot_polar(directory,output/'exploratory/figures'/path.stem,config)
    if args.stage in ('full','plot'):
        from review_psp import summarize
        summarize(output,config)
    print(f'PSP {args.stage} stage complete.',flush=True)

if __name__=='__main__':main()
