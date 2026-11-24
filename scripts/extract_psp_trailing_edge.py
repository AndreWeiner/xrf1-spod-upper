"""Pressure-based trailing-edge separation proxy at 95% local chord."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from extract_psp_mean_shocks import POLARS


def extract_section(mean, coordinates, keep, chord_fraction=.95, search=(.2,.65), sigma=2.):
    x=coordinates[...,0]
    xi=(x-x[:,:1])/(x[:,-1:]-x[:,:1])
    points=[];values=[];valid=[]
    for row in range(len(x)):
        j=int(np.searchsorted(xi[row],chord_fraction));j=np.clip(j,1,x.shape[1]-1)
        t=(chord_fraction-xi[row,j-1])/(xi[row,j]-xi[row,j-1])
        points.append((1-t)*coordinates[row,j-1]+t*coordinates[row,j])
        values.append((1-t)*mean[row,j-1]+t*mean[row,j])
        valid.append(keep[row,j-1] and keep[row,j])
    points=np.asarray(points);values=np.asarray(values);valid=np.asarray(valid)&np.isfinite(values)
    valid[:5]=False;valid[-5:]=False
    span=-points[:,1]
    fraction=(span-span.min())/(span.max()-span.min())
    if not np.all(np.diff(span)>0):raise ValueError('Nonmonotone section span')
    # Smooth at uniform physical span spacing; retain original validity separately.
    uniform=np.linspace(span.min(),span.max(),len(span))
    filtered=gaussian_filter1d(np.interp(uniform,span[valid],values[valid]),sigma,mode='reflect')
    slope=np.interp(span,uniform,np.gradient(filtered,uniform,edge_order=2))
    # Do not select a filled sample or filter across a long (>10 rows) gap.
    reliable=valid.copy()
    missing=np.flatnonzero(~valid)
    for gap in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
        if len(gap)>10:
            reliable[max(0,gap[0]-6):min(len(span),gap[-1]+7)]=False
    candidates=reliable&(fraction>=search[0])&(fraction<=search[1])
    if not candidates.any():raise ValueError('No valid samples in separation search region')
    index=int(np.argmax(np.where(candidates,np.abs(slope),-np.inf)))
    return dict(section_xyz_m=points,mean_cp=np.where(valid,values,np.nan),valid=valid,
        filtered_cp=np.interp(span,uniform,filtered),dcp_dspan_per_m=slope,
        span_fraction=fraction,candidate=candidates,index=index,
        marker_xyz_m=points[index],trailing_edge_xyz_m=coordinates[index,-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--chord-fraction',type=float,default=.95)
    parser.add_argument('--search',nargs=2,type=float,default=[.2,.65])
    parser.add_argument('--sigma',type=float,default=2.)
    args=parser.parse_args()
    if not 0<args.chord_fraction<1 or not 0<=args.search[0]<args.search[1]<=1 or args.sigma<=0:parser.error('Invalid section/search/sigma')
    load_dotenv(ROOT/'.env');base=configured_path('OUTPUT_DIR')/'psp'
    output=args.output_dir or base/'paper/trailing_edge_separation';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for polar in POLARS:
        directory=base/'working'/f'polar_{polar}'
        with np.load(directory/'fields.npz') as f,np.load(directory/'masks.npz') as m:
            mean=f['mean'];coordinates=f['coordinates']/1000;keep=m['p5_t10'].copy()
            keep[:5]=False;keep[-5:]=False;keep[:,:5]=False;keep[:,-5:]=False
            result=extract_section(mean,coordinates,keep,args.chord_fraction,args.search,args.sigma)
        np.savez_compressed(output/f'polar_{polar}.npz',**result)
        i=result['index'];run=json.loads((directory/'run.json').read_text())
        records.append(dict(polar=polar,angle_deg=run['angle_of_attack_deg'],row=i,
            span_fraction=float(result['span_fraction'][i]),span_m=float(-result['marker_xyz_m'][1]),
            marker_xyz_m=result['marker_xyz_m'].tolist(),dcp_dspan_per_m=float(result['dcp_dspan_per_m'][i]),
            inputs={n:sha256(directory/n) for n in ['fields.npz','masks.npz']}))
        print(polar,records[-1]['span_fraction'],flush=True)
    manifest=dict(settings={**vars(args),'output_dir':str(output)},cases=records,script_sha256=sha256(Path(__file__)),
        definition='Maximum absolute dcp/d(-y) along a near-trailing-edge constant local-chord section; pressure-based separation proxy, not a wall-shear criterion.',
        filtering='Use only original unmasked bracketing section samples; linear span interpolation for Gaussian filtering on uniform physical span; sigma in uniform sample steps. Selection restricted to valid samples and excludes long-gap neighborhoods.',
        span_normalization='(section span - section root span)/(section tip span - section root span)',marker='Cross on the sampled 95%-chord section, not projected to the geometric trailing edge.')
    (output/'extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (output/'markers.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['polar','angle_deg','span_fraction','span_m','x_m','y_m','z_m','dcp_dspan_per_m'])
        for r in records:writer.writerow([r['polar'],r['angle_deg'],r['span_fraction'],r['span_m'],*r['marker_xyz_m'],r['dcp_dspan_per_m']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import STYLE,export
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(5,2,figsize=(180/25.4,190/25.4),layout='constrained')
        for row,r in enumerate(records):
            with np.load(output/f'polar_{r["polar"]}.npz') as d:
                eta=d['span_fraction'];i=int(d['index'])
                axes[row,0].plot(eta,d['mean_cp'],'.',ms=2,label='Unmasked samples')
                axes[row,0].plot(eta,d['filtered_cp'],lw=1,label='Filtered profile')
                axes[row,1].plot(eta,d['dcp_dspan_per_m'],lw=1)
                for ax in axes[row]:
                    ax.axvspan(*args.search,color='.9');ax.axvline(eta[i],color='black',ls='--',lw=.8)
                    ax.set_xlim(0,1);ax.grid(alpha=.15)
                axes[row,0].set_ylabel(rf'$\alpha={r["angle_deg"]:.1f}^\circ$'+'\n'+r'$c_p$')
                axes[row,1].set_ylabel(r'$\mathrm{d}c_p/\mathrm{d}(-y)\;[\mathrm{m}^{-1}]$')
        for ax in axes[-1]:ax.set_xlabel('Fraction of displayed span')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
        export(fig,output/'section_diagnostics');plt.close(fig)
    print(output)


if __name__=='__main__':main()
