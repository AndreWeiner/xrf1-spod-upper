"""Map full-chord leading SPOD energy on the phase-speed panel grid."""
import argparse
import json
import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import POLARS, read_case, prefix_basis
from plot_style import STYLE, export
from psp_settings import selected_result

CASES = ['0257','0258','azdes','0259','0260','0261']


def compute(output, working, key, edges):
    case = read_case(output,key)
    prefix_basis(case,working)
    numerical = key=='azdes'
    polar = '0258' if numerical else key
    grid_path = output/'psp/working'/f'polar_{polar}'/'fields.npz'
    with np.load(grid_path) as data:
        grid_y = -data['coordinates'][...,1]/1000
    if numerical:
        with h5py.File(output/'surface/geometry.h5') as geometry:
            y = geometry['zones/z000000/points'][:,1]
        spectra = output/'analysis/working/spod_rank_sensitivity/cp/rank_200/adaptive_0.0001/spectra.npz'
    else:
        y = -case['coordinates'][:,1]/1000
        spectra = selected_result(output,key)/'spectra.npz'
    span = (y-grid_y.min())/np.ptp(grid_y)
    keep = (case['St']>=.1)&(case['St']<=.7)
    st = case['St'][keep]
    coeff = case['coeff'][:,keep]
    assert sha256(spectra)==case['source']['spectra_sha256']
    with np.load(spectra) as data:
        eigenvalue = data['eigvals'][keep,0]
    basis,area = case['basis'],case['weight']
    inside = (span>=edges[0])&(span<=edges[-1])
    labels = np.minimum(np.searchsorted(edges,span,side='right')-1,len(edges)-2)
    parts = [inside&(labels==j) for j in range(len(edges)-1)]+[~inside]
    fractions = []
    selected = np.unique(np.linspace(0,len(st)-1,6,dtype=int))
    for part in parts:
        weighted = basis[part]*area[part,None].sqrt()
        gram = (weighted.T@weighted).to(torch.complex128)
        share = (coeff.conj()*(gram@coeff)).sum(0).real.numpy()
        assert share.min()>=-1e-11
        direct = (weighted.to(torch.complex128)@coeff[:,selected]).abs().square().sum(0).numpy()
        np.testing.assert_allclose(share[selected],direct,atol=1e-11,rtol=1e-9)
        fractions.append(share.clip(0))
    fractions = np.array(fractions)
    np.testing.assert_allclose(fractions.sum(0),1,atol=1e-8,rtol=0)
    np.savez_compressed(working/f'polar_{key}_full_chord_energy.npz',St=st,span_edges=edges,
                        modal_fraction=fractions[:-1],outside_fraction=fractions[-1],
                        leading_eigenvalue=eigenvalue,segment_energy=fractions[:-1]*eigenvalue)
    print(key,len(st),'bins',flush=True)
    return st,eigenvalue,fractions[:-1]*eigenvalue,case['source']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    markers = parser.add_mutually_exclusive_group()
    markers.add_argument('--mark-concentration',action='store_true')
    markers.add_argument('--span-integrated-peak',action='store_true',
                         help='Mark the peak of the energy summed over the displayed span')
    parser.add_argument('--buffered-peak',action='store_true')
    parser.add_argument('--muted',action='store_true',help='Reduce colormap contrast')
    args = parser.parse_args()
    if args.buffered_peak and not args.mark_concentration:
        parser.error('--buffered-peak requires --mark-concentration')
    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    working = output/'psp/working/spanwise_mode_energy'
    working.mkdir(parents=True,exist_ok=True)
    destination = output/'psp/exploratory/figures/spanwise_mode_energy'
    destination.mkdir(parents=True,exist_ok=True)
    edges = np.linspace(.45,1,45)
    cases = {key:compute(output,working,key,edges) for key in CASES}
    # Use the established dataset references from the spectral comparison.
    peaks = {key:float(np.max(cases[key][1][cases[key][0]>=.2])) for key in ['0258','azdes']}
    maps = {key:cases[key][2]/peaks['azdes' if key=='azdes' else '0258'] for key in CASES}
    integrated_maps = {key:values.sum(axis=0) for key,values in maps.items()}
    integrated_limit = 1.08*max(values.max() for values in integrated_maps.values())
    positive = np.concatenate([values[values>0] for values in maps.values()])
    lo,hi = np.quantile(positive,[.01,.995])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    cmap = plt.get_cmap('viridis')
    if args.muted:
        colors = cmap(np.linspace(0,1,256))
        colors[:,:3] = .72*colors[:,:3]+.28*.5
        cmap = ListedColormap(colors,name='muted_viridis')
    for upper,tag in [(.35,'035'),(.7,'070')]:
        with plt.rc_context(STYLE):
            fig,axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=(240/25.4,145/25.4),layout='constrained')
            annotations = {}
            for panel,(ax,key) in enumerate(zip(axes.flat,CASES)):
                st = cases[key][0]
                keep = st<=upper
                freq = st[keep]
                bounds = np.r_[freq[0]-np.diff(freq)[0]/2,(freq[:-1]+freq[1:])/2,freq[-1]+np.diff(freq)[-1]/2]
                mesh = ax.pcolormesh(bounds,edges,maps[key][:,keep],cmap=cmap,
                                     norm=LogNorm(vmin=lo,vmax=hi),rasterized=True)
                if args.span_integrated_peak:
                    search = np.flatnonzero(st>=.25)
                    integrated = integrated_maps[key]
                    peak_st = st[search[integrated[search].argmax()]]
                    spectral_peak_st = st[search[cases[key][1][search].argmax()]]
                    # Smooth only eligible bins so lower frequencies cannot influence the peak.
                    smoothed = gaussian_filter(maps[key][:,search],
                                               sigma=(1,.01/np.median(np.diff(st))))
                    i,j = np.unravel_index(smoothed.argmax(),smoothed.shape)
                    local_st = st[search[j]]
                    local_eta = (edges[i]+edges[i+1])/2
                    energy_ax = ax.twinx()
                    energy_ax.patch.set_visible(False)
                    energy_ax.grid(False)
                    energy_ax.plot(freq,integrated[keep],color='C3',lw=1.15,zorder=5)
                    energy_ax.set_ylim(0,integrated_limit)
                    energy_ax.tick_params(axis='y',colors='C3',right=panel%3==2,
                                          labelright=panel%3==2)
                    if panel%3==2:
                        energy_ax.set_ylabel('Integrated energy / reference',color='C3')
                    if local_st<=upper:
                        energy_ax.plot(local_st,local_eta,transform=ax.transData,
                                       marker='*',ms=8,mfc='#ffbf48',mec='.15',mew=.65,zorder=6)
                    annotations[key] = {'peak_St':float(peak_st),
                                        'leading_eigenvalue_peak_St':float(spectral_peak_st),
                                        'smoothed_local_peak_St':float(local_st),
                                        'smoothed_local_peak_span':float(local_eta)}
                if args.mark_concentration:
                    values = maps[key][:,keep]
                    eligible = np.broadcast_to(freq>.2,values.shape)
                    rows,cols = np.where(eligible)
                    widths = np.diff(bounds)
                    mass = values[rows,cols]*widths[cols]
                    order = np.argsort(-values[rows,cols],kind='stable')
                    count = np.searchsorted(np.cumsum(mass[order]),.90*mass.sum())+1
                    selected = order[:count]
                    region = np.zeros_like(eligible)
                    region[rows[selected],cols[selected]] = True
                    boundary = []
                    for i,j in np.argwhere(region):
                        x0,x1 = max(bounds[j],.2),min(bounds[j+1],upper)
                        y0,y1 = edges[i:i+2]
                        for outside,segment in [
                            (j==0 or not region[i,j-1],[(x0,y0),(x0,y1)]),
                            (j==region.shape[1]-1 or not region[i,j+1],[(x1,y0),(x1,y1)]),
                            (i==0 or not region[i-1,j],[(x0,y0),(x1,y0)]),
                            (i==region.shape[0]-1 or not region[i+1,j],[(x0,y1),(x1,y1)]),
                        ]:
                            if outside:
                                boundary.append(segment)
                    ax.add_collection(LineCollection(boundary,colors='white',linewidths=.65,zorder=3))
                    if args.buffered_peak:
                        search = np.flatnonzero(st>=.25)
                        smoothed = gaussian_filter1d(maps[key],.01/np.median(np.diff(st)),axis=1)
                        i,j = np.unravel_index(smoothed[:,search].argmax(),smoothed[:,search].shape)
                        peak_st = st[search[j]]
                        ax.axvline(.25,color='#ffbf48',ls=':',lw=.8,zorder=4)
                    else:
                        peak = order[0]
                        i,j = rows[peak],cols[peak]
                        peak_st = freq[j]
                    peak_eta = (edges[i]+edges[i+1])/2
                    if peak_st<=upper:
                        ax.plot(peak_st,peak_eta,marker='*',ms=8,mfc='#ffbf48',mec='.15',mew=.65,zorder=5)
                    ax.axvline(.2,color='white',ls='--',lw=.8,zorder=4)
                    annotations[key] = {'peak_St':float(peak_st),'peak_span':float(peak_eta),
                                        'outlined_energy_fraction':float(mass[selected].sum()/mass.sum())}
                label = r'AZDES $\alpha=4.0^\circ$' if key=='azdes' else rf'PSP $\alpha={POLARS[key]:.1f}^\circ$'
                ax.set(title=label,xlim=(.1,upper),ylim=(.45,1))
            fig.colorbar(mesh,ax=list(axes.flat),extend='both',label='Segment modal energy / reference eigenvalue')
            if args.mark_concentration:
                fig.legend(handles=[
                    Line2D([],[],color='.35',ls='--',label=r'$St=0.20$'),
                    Line2D([],[],color='.35',label=r'90\% energy boundary (white in panels)'),
                    Line2D([],[],marker='*',ls='',mfc='#ffbf48',mec='.15',label=r'Smoothed peak for $St\geq0.25$' if args.buffered_peak else 'Peak above cutoff')],
                    loc='outside lower center',ncol=3,frameon=False,fontsize=8)
                for ax in axes[-1]:
                    ax.set_xlabel(r'$St$')
            elif args.span_integrated_peak:
                fig.legend(handles=[
                    Line2D([],[],color='C3',label='Span-integrated energy (right axis)'),
                    Line2D([],[],marker='*',ls='',mfc='#ffbf48',mec='.15',
                           label=r'Smoothed local peak for $St\geq0.25$')],
                    loc='outside lower center',ncol=2,frameon=False,fontsize=8)
                for ax in axes[-1]:
                    ax.set_xlabel(r'$St$')
            else:
                fig.supxlabel(r'$St$')
            fig.supylabel(r'Measured span $\eta$')
            fig.suptitle('Full-chord leading SPOD modal energy')
            stem = destination/f'psp_azdes_full_chord_modal_energy_01_{tag}_span45'
            if args.mark_concentration:
                stem = stem.with_name(stem.name+('_marked90_buffered_peak' if args.buffered_peak else '_marked90'))
            if args.span_integrated_peak:
                stem = stem.with_name(stem.name+'_span_integrated_overlay')
            if args.muted:
                stem = stem.with_name(stem.name+'_muted')
            export(fig,stem)
            plt.close(fig)
            stem.with_suffix('.json').write_text(json.dumps({
                'inputs':{key:cases[key][3] for key in CASES},'reference_eigenvalues':peaks,
                'reference':'Peak leading eigenvalue within St 0.2 to 0.7. Shared PSP 4.0 reference for PSP and separate AZDES reference.',
                'method':'Leading eigenvalue times the area-weighted squared mode norm within each full-chord span segment. Original PSP retained points and native numerical surface points.',
                'peak_search':(
                    {'minimum_St':.25,'maximum_St':.7,'integration_span':[float(edges[0]),float(edges[-1])],
                     'local_sigma_St':.01,'local_sigma_span_bins':1}
                    if args.span_integrated_peak else
                    {'minimum_St':.25,'sigma_St':.01,'maximum_St':.7} if args.buffered_peak else None
                ),
                'energy_annotations':annotations,'span_edges':edges.tolist(),'color_limits':[float(lo),float(hi)],'colormap':cmap.name
            },indent=2)+'\n')


if __name__=='__main__':
    main()
