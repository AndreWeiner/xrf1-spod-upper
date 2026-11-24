"""Compare surface-gradient and trace-gradient estimates across the shock band."""
import argparse
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path
from mode_similarity_study import POLARS
from plot_band_speed_maps import segment_speed
from plot_style import STYLE, export
from reference_scales import reference


def compute(working,polar,st,span,band,coeff,case,coords,mask,tri,sigma,truncate,*,sign_filter=True,amplitude_filter=True,segments=12,slope_only=False,st_max=.35,minimum_slope=1.,spanwise_component=False):
    import torch
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import LinearNDInterpolator
    from flowtorch.data import curvilinear_gradient, replace_masked_values

    full = np.zeros((*mask.shape,case['rank']))
    full.reshape(-1,case['rank'])[case['indices']] = case['basis'].numpy()
    repaired = torch.from_numpy(full)
    if not mask.all():
        repaired = replace_masked_values(*coords[:2],repaired,torch.from_numpy(mask))
    del full
    basis = torch.from_numpy(gaussian_filter(repaired.numpy(),(sigma,sigma,0),
                                            mode='reflect',truncate=truncate)).to(torch.complex128)
    del repaired
    tangent = np.gradient(band['coordinates'],axis=0,edge_order=2)
    tangent /= np.linalg.norm(tangent,axis=-1,keepdims=True)
    chord,inflow = reference('numerical') if polar=='azdes' else reference('psp',polar)
    omega = 2*np.pi*st*inflow/chord
    results = []
    spanwise = []
    targets = ([ (.15,3),(.20,0)] if sign_filter else [(.20,0),(.200962,1)]) if segments==12 else []
    examples = {int(np.argmin(abs(st-target))):row for target,row in targets} if polar=='0258' else {}
    samples = {}
    for first in range(0,len(st),12):
        field = basis @ coeff[:,first:first+12]
        real = curvilinear_gradient(field.real,*coords,edge_order=2)
        imag = curvilinear_gradient(field.imag,*coords,edge_order=2)
        power = field.abs().square()
        gradient = (field.real[...,None]*imag-field.imag[...,None]*real)/power.clamp_min(1e-300)[...,None]
        channels = torch.cat([field.real[...,None],field.imag[...,None],gradient],axis=-1)
        if spanwise_component:
            # Outboard is -Y in the mapped geometry. Compute the vector
            # component on the surface before interpolating onto the band.
            norm_squared = gradient.square().sum(dim=-1)
            frequency = torch.from_numpy(omega[first:first+field.shape[-1]])
            outboard = frequency*gradient[...,1]/norm_squared.clamp_min(1e-300)
            channels = torch.cat([channels,outboard[...,None]],axis=-1)
        sampled = LinearNDInterpolator(tri,channels.numpy().reshape(len(tri.points),-1))(band['target'])
        sampled = sampled.reshape(*tangent.shape[:2],field.shape[-1],channels.shape[-1])
        modes = sampled[...,0]+1j*sampled[...,1]
        local = np.einsum('snfc,snc->snf',sampled[...,2:5],tangent)
        weight = band['weights'][...,None]
        power = abs(modes)**2
        total = (weight*power).sum(1)
        slope = (weight*power*local).sum(1)/total
        coherence = abs((weight*abs(modes)*modes).sum(1))/total
        agreement = (weight*power*(np.sign(local)==np.sign(slope[:,None,:]))).sum(1)/total
        amplitude = np.sqrt(total/weight.sum(1))
        for index in examples:
            if first <= index < first+field.shape[-1]:
                samples[index] = local[:,:,index-first].copy()
        if spanwise_component:
            spanwise.append((weight*power*sampled[...,5]).sum(1)/total)
        results.append(np.stack([slope,coherence,agreement,amplitude]))
    slope,coherence,agreement,amplitude = np.concatenate(results,axis=2)
    valid = (coherence>=.9)&(abs(slope)>minimum_slope)
    if amplitude_filter:
        valid &= amplitude>=.05*amplitude.max(0)
    if sign_filter:
        valid &= agreement>=.75
    suffix = '' if sign_filter else '_no_sign_filter'
    if not amplitude_filter:
        suffix += '_no_amplitude_filter'
    if slope_only:
        valid = abs(slope)>minimum_slope
        suffix += '_slope_only'
    assert np.isfinite(slope).all()
    edges = np.linspace(.4,1,segments+1)
    if segments != 12:
        suffix += f'_segments{segments}'
    speed,coverage,segment_agreement,mean = segment_speed(span,slope,valid,2*np.pi*st*inflow/chord,edges,
                                                      minimum_agreement=.75 if sign_filter and not slope_only else 0,
                                                      minimum_coverage=0 if slope_only else .5, minimum_slope=minimum_slope)
    old = np.empty((0,len(st)))
    if st_max == .35:
        with np.load(working/f'polar_{polar}_band_segment_speed.npz') as previous:
            np.testing.assert_array_equal(previous['St'],st)
            old = previous['speed_m_per_s']
    else:
        suffix += f'_stmax{st_max:g}'
    if minimum_slope != 1:
        suffix += f'_minslope{minimum_slope:g}'
    if spanwise_component:
        suffix += '_direct_spanwise'
        station_speed = np.concatenate(spanwise,axis=1)
        cells = np.r_[span[0],(span[1:]+span[:-1])/2,span[-1]]
        width = np.maximum(0,np.minimum(cells[None,1:],edges[1:,None])
                           -np.maximum(cells[None,:-1],edges[:-1,None]))
        usable = valid & np.isfinite(station_speed)
        denominator = width @ usable.astype(float)
        projected = np.divide(width @ np.where(usable,station_speed,0),denominator,
                              out=np.full_like(speed,np.nan),where=(denominator>0)&np.isfinite(speed))
        common = np.isfinite(projected)&np.isfinite(speed)
        print(polar,'sign agreement with along-shock speed',np.mean(np.sign(projected[common])==np.sign(speed[common])),
              'median absolute speeds, along and spanwise',np.median(abs(speed[common])),np.median(abs(projected[common])))
        old,speed = speed,projected
    np.savez_compressed(working/f'polar_{polar}_surface_band_speed{suffix}.npz',St=st,span_edges=edges,
                        speed=speed,old=old,mean_slope=mean,coverage=coverage,agreement=segment_agreement)
    if examples:
        plot_samples(working,st,span,edges,examples,samples,slope,coherence,
                     amplitude/amplitude.max(0),valid,coverage,segment_agreement,agreement,suffix,sign_filter,mean,amplitude_filter)
    if old.shape != speed.shape:
        print(polar,'resolved coverage',np.isfinite(speed).mean(),flush=True)
        return
    both = np.isfinite(old)&np.isfinite(speed)
    print(polar,'common direction agreement',np.mean(np.sign(old[both])==np.sign(speed[both])),
          'median speed difference',np.median(abs(speed[both]-old[both])),
          'coverage old and new',np.isfinite(old).mean(),np.isfinite(speed).mean(),flush=True)


def plot_samples(working,st,span,edges,examples,samples,slope,coherence,amplitude,valid,coverage,agreement,station_agreement,suffix,sign_filter,mean,amplitude_filter):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    destination = working.parents[1]/'exploratory/figures/displacement_directional'
    offsets = np.linspace(-4,4,next(iter(samples.values())).shape[1])
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(3,2,figsize=(210/25.4,140/25.4),layout='constrained')
        for col,(j,row) in enumerate(examples.items()):
            left,right = edges[row:row+2]
            use = (span>=left)&(span<=right)
            x = span[use]
            mesh = axes[0,col].pcolormesh(x,offsets,samples[j][use].T,shading='nearest',
                                        cmap='RdBu_r',norm=SymLogNorm(2,vmin=-100,vmax=100))
            axes[0,col].set(title=rf'$St={st[j]:.3f},\quad\eta={left:.2f}\mathrm{{--}}{right:.2f}$',
                           ylabel='Shock-normal offset [mm]')
            axes[1,col].plot(x,slope[use,j],color='C0')
            bad = use & ~valid[:,j]
            axes[1,col].scatter(span[bad],slope[bad,j],marker='x',s=16,color='C1',label='Excluded station')
            axes[1,col].axhspan(-1,1,color='.85')
            axes[1,col].set(ylabel='Band slope [rad/m]')
            axes[1,col].legend(fontsize=8)
            for reason,(values,label) in enumerate([(coherence[use,j]<.9,'Phase coherence'),
                                 ((amplitude[use,j]<.05)&amplitude_filter,'Amplitude'),
                                 ((station_agreement[use,j]<.75)&sign_filter,'Slope sign agreement'),
                                 (abs(slope[use,j])<=1,'Near-zero slope')]):
                axes[2,col].plot(x[values],np.full(values.sum(),reason),'.',ms=5)
            axes[2,col].set_yticks(range(4),['Phase coherence','Amplitude','Slope sign agreement','Near-zero slope'])
            if not sign_filter:
                axes[2,col].set_yticks([0,1,3],['Phase coherence','Amplitude','Near-zero slope'])
            axes[2,col].set_ylim(-.4,3.4)
            axes[2,col].set(xlabel=r'Measured span $\eta$',
                           title=(f'Valid span {coverage[row,j]:.0%}, mean slope {mean[row,j]:.2f} rad/m' if not sign_filter else f'Valid span {coverage[row,j]:.0%}, segment agreement {agreement[row,j]:.0%}').replace('%',r'\%'))
            for ax in axes[:,col]:
                ax.set_xlim(left,right)
                ax.grid(ls=':',alpha=.3)
        fig.colorbar(mesh,ax=list(axes[0]),label='Sampled along-band phase gradient [rad/m]',extend='both')
        export(fig,destination/('psp_0258_unresolved_phase_gradient_examples'+suffix))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-slope-sign-filter',action='store_true')
    parser.add_argument('--no-amplitude-filter',action='store_true')
    parser.add_argument('--span-segments',type=int,default=12)
    parser.add_argument('--slope-only',action='store_true')
    args = parser.parse_args()
    suffix = '_no_sign_filter' if args.no_slope_sign_filter else ''
    if args.no_amplitude_filter:
        suffix += '_no_amplitude_filter'
    reference_suffix = suffix
    if args.span_segments == 48 and not args.slope_only:
        reference_suffix += "_segments24"
    if args.span_segments != 12:
        suffix += f'_segments{args.span_segments}'
    if args.slope_only:
        reference_suffix = suffix
        suffix = suffix.replace(f'_segments{args.span_segments}','')+'_slope_only'
        if args.span_segments != 12:
            suffix += f'_segments{args.span_segments}'
    load_dotenv(ROOT/'.env')
    base = configured_path('OUTPUT_DIR')/'psp'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,2,sharex=True,sharey=True,layout='constrained',
                                figsize=(200/25.4,125/25.4))
        for col,polar in enumerate(['0258','0261']):
            with np.load(base/'working/displacement_directional'/f'polar_{polar}_surface_band_speed{suffix}.npz') as a:
                if args.span_segments != 12:
                    with np.load(base/'working/displacement_directional'/f'polar_{polar}_surface_band_speed{reference_suffix}.npz') as previous:
                        old_speed = previous['speed']
                        old_edges = previous['span_edges']
                    print(polar,np.isfinite(old_speed).mean(),np.isfinite(a['speed']).mean())
                elif args.no_amplitude_filter:
                    reference_suffix = '_no_sign_filter' if args.no_slope_sign_filter else ''
                    with np.load(base/'working/displacement_directional'/f'polar_{polar}_surface_band_speed{reference_suffix}.npz') as previous:
                        old_speed = previous['speed']
                    print(polar,np.isfinite(old_speed).mean(),np.isfinite(a['speed']).mean())
                elif args.no_slope_sign_filter:
                    with np.load(base/'working/displacement_directional'/f'polar_{polar}_surface_band_speed.npz') as previous:
                        old_speed = previous['speed']
                    print(polar,np.isfinite(old_speed).mean(),np.isfinite(a['speed']).mean())
                else:
                    old_speed = a['old']
                if args.span_segments == 12:
                    old_edges = a['span_edges']
                for row,values in enumerate([old_speed,a['speed']]):
                    ax=axes[row,col]
                    edges = old_edges if row==0 else a['span_edges']
                    mesh=ax.pcolormesh(a['St'],(edges[1:]+edges[:-1])/2,values,
                                       shading='nearest',cmap=plt.get_cmap('RdBu_r').with_extremes(bad='.85'),
                                       norm=SymLogNorm(20,vmin=-300,vmax=300),rasterized=True)
                    label=(['With sign filters','Without sign filters'] if args.no_slope_sign_filter else ['Trace gradient','Surface gradient'])[row]
                    if args.no_amplitude_filter:
                        label = ['With amplitude cutoff','Without amplitude cutoff'][row]
                    if args.span_segments != 12:
                        label = f'{len(edges)-1} span segments'
                    if args.slope_only:
                        label = ['Previous filters','Minimum slope only'][row]
                    ax.set(title=rf'$\alpha={POLARS[polar]:.1f}^\circ$ · {label}',xlim=(.1,.35),ylim=(.4,1))
        bar=fig.colorbar(mesh,ax=list(axes.flat),label='Along-shock phase speed [m/s]',extend='both')
        bar.set_ticks([-300,-100,0,100,300],labels=['-300','-100','0','100','300'])
        fig.supxlabel(r'$St$')
        fig.supylabel(r'Measured span $\eta$')
        export(fig,base/'exploratory/figures/displacement_directional'/('psp_surface_band_speed_comparison'+suffix))
        plt.close(fig)


if __name__ == '__main__':
    main()
