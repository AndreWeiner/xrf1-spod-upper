"""Map segment-averaged along-shock phase speed over frequency and span."""
import json
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import POLARS
from plot_style import STYLE, export
from reference_scales import reference


def segment_speed(span, slope, valid, omega, edges, *, minimum_agreement=.75, minimum_coverage=.5, minimum_slope=1.):
    speed = np.full((len(edges)-1,len(omega)), np.nan)
    coverage = np.zeros_like(speed)
    agreement = np.zeros_like(speed)
    mean_slope = np.full_like(speed,np.nan)
    # Use the overlap of each native station's span cell with the segment.
    cells = np.r_[span[0],(span[1:]+span[:-1])/2,span[-1]]
    for j,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
        width = np.maximum(0,np.minimum(cells[1:],right)-np.maximum(cells[:-1],left))
        weight = width[:,None]*valid
        total = weight.sum(0)
        coverage[j] = total/(right-left)
        mean = np.divide((weight*np.where(valid,slope,0)).sum(0),total,
                         out=np.full_like(total,np.nan),where=total>0)
        mean_slope[j] = mean
        agreement[j] = np.divide((weight*(np.sign(slope)==np.sign(mean))).sum(0),total,
                                 out=np.zeros_like(total),where=total>0)
        resolved = (coverage[j]>=minimum_coverage)&(agreement[j]>=minimum_agreement)&(abs(mean)>minimum_slope)
        speed[j] = np.divide(-omega,mean,out=np.full_like(mean,np.nan),where=resolved)
    return speed,coverage,agreement,mean_slope


def main():
    load_dotenv(ROOT/'.env')
    base = configured_path('OUTPUT_DIR')/'psp'
    working = base/'working/displacement_directional'
    destination = base/'exploratory/figures/displacement_directional'
    edges = np.linspace(.4,1.,13)
    x = np.linspace(.4,1.,121)
    k = np.broadcast_to([-4.,4.],(len(x),2))
    valid = np.ones_like(k,dtype=bool)
    valid[:,1] = False
    test = segment_speed(x,k,valid,np.array([8.,8.]),edges)[0]
    np.testing.assert_allclose(test[:,0],2.)
    assert np.isnan(test[:,1]).all()

    cases = []
    for polar in ['0257','0258','0259','0260','0261']:
        source = working/f'polar_{polar}_band_phase_fractions.npz'
        with np.load(source) as data:
            st = data['St']
            chord,inflow = reference('psp',polar)
            speed,coverage,agreement,slope = segment_speed(
                data['span_fraction'],data['slope_rad_per_m'],data['valid'],2*np.pi*st*inflow/chord,edges)
        np.savez_compressed(working/f'polar_{polar}_band_segment_speed.npz',
                            St=st,span_edges=edges,speed_m_per_s=speed,valid_span_fraction=coverage,
                            sign_agreement=agreement,mean_slope_rad_per_m=slope)
        cases.append((polar,st,speed,sha256(source)))
        print(polar, 'resolved', round(np.isfinite(speed).mean(),3))
    limit = 50*np.ceil(np.nanquantile(np.concatenate([abs(case[2]).ravel() for case in cases]),.95)/50)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import Patch
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,3,sharex=True,sharey=True,layout='constrained',
                                figsize=(240/25.4,145/25.4))
        cmap = plt.get_cmap('RdBu_r').with_extremes(bad='.85')
        norm = SymLogNorm(linthresh=20,vmin=-limit,vmax=limit)
        for ax,(polar,st,speed,_) in zip(axes.flat,cases):
            freq_edges = np.r_[st[0]-(st[1]-st[0])/2,(st[1:]+st[:-1])/2,st[-1]+(st[-1]-st[-2])/2]
            mesh = ax.pcolormesh(freq_edges,edges,speed,cmap=cmap,norm=norm,rasterized=True)
            ax.set(title=rf'$\alpha={POLARS[polar]:.1f}^\circ$',xlim=(.1,.35),ylim=(.4,1.))
            ax.set_yticks([.4,.5,.6,.7,.8,.9,1.])
        axes.flat[-1].axis('off')
        axes.flat[-1].legend(handles=[Patch(color='.85',label='Unresolved segment')],
                              loc='center',frameon=False)
        fig.colorbar(mesh,ax=list(axes.flat[:5]),extend='both',
                     label='Along-shock phase speed [m/s]',
                     ticks=[v for v in [-200,-100,-50,-20,0,20,50,100,200] if abs(v)<=limit])
        fig.supxlabel(r'$St$')
        fig.supylabel(r'Measured span $\eta$')
        fig.suptitle('Segment-averaged phase speed from the mean-shock band')
        stem = destination/'psp_band_segment_phase_speed'
        export(fig,stem)
        plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({
        'input_sha256':{polar:digest for polar,_,_,digest in cases},
        'span_edges':edges.tolist(),'minimum_accepted_span_fraction':.5,
        'minimum_segment_sign_agreement':.75,'minimum_abs_mean_slope_rad_per_m':1,
        'method':'Span-length weighted mean of accepted cross-band phase slopes, then -omega / mean slope.',
        'color_limit_m_per_s':float(limit),'linear_color_threshold_m_per_s':20
    },indent=2)+'\n')


if __name__ == '__main__':
    main()
