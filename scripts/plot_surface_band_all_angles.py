"""Plot slope-only surface-band phase speeds over two Strouhal ranges."""
import argparse
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path
from mode_similarity_study import POLARS
from plot_style import STYLE, export
from reference_scales import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-slope',type=float,default=1.)
    parser.add_argument('--swept-speed-ceiling',action='store_true',
                        help='Apply a multiple of the 30-degree swept freestream component after averaging')
    parser.add_argument('--swept-speed-factor',type=float,default=1.5)
    parser.add_argument('--span-min',type=float,default=.4)
    parser.add_argument('--spanwise-component',action='store_true')
    args = parser.parse_args()
    threshold_tag = f'_minslope{args.minimum_slope:g}' if args.minimum_slope != 1 else ''
    load_dotenv(ROOT/'.env')
    base = configured_path('OUTPUT_DIR')/'psp'
    working = base/'working/displacement_directional'
    suffix = '_no_sign_filter_no_amplitude_filter_slope_only_segments48_stmax0.7'
    cases = []
    output_tag = threshold_tag + (f"_sweep{100*args.swept_speed_factor:g}" if args.swept_speed_ceiling else "")
    if args.span_min != .4:
        output_tag += f'_span{100*args.span_min:g}'
    component_tag = '_direct_spanwise' if args.spanwise_component else ''
    output_tag += component_tag
    quantity = 'Spanwise phase velocity' if args.spanwise_component else 'Along-shock phase speed'
    labels = {polar:rf'$\alpha={POLARS[polar]:.1f}^\circ$' for polar in ['0257','0258','0259','0260','0261']}
    labels['azdes'] = r'AZDES $\alpha=4.0^\circ$'
    for polar in ['0257','0258','azdes','0259','0260','0261']:
        with np.load(working/f'polar_{polar}_surface_band_speed{suffix}{threshold_tag}{component_tag}.npz') as data:
            st, edges, speed = (data[key] for key in ['St','span_edges','speed'])
        assert speed.shape == (48,len(st)) and st[0]<.103 and st[-1]>.697
        matches = np.flatnonzero(np.isclose(edges,args.span_min,atol=1e-10,rtol=0))
        if len(matches)!=1 or matches[0]>=len(edges)-1:
            parser.error('--span-min must match a segment edge below 1.0')
        start = matches[0]
        edges, speed = edges[start:], speed[start:]
        chord,inflow = reference('numerical') if polar=='azdes' else reference('psp',polar)
        ceiling = args.swept_speed_factor*inflow*np.sin(np.deg2rad(30))
        above = np.isfinite(speed)&(abs(speed)>ceiling) if args.swept_speed_ceiling else np.zeros_like(speed,dtype=bool)
        cases.append((polar,st,edges,speed,above))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, SymLogNorm
    from matplotlib.patches import Patch
    for maximum,tag in [(.35,'035'),(.70,'070')]:
        with plt.rc_context(STYLE):
            fig,axes = plt.subplots(2,3,sharex=True,sharey=True,layout='constrained',
                                    figsize=(240/25.4,145/25.4))
            for ax,(polar,st,edges,speed,above) in zip(axes.flat,cases):
                keep = st<=maximum
                freq = st[keep]
                freq_edges = np.r_[freq[0]-np.diff(freq)[0]/2,
                                    (freq[1:]+freq[:-1])/2,freq[-1]+np.diff(freq)[-1]/2]
                mesh = ax.pcolormesh(freq_edges,edges,np.where(above[:,keep],np.nan,speed[:,keep]),
                                     cmap=plt.get_cmap('RdBu_r').with_extremes(bad='.85'),
                                     norm=SymLogNorm(20,vmin=-300,vmax=300),rasterized=True)
                if args.swept_speed_ceiling:
                    ax.pcolormesh(freq_edges,edges,np.ma.masked_where(~above[:,keep],above[:,keep]),
                                  cmap=ListedColormap(['#a985bf']),vmin=0,vmax=1,rasterized=True)
                ax.set(title=labels[polar],xlim=(.1,maximum),ylim=(args.span_min,1))
                print(polar, maximum, len(freq), round((np.isfinite(speed[:,keep])&~above[:,keep]).mean(),4))

            bar = fig.colorbar(mesh,ax=list(axes.flat),extend='both',label=f'{quantity} [m/s]')
            bar.set_ticks([-300,-100,-20,0,20,100,300],labels=['-300','-100','-20','0','20','100','300'])
            fig.supxlabel(r'$St$')
            fig.supylabel(r'Measured span $\eta$')
            if args.swept_speed_ceiling:
                fig.suptitle(rf'{quantity} with a post-average ceiling of ${args.swept_speed_factor:g}U_\infty\sin 30^\circ$')
                fig.legend(handles=[Patch(color='#a985bf',label='Above ceiling'),Patch(color='.85',label='Originally unresolved')],
                           loc='outside lower center',ncol=2,frameon=False)
                fig.supxlabel('').set_visible(False)
                for ax in axes[-1]:
                    ax.set_xlabel(r'$St$')
            else:
                fig.suptitle(f'{quantity} with {len(cases[0][2])-1} segments and a {args.minimum_slope:g} rad/m cutoff (gray: unresolved)')
            export(fig,base/'exploratory/figures/displacement_directional'/f'psp_surface_band_speed_all_angles_azdes_01_{tag}{output_tag}')
            plt.close(fig)

        with plt.rc_context(STYLE):
            fig,axes = plt.subplots(2,3,sharex=True,sharey=True,layout='constrained',
                                    figsize=(240/25.4,145/25.4))
            for ax,(polar,st,edges,speed,above) in zip(axes.flat,cases):
                keep = st<=maximum
                freq = st[keep]
                values = speed[:,keep]
                width = np.diff(edges)[:,None]
                retained = ~above[:,keep]
                fractions = np.array([(width*(values>0)*retained).sum(0),
                                      (width*(values<0)*retained).sum(0),
                                      (width*~np.isfinite(values)).sum(0),
                                      (width*above[:,keep]).sum(0)])/width.sum()
                np.testing.assert_allclose(fractions.sum(0),1,atol=1e-12)
                ax.stackplot(freq,100*fractions,colors=['C0','C1','.85','#a985bf'])
                ax.set(title=labels[polar],xlim=(.1,maximum),ylim=(0,100))
                ax.grid(ls=':',alpha=.35)
                if maximum==.70:
                    working_file=working/f'polar_{polar}_speed_direction_fractions{output_tag}.npz'
                    np.savez_compressed(working_file,St=freq,outboard=fractions[0],
                                        inboard=fractions[1],unresolved=fractions[2],above_ceiling=fractions[3])
            legend = [('C0','Outboard'),('C1','Inboard'),('.85','Originally unresolved' if args.swept_speed_ceiling else 'Unresolved')]
            if args.swept_speed_ceiling:
                legend.append(('#a985bf','Above ceiling'))
            fig.legend(handles=[Patch(color=c,label=name) for c,name in legend],
                       loc='outside lower center',ncol=3,frameon=False)
            for ax in axes[-1]:
                ax.set_xlabel(r'$St$')
            fig.supylabel(r'Fraction of evaluated span [\%]')
            fig.suptitle(f'Directional fractions from the {quantity.lower()} maps')
            export(fig,base/'exploratory/figures/displacement_directional'/f'psp_speed_direction_fractions_azdes_01_{tag}{output_tag}')
            plt.close(fig)


if __name__ == '__main__':
    main()
