"""Plot mean signed along-shock speed over the outer half of the measured span."""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dotenv import load_dotenv
from convert_surface import ROOT, configured_path
from mode_similarity_study import POLARS
from plot_style import STYLE, export
from reference_scales import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--smooth-trends',action='store_true')
    parser.add_argument('--linear-fits',action='store_true')
    parser.add_argument('--window-means',action='store_true')
    parser.add_argument('--fits-only',action='store_true')
    parser.add_argument('--angle-summary',action='store_true')
    args = parser.parse_args()
    if args.angle_summary and not (args.normalize and args.linear_fits and not any((args.window_means,args.smooth_trends,args.fits_only))):
        parser.error('--angle-summary requires --normalize --linear-fits without other plot variants')
    tag = '_uinf' if args.normalize else ''
    if args.smooth_trends:
        tag += '_smoothed_trends'
    if args.fits_only:
        tag += '_fits_only'
    elif args.linear_fits:
        tag += '_linear_fits'
    if args.window_means:
        tag += '_windows005'
    if args.angle_summary:
        tag += '_angle_summary_bands'
    load_dotenv(ROOT/'.env')
    base = configured_path('OUTPUT_DIR')/'psp'
    working = base/'working/displacement_directional'
    suffix = '_no_sign_filter_no_amplitude_filter_slope_only_segments48_stmax0.7_minslope4'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        summaries = []
        if args.angle_summary:
            fig, (ax, summary) = plt.subplots(1,2,figsize=(240/25.4,105/25.4),
                                              width_ratios=[1.55,1],layout='constrained')
        else:
            fig, ax = plt.subplots(figsize=(180/25.4,95/25.4),layout='constrained')
        for index, polar in enumerate(['0257','0258','0259','0260','0261','azdes']):
            with np.load(working/f'polar_{polar}_surface_band_speed{suffix}.npz') as data:
                edges = data['span_edges']
                freq = data['St'] > .25
                span = edges[:-1] >= .5-1e-12
                st = data['St'][freq]
                speed = data['speed'][np.ix_(span,freq)]
            assert span.sum()==40
            width = np.diff(edges)[span,None]
            valid = np.isfinite(speed)
            resolved_width = (width*valid).sum(axis=0)
            mean = np.divide((width*np.where(valid,speed,0)).sum(axis=0),resolved_width,
                             out=np.full(len(st),np.nan),where=resolved_width>0)
            label = r'AZDES $4.0^\circ$' if polar=='azdes' else rf'PSP ${POLARS[polar]:.1f}^\circ$'
            _, inflow = reference("numerical") if polar=="azdes" else reference("psp",polar)
            plotted = mean/inflow if args.normalize else mean
            color = 'C1' if polar=='azdes' else f'C{index}'
            linestyle = '--' if polar=='azdes' else '-'
            diagnostics = {}
            if args.window_means:
                edges = np.arange(25,71,5)/100
                means = np.full(len(edges)-1,np.nan)
                counts = np.zeros(len(means),dtype=int)
                for j,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
                    keep = (st>=left)&(st<=right if j==len(means)-1 else st<right)&np.isfinite(plotted)
                    counts[j] = keep.sum()
                    if counts[j]:
                        means[j] = plotted[keep].mean()
                assert counts.sum()==np.isfinite(plotted).sum()
                centers = (edges[:-1]+edges[1:])/2
                ax.plot(st,plotted,color=color,ls=linestyle,lw=.65,alpha=.25)
                ax.plot(centers,means,label=label,color=color,ls=linestyle,lw=1.5,marker='o',ms=2.5)
                diagnostics = dict(window_edges_St=edges,window_centers_St=centers,
                                   window_means=means,window_counts=counts)
            elif args.smooth_trends:
                valid_mean = np.isfinite(plotted)
                sigma = .008 / np.median(np.diff(st))
                weights = gaussian_filter1d(valid_mean.astype(float),sigma,mode='nearest')
                filtered = gaussian_filter1d(np.where(valid_mean,plotted,0),sigma,mode='nearest')
                smoothed = np.divide(filtered,weights,out=np.full_like(filtered,np.nan),where=valid_mean)
                slope, intercept = np.polyfit(st[valid_mean],plotted[valid_mean],1)
                trend = slope*st+intercept
                ax.plot(st,plotted,color=color,ls=linestyle,lw=.65,alpha=.28)
                ax.plot(st,smoothed,label=label,color=color,ls=linestyle,lw=1.4)
                ax.plot(st,trend,color=color,ls=':',lw=1.15,alpha=.8)
                diagnostics = dict(smoothed=smoothed,linear_trend=trend,
                                   trend_slope_per_St=slope,smoothing_sigma_St=.008)
                print(polar,'trend slope',slope)
            elif args.linear_fits or args.fits_only:
                valid_mean = np.isfinite(plotted)
                slope, intercept = np.polyfit(st[valid_mean],plotted[valid_mean],1)
                trend = slope*st+intercept
                if not args.fits_only:
                    ax.plot(st,plotted,color=color,ls='-',lw=.75,alpha=.35)
                ax.plot(st,trend,label=label,color=color,ls='--' if args.linear_fits else linestyle,lw=1.6,
                        marker='s' if args.linear_fits and polar=='azdes' else None,markevery=8,ms=2.8)
                midpoint_speed = slope*.475+intercept
                diagnostics = dict(linear_trend=trend,trend_slope_per_St=slope,
                                   fitted_speed_at_St0475=midpoint_speed)
                if args.angle_summary:
                    angle = 4.0 if polar=='azdes' else POLARS[polar]
                    segment_fits = []
                    for segment in speed:
                        available = np.isfinite(segment)
                        if available.mean() >= .75:
                            gradient, offset = np.polyfit(st[available]-.475,segment[available]/inflow,1)
                            segment_fits.append((offset,gradient))
                    bounds = np.quantile(segment_fits,[.1,.9],axis=0)
                    summaries.append((angle,midpoint_speed,slope,*bounds.ravel()))
                    diagnostics.update(segment_fit_bounds_p10_p90=bounds,
                                       segment_fit_count=len(segment_fits),
                                       segment_fit_minimum_bin_fraction=.75)
                    print(polar,f'fit at St=0.475: {midpoint_speed:.3f}, slope: {slope:+.3f}')
            else:
                ax.plot(st,plotted,label=label,color=color,ls=linestyle,lw=1.4)
            np.savez_compressed(working/f'polar_{polar}_mean_band_speed_span50_st025_070_minslope4{tag}.npz',
                                St=st,mean_signed_speed_m_per_s=mean,mean_speed_over_uinf=mean/inflow,resolved_span_fraction=resolved_width/.5,**diagnostics)
            print(polar,len(st),np.nanmin(mean),np.nanmax(mean),np.min(resolved_width/.5))
        ax.set(xlim=(.25,.7),xlabel=r'$St$',ylabel=r'$\langle c_s\rangle/U_\infty$' if args.normalize else 'Mean along-shock phase speed [m/s]')
        ax.axhline(0,color='.5',ls=':',lw=.7)
        ax.grid(ls=':',alpha=.4)
        ax.legend(ncol=2,fontsize=8)
        if args.angle_summary:
            angle,speed,slope,speed_lo,slope_lo,speed_hi,slope_hi = np.array(summaries).T
            right = summary.twinx()
            speed_color, slope_color = 'C0','C3'
            for axis,lower,upper,color in [(summary,speed_lo,speed_hi,speed_color),
                                           (right,slope_lo,slope_hi,slope_color)]:
                axis.fill_between(angle[:-1],lower[:-1],upper[:-1],color=color,alpha=.15)
                axis.vlines(angle[-1],lower[-1],upper[-1],color=color,lw=1.2,alpha=.7)
            handles = [
                summary.plot(angle[:-1],speed[:-1],'o-',color=speed_color,label='PSP speed')[0],
                right.plot(angle[:-1],slope[:-1],'^-',color=slope_color,label='PSP slope')[0],
                summary.plot(angle[-1],speed[-1],'o',color=speed_color,mfc='white',mew=1.3,label='AZDES speed')[0],
                right.plot(angle[-1],slope[-1],'^',color=slope_color,mfc='white',mew=1.3,label='AZDES slope')[0],
            ]
            summary.set(xlabel=r'Angle of attack $\alpha$ [deg]',xticks=angle[:-1],
                        ylabel=r'Fitted $\langle c_s\rangle/U_\infty$ at $St=0.475$',
                        title='Speed and slope with spanwise spread')
            right.set_ylabel(r'Fit slope $\mathrm{d}(\langle c_s\rangle/U_\infty)/\mathrm{d}St$',color=slope_color)
            right.tick_params(axis='y',labelcolor=slope_color)
            summary.yaxis.label.set_color(speed_color)
            summary.tick_params(axis='y',labelcolor=speed_color)
            summary.grid(ls=':',alpha=.4)
            summary.legend(handles=handles,ncol=2,fontsize=7,loc='upper center')
            summary.margins(y=.28)
            right.margins(y=.28)
            ax.set_title('Native bins and linear fits')
        if args.smooth_trends:
            fig.supxlabel(r'Faint: native bins. Thick: Gaussian smoothing. Dotted: linear fit.',fontsize=8)
        if args.linear_fits:
            caption = 'Faint lines show native bins. Dashed lines show linear fits.'
            if args.angle_summary:
                caption += ' Bands and vertical bars show segment percentiles (10th–90th).'
            fig.supxlabel(caption,fontsize=8)
        if args.window_means:
            fig.supxlabel(r'Faint lines show native bins. Markers show means in $\Delta St=0.05$ windows.',fontsize=8)
        if args.fits_only:
            fig.suptitle(r'Linear fits of mean along-shock phase speed')
        else:
            fig.suptitle(r'Along-shock phase speed averaged over $0.50\leq\eta\leq1.00$')
        export(fig,base/'exploratory/figures/displacement_directional'/f'psp_azdes_mean_band_speed_span50_st025_070_minslope4{tag}')
        plt.close(fig)


if __name__=='__main__':
    main()
