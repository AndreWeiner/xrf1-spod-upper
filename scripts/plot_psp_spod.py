"""Plot the saved PSP rank/adaptive-tolerance comparison on Strouhal axes."""
import argparse
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path,sha256
from reference_scales import strouhal,CONFIG as REFERENCE_CONFIG
from plot_style import STYLE,WIDTH_MM,export

CONFIG=ROOT/'config/psp_spod_study.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--polar',help='PSP polar ID');parser.add_argument('--no-tex',action='store_true');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env');config=json.loads(CONFIG.read_text());polar=args.polar or config['polar']
    if polar not in config['polars']:raise ValueError('Polar is not included in the study configuration')
    angle=config['polars'][polar]
    base=configured_path('OUTPUT_DIR')/'psp';working=base/'working/spod'/f'polar_{polar}'
    destination=args.output_dir or base/'exploratory/figures/spod'/f'polar_{polar}';destination.mkdir(parents=True,exist_ok=True)
    records={};rows=[]
    for rank in config['ranks']:
        for tol in config['adaptive_tolerances']:
            path=working/f'rank_{rank}'/f'adaptive_{tol:g}';run=json.loads((path/'run.json').read_text())
            if run['status']!='complete' or run['polar']!=polar or run['angle_deg']!=angle:raise ValueError('Incomplete comparison')
            assert run['rank']==rank and run['tolerance']==tol and run['requested_max_tapers']==200 and run['nfft']==3639
            assert sha256(path/'spectra.npz')==run['spectra_sha256']
            with np.load(path/'spectra.npz') as arrays:data={k:arrays[k].copy() for k in arrays.files}
            data['st']=strouhal(data['frequency'],'psp',polar);records[rank,tol]=(run,data)
            positive=np.flatnonzero(data['st']>0);peak=positive[np.argmax(data['eigvals'][positive,0])]
            rows.append({'rank':rank,'tolerance':tol,'max_tapers':run['observed_max_tapers'],
                'cap_hit_percent':100*run['cap_hit_fraction'],'tolerance_met_percent':100*run['tolerance_met_fraction'],
                'peak_strouhal':float(data['st'][peak]),'peak_leading_eigenvalue':float(data['eigvals'][peak,0]),
                'tapers_at_peak':int(data['n_tapers'][peak]),'residual_at_peak':float(data['final_residual'][peak]),
                'seconds':run['seconds']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#009E73'];lines=['-','--','-.']
    labels=[r'$\varepsilon=10^{-4}$',r'$\varepsilon=5\times10^{-5}$',r'$\varepsilon=10^{-5}$']
    style={**STYLE,'text.usetex':config['use_tex'] and not args.no_tex}
    with plt.rc_context(style):
        fig,axes=plt.subplots(3,len(config['ranks']),squeeze=False,figsize=(WIDTH_MM['double']/25.4,160/25.4),layout='constrained',sharex=True,sharey='row')
        for column,rank in enumerate(config['ranks']):
            for tol,color,line,label in zip(config['adaptive_tolerances'],colors,lines,labels):
                _,data=records[rank,tol];valid=data['st']>0;st=data['st'][valid]
                axes[0,column].loglog(st,data['eigvals'][valid,0],color=color,ls=line,label=label)
                axes[1,column].semilogx(st,data['n_tapers'][valid],color=color,ls=line)
                residual=data['final_residual'][valid];finite=np.isfinite(residual)&(residual>0)
                axes[2,column].loglog(st[finite],residual[finite],color=color,ls=line)
                axes[2,column].axhline(tol,color=color,ls=':',lw=.7,alpha=.6)
            axes[0,column].set_title(f'POD rank {rank}');axes[0,column].legend(fontsize=7,loc='lower left')
            axes[1,column].set_ylim(0,205)
            axes[2,column].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
        for ax in axes.flat:ax.grid(alpha=.2)
        axes[0,0].set_ylabel('Leading SPOD eigenvalue');axes[1,0].set_ylabel('Number of tapers');axes[2,0].set_ylabel('Final leading-mode residual')
        fig.suptitle(rf'PSP polar {polar}, $\alpha={angle:g}^\circ$: adaptive tapering',fontsize=11)
        export(fig,destination/'adaptive_rank_comparison');plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(WIDTH_MM['double']/25.4,150/25.4),layout='constrained',sharex=True)
        rank_diagnostics=[]
        rank_styles={200:':',400:'-',800:'--'}
        pairs=list(zip(config['ranks'][:-1],config['ranks'][1:]))
        for tol,color,label in zip(config['adaptive_tolerances'],colors,labels):
            for rank in config['ranks']:
                _,data=records[rank,tol];valid=data['st']>0
                axes[0].loglog(data['st'][valid],data['eigvals'][valid,0],color=color,ls=rank_styles[rank],label=f'{label}, rank {rank}')
            for (lower,higher),line in zip(pairs,[':', '--']):
                _,a=records[lower,tol];_,b=records[higher,tol];valid=a['st']>0
                assert np.array_equal(a['st'],b['st'])
                relative=100*(b['eigvals'][:,0]/a['eigvals'][:,0]-1)
                axes[1].semilogx(a['st'][valid],relative[valid],color=color,ls=line,label=f'{label}, {higher}/{lower}')
                first=a['reduced_modes'][:,:,0];second=b['reduced_modes'][:,:,0]
                overlap=np.abs(np.sum(first.conj()*second[:,:lower],axis=1))**2
                assert np.isfinite(overlap).all() and overlap.min()>=0 and overlap.max()<=1+1e-7
                axes[2].semilogx(a['st'][valid],overlap[valid],color=color,ls=line)
                peak=1+np.argmax(a['eigvals'][1:,0])
                diagnostic={'tolerance':tol,'lower_rank':lower,'higher_rank':higher,
                    'lower_rank_peak_strouhal':float(a['st'][peak]),
                    'eigenvalue_change_at_lower_rank_peak_percent':float(relative[peak]),
                    'leading_mode_overlap_at_lower_rank_peak':float(overlap[peak]),
                    'leading_eigenvalue_relative_change_percent_median':float(np.median(relative[valid])),
                    'leading_eigenvalue_relative_change_percent_range':[float(relative[valid].min()),float(relative[valid].max())],
                    'leading_mode_overlap_median':float(np.median(overlap[valid]))}
                if lower==400:
                    diagnostic.update(rank400_peak_strouhal=diagnostic['lower_rank_peak_strouhal'],
                        eigenvalue_change_at_rank400_peak_percent=diagnostic['eigenvalue_change_at_lower_rank_peak_percent'],
                        leading_mode_overlap_at_rank400_peak=diagnostic['leading_mode_overlap_at_lower_rank_peak'])
                rank_diagnostics.append(diagnostic)
        axes[0].set_ylabel('Leading SPOD eigenvalue');axes[0].legend(ncol=3,fontsize=7)
        pair_handles,pair_labels=axes[1].get_legend_handles_labels()
        fig.legend(pair_handles,pair_labels,loc='outside lower center',ncol=3,fontsize=7,
            title='Rank pairs (higher/lower)',title_fontsize=8,frameon=False)
        axes[1].set_yscale('symlog',linthresh=10);axes[1].set_ylabel('Eigenvalue change [percent]\n(symmetric log scale)');axes[2].set_ylabel('Leading-mode overlap');axes[2].set_ylim(0,1.05);axes[2].set_xlabel(r'$St=f c_{\mathrm{MAC}}/U_\infty$')
        for ax in axes:ax.grid(alpha=.2)
        export(fig,destination/'rank_sensitivity');plt.close(fig)
    lines=[f'# PSP polar {polar} ({angle:g} degrees): rank and adaptive taper comparison','',
        'All 3,639 snapshots; saved Float64, area-weighted SVD; default automatic mask plus five-pixel crop without Mask 2; snapshot spatial and pointwise temporal means removed. Native PAMSPOD reuses the saved factors. Maximum 200 tapers; leading spectra are not normalized between ranks.','',
        '| Rank | Tolerance | Max tapers | At cap [%] | Tolerance met [%] | Peak St | Tapers at peak | Residual at peak |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["rank"]} | {r["tolerance"]:g} | {r["max_tapers"]} | {r["cap_hit_percent"]:.2f} | {r["tolerance_met_percent"]:.2f} | {r["peak_strouhal"]:.6g} | {r["tapers_at_peak"]} | {r["residual_at_peak"]:.4g} |')
    lines+=['','Rank-sensitivity curves compare adaptive outcomes at matched requested tolerances; their selected taper counts can differ. Leading-mode overlap is the squared absolute inner product in the common area-orthonormal POD basis, with the lower-rank coefficients padded by zeros; adjacent-rank comparisons are shown. Native neighboring-frequency stopping may terminate selection before the requested tolerance is met. No cap hits alone does not demonstrate tolerance convergence. Residual reference lines in the diagnostic figure mark the requested tolerances.','']
    (working/'comparison.md').write_text('\n'.join(lines))
    (working/'comparison.json').write_text(json.dumps({'polar':polar,'angle_deg':angle,'rows':rows,'rank_diagnostics':rank_diagnostics,'config_sha256':sha256(CONFIG),'reference_scales_sha256':sha256(REFERENCE_CONFIG)},indent=2)+'\n')
    print(f'Plotted and reviewed all {len(rows)} variants')

if __name__=='__main__':main()
