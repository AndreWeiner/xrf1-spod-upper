"""Rank-review tables and exploratory plots from completed PSP SVD spectra."""
import argparse
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import torch
from convert_surface import ROOT,configured_path,sha256
from plot_style import STYLE,WIDTH_MM,export


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    base=configured_path('OUTPUT_DIR')/'psp';working=base/'working/svd'
    destination=base/'exploratory/figures/svd';destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for directory in sorted(working.glob('polar_*')):
        record=json.loads((directory/'run.json').read_text())
        if record['status']!='complete':continue
        metadata=torch.load(directory/'checkpoint/metadata.pt',weights_only=True)
        record['svht_rank_diagnostic']=int(metadata['opt_rank'])
        with np.load(directory/'spectrum.npz') as data:
            singular=data['singular_values'][:record['resolved_rank']].copy()
        energy=singular**2;cumulative=np.cumsum(energy)/energy.sum()
        records.append((record,singular,cumulative))
    if not records:raise ValueError('No complete PSP SVDs')
    lines=['# PSP SVD rank review','',
        'Default automatic mask (5 × 5, threshold 10, temporal maximum), five-pixel crop, no Mask 2. Each snapshot’s area-weighted spatial mean is removed, followed by pointwise temporal-mean subtraction inside FlowTorch SVD. Existing FlowTorch geometric nodal areas define the inner product. All 3,639 snapshots per polar; Float64; full retained rank 3,638. Energy fractions refer to the doubly centered fluctuations, not the original Cp field.','',
        '| Polar | Angle [deg] | Points | SVHT diagnostic | Rank 90% | Rank 95% | Rank 99% | Rank 99.5% | Rank 99.9% | Energy at rank 400 [%] | Energy at rank 800 [%] |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for r,s,c in records:
        row={'polar':r['polar'],'angle_deg':r['angle_deg'],'points':r['points'],'svht_rank_diagnostic':r['svht_rank_diagnostic']}
        row.update({f'rank_{p}':int(np.searchsorted(c,p/100)+1) for p in [90,95,99,99.5,99.9]})
        row.update({f'energy_rank_{k}_percent':float(c[k-1]*100) for k in [400,800]})
        rows.append(row)
        lines.append(f'| {r["polar"]} | {r["angle_deg"]:g} | {r["points"]} | {r["svht_rank_diagnostic"]} | '+' | '.join(str(row[f'rank_{p}']) for p in [90,95,99,99.5,99.9])+f' | {c[399]*100:.3f} | {c[799]*100:.3f} |')
    lines += ['','Energy is the sum of squared singular values. A high energy threshold may retain measurement noise as well as coherent fluctuations; these curves alone do not establish an optimal denoising rank. The native FlowTorch singular-value hard-threshold (SVHT) rank is also reported as a noise-filtering diagnostic. Its noise-model suitability has not been established for these spatially weighted measurements; it does not automatically select the analysis rank. No final SPOD rank has been selected.','']
    (working/'rank_review.md').write_text('\n'.join(lines))
    with (working/'rank_review.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#332288','#000000']
    styles=['-','--','-.',':','-','--','-.',':']
    with plt.rc_context({**STYLE,'text.usetex':not args.no_tex}):
        fig,axes=plt.subplots(3,1,figsize=(WIDTH_MM['double']/25.4,160/25.4),sharex=True,layout='constrained')
        for (r,s,c),color,line in zip(records,colors,styles):
            rank=np.arange(1,len(s)+1);label=r['polar']+rf', $\alpha={r["angle_deg"]:g}^\circ$'
            axes[0].semilogy(rank,s/s[0],color=color,ls=line,label=label)
            axes[1].plot(rank,c*100,color=color,ls=line)
            # Exact zero at full rank is omitted from the logarithmic error panel.
            tail=np.concatenate((np.cumsum((s*s)[::-1])[::-1][1:],[0.]))
            error=np.sqrt(tail/np.sum(s*s));valid=error>0
            axes[2].semilogy(rank[valid],error[valid],color=color,ls=line)
        axes[0].set_ylabel(r'$\sigma_j/\sigma_1$');axes[0].set_title('PSP: spatial and temporal means removed')
        axes[0].legend(ncol=4,fontsize=7,loc='upper right')
        axes[1].set_ylabel('Retained energy [percent]');axes[1].set_ylim(0,101)
        for level in [90,95,99]:axes[1].axhline(level,color='.6',lw=.6,ls=':')
        axes[2].set_ylabel('Relative weighted RMS error');axes[2].set_xlabel('Retained POD rank')
        for ax in axes:ax.grid(alpha=.2);ax.set_xlim(1,3638)
        export(fig,destination/'rank_comparison');plt.close(fig)
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,75/25.4),layout='constrained')
        for (r,s,c),color,line in zip(records,colors,styles):
            rank=np.arange(1,len(s)+1);ax.loglog(rank,s/s[0],color=color,ls=line,label=r['polar']+rf', $\alpha={r["angle_deg"]:g}^\circ$')
            diagnostic=r['svht_rank_diagnostic'];ax.plot(diagnostic,s[diagnostic-1]/s[0],marker='o',color=color,ms=4)
        ax.set(xlabel='POD mode index',ylabel=r'$\sigma_j/\sigma_1$',title='PSP singular spectra: dots indicate native SVHT rank')
        ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=7,loc='lower left');export(fig,destination/'singular_spectra_log_rank');plt.close(fig)
    summary={'complete_polars':len(records),'selected_ranks':None,'rows':rows,
        'inputs':{r['polar']:{'run_sha256':sha256(working/f'polar_{r["polar"]}/run.json'),'spectrum_sha256':sha256(working/f'polar_{r["polar"]}/spectrum.npz')} for r,_,_ in records}}
    (working/'rank_review.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Reviewed {len(records)} complete polars; rank remains unselected')

if __name__=='__main__':main()
