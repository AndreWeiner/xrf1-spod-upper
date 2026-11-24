"""Cross-frequency similarity of SPOD mode branches 1/2 or 2/2."""
from psp_settings import selected_settings, selected_result
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.analysis import mode_similarity
from flowtorch.visualization import plot_mode_similarity
from convert_surface import ROOT, configured_path, sha256
from plot_paper_mode_similarity import CASES
from reference_scales import strouhal
from plot_style import STYLE, WIDTH_MM, export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode-pair',choices=['12','22'],default='12')
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output=configured_path('OUTPUT_DIR')
    second_self=args.mode_pair=='22'
    pair=[2,2] if second_self else [1,2]
    working=output/'analysis/working/mode_branch_similarity';working.mkdir(parents=True,exist_ok=True)
    destination=args.output_dir or output/('exploratory/figures/mode_similarity' if second_self else 'paper/figures');destination.mkdir(parents=True,exist_ok=True)
    results={};records=[]
    for key,label,angle in CASES:
        path=(output/'analysis/working/spod_rank_sensitivity/cp/rank_200/adaptive_0.0001' if key=='azdes' else
              selected_result(output, key))
        run=json.loads((path/'run.json').read_text())
        reference=json.loads((output/'analysis/working/mode_similarity'/f'self_{key}.json').read_text())
        assert run['status']=='complete' and run['rank']==200
        assert run['tolerance']==(1e-4 if key=='azdes' else selected_settings()[1])
        assert sha256(path/'spectra.npz')==run['spectra_sha256']==reference['first_source']['spectra_sha256']
        with np.load(path/'spectra.npz') as arrays:
            st=strouhal(arrays['frequency'],'numerical' if key=='azdes' else 'psp',None if key=='azdes' else key)
            first=torch.from_numpy(arrays['reduced_modes'][:,:,pair[0]-1].copy().T)
            second=torch.from_numpy(arrays['reduced_modes'][:,:,1].copy().T)
            available_first=arrays['n_tapers']>=pair[0];available_second=arrays['n_tapers']>1
            eigenvalues=arrays['eigvals'].copy()
        assert first.dtype==second.dtype==torch.complex128
        # Weighted POD coordinates preserve the physical area-weighted product.
        similarity=mode_similarity(first,second).numpy()
        similarity[~available_first,:]=np.nan;similarity[:,~available_second]=np.nan
        diagonal=similarity.diagonal();valid=np.isfinite(diagonal)
        orthogonality_error=float(abs(diagonal[valid]-(1 if second_self else 0)).max())
        assert orthogonality_error<1e-10,orthogonality_error
        # Independent direct complex normalized dot products for sampled entries.
        indices=np.unique(np.linspace(0,len(st)-1,9,dtype=int))
        a=first.numpy()[:,indices];b=second.numpy()[:,indices]
        expected=abs(a.conj().T@b)/np.outer(np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=0))
        actual=similarity[np.ix_(indices,indices)];valid=np.isfinite(actual)
        np.testing.assert_allclose(actual[valid],expected[valid],atol=1e-12,rtol=0)
        if second_self:
            np.testing.assert_allclose(similarity,similarity.T,atol=1e-12,rtol=0,equal_nan=True)
        matrix_path=working/f'{key}_phi{pair[0]}_phi2.npz'
        np.savez_compressed(matrix_path,St=st,similarity=similarity,
                            lambda1=eigenvalues[:,0],lambda2=eigenvalues[:,1])
        results[key]=(st,similarity)
        records.append({'case':key,'angle_deg':angle,'rank':200,'tolerance':run['tolerance'],
                        'spectra_sha256':run['spectra_sha256'],'same_frequency_unit_diagonal_max_error' if second_self else 'same_frequency_orthogonality_max_error':orthogonality_error,
                        'unavailable_second_mode_bins':int((~available_second).sum()),
                        'matrix_sha256':sha256(matrix_path)})
    import matplotlib.pyplot as plt
    for upper,suffix in [(.7,''),(.35,'_zoom_01_035')]:
        with plt.rc_context(dict(STYLE, **{'text.usetex':not args.no_tex})):
            fig,axes=plt.subplots(2,3,sharex=True,sharey=True,
                                 figsize=(WIDTH_MM['double']/25.4,120/25.4),layout='constrained')
            for panel,(ax,(key,label,angle)) in enumerate(zip(axes.flat,CASES)):
                st,matrix=results[key];keep=(st>=.1)&(st<=upper)
                coordinate=st[keep].copy();values=matrix[np.ix_(keep,keep)].copy()
                plot_mode_similarity(torch.from_numpy(coordinate),torch.from_numpy(coordinate),
                                     torch.from_numpy(values),triangle='lower' if second_self else 'full',ax=ax,show_colorbar=False,
                                     mesh_kwargs={'rasterized':True,'vmin':0.,'vmax':1.})
                displayed=ax.collections[0].get_array()
                if second_self:
                    hidden=np.triu(np.ones_like(values,dtype=bool),k=0)|~np.isfinite(values.T)
                    ax.collections[0].set_array(np.ma.array(values.T,mask=hidden))
                    np.testing.assert_allclose(ax.collections[0].get_array().compressed(),values.T[~hidden],rtol=0,atol=0)
                else:
                    np.testing.assert_allclose(displayed.filled(np.nan),values.T,equal_nan=True,rtol=0,atol=0)
                ax.set(xlim=(.1,upper),ylim=(.1,upper),xlabel='',ylabel='',aspect='equal')
                ticks=[.1,.3,.5,.7] if upper==.7 else [.1,.15,.2,.25,.3,.35]
                ax.set_xticks(ticks);ax.set_yticks(ticks)
                ax.set_title(rf'({chr(97+panel)}) {label}, $\alpha={angle:.1f}^\circ$',fontsize=9)
            fig.supxlabel(r'$St_1$' if second_self else r'$St_1\;(\phi_1)$',fontsize=10)
            fig.supylabel(r'$St_2$' if second_self else r'$St_2\;(\phi_2)$',fontsize=10)
            fig.colorbar(axes.flat[0].collections[0],ax=list(axes.flat),label=r'$\rho_{22}$' if second_self else r'$\rho_{12}$',shrink=.85,ticks=[0,.2,.4,.6,.8,1])
            stem=destination/(('spod_mode_second_self_similarity' if second_self else 'spod_mode_branch_similarity')+suffix);export(fig,stem);plt.close(fig)
            report={'status':'pass','mode_pair':pair,'rank':200,'inputs':records,'size_mm':[180,120],
                    'strouhal_limits':[.1,upper],'color_scale':'linear 0 to 1',
                    'definition':f'rho(i,j)=abs(phi{pair[0]}(St_i)^H W phi2(St_j))/(normW(phi{pair[0]})*normW(phi2)); not squared',
                    'display':'lower triangle excluding unit diagonal; second mode at both frequencies; no smoothing' if second_self else 'full nonsymmetric matrix; horizontal branch1 frequency, vertical branch2 frequency; no smoothing; zero diagonal retained',
                    'validation':'Sampled products independently checked; same-frequency expected diagonal checked; plotted values match stored matrix',
                    'script_sha256':sha256(Path(__file__)),'use_tex':not args.no_tex}
            stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
            print(stem.with_suffix('.pdf'),flush=True)


    if second_self:
        with np.load(output/'analysis/working/mode_similarity/self_0258.npz') as arrays:
            first_st=arrays['first_St'].copy();first_self=arrays['similarity'].copy()
        st,second_matrix=results['0258']
        np.testing.assert_array_equal(first_st,st)
        keep=(st>=.1)&(st<=.35);coordinate=st[keep].copy()
        with plt.rc_context(dict(STYLE, **{'text.usetex':not args.no_tex})):
            fig,axes=plt.subplots(1,2,sharex=True,sharey=True,
                                 figsize=(WIDTH_MM['double']/25.4,85/25.4),layout='constrained')
            for ax,matrix,mode in zip(axes,[first_self,second_matrix],[1,2]):
                values=matrix[np.ix_(keep,keep)].copy()
                plot_mode_similarity(torch.from_numpy(coordinate),torch.from_numpy(coordinate),
                    torch.from_numpy(values),triangle='lower',ax=ax,show_colorbar=False,
                    mesh_kwargs={'rasterized':True,'vmin':0.,'vmax':1.})
                hidden=np.triu(np.ones_like(values,dtype=bool))|~np.isfinite(values.T)
                ax.collections[0].set_array(np.ma.array(values.T,mask=hidden))
                ax.set(xlim=(.1,.35),ylim=(.1,.35),aspect='equal',xlabel='',ylabel='')
                ax.set_xticks([.1,.15,.2,.25,.3,.35]);ax.set_yticks([.1,.15,.2,.25,.3,.35])
                ax.set_title(rf'({chr(96+mode)}) PSP, $\alpha=4.0^\circ$, mode {mode}')
            fig.supxlabel(r'$St_1$');fig.supylabel(r'$St_2$')
            fig.colorbar(axes[0].collections[0],ax=list(axes),label=r'$\rho$',shrink=.85)
            stem=destination/'psp4_mode_self_similarity_1_vs_2'
            export(fig,stem);plt.close(fig)
            stem.with_suffix('.json').write_text(json.dumps({
                'mode_pairs':[[1,1],[2,2]],'strouhal_limits':[.1,.35],
                'color_limits':[0,1],'rank':200,'psp_tolerance':selected_settings()[1],
                'smoothing':'none','diagonal':'hidden','inputs':records,
                'first_mode_matrix_sha256':sha256(output/'analysis/working/mode_similarity/self_0258.npz'),
                'script_sha256':sha256(Path(__file__))},indent=2)+'\n')
            print(stem.with_suffix('.pdf'),flush=True)


if __name__=='__main__':main()
