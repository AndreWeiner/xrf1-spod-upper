"""Six-case appendix figure of leading SPOD self-similarity, lower triangles."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.visualization import plot_mode_similarity
from convert_surface import ROOT, configured_path, sha256
from psp_settings import selected_settings
from plot_style import STYLE, WIDTH_MM, export

CASES = [('0257', 'PSP', 3.5), ('0258', 'PSP', 4), ('azdes', 'AZDES', 4),
         ('0259', 'PSP', 4.5), ('0260', 'PSP', 5), ('0261', 'PSP', 6)]


def draw(output, destination, upper, no_tex=False, levels=None, cmap=None):
    working = output/'analysis/working/mode_similarity'
    destination.mkdir(parents=True, exist_ok=True)
    inputs = []
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    boundaries=np.linspace(0,1,levels+1) if levels else None
    with plt.rc_context(dict(STYLE, **{'text.usetex':not no_tex})):
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,
                                 figsize=(WIDTH_MM['double']/25.4,120/25.4), layout='constrained')
        for index, (ax, (key, label, angle)) in enumerate(zip(axes.flat, CASES)):
            path = working/f'self_{key}.npz'
            record = json.loads(path.with_suffix('.json').read_text())
            assert record['status']=='complete' and record['first']==record['second']==key
            assert record['first_source']['spod_rank']==record['second_source']['spod_rank']==200
            assert record['first_source']['adaptive_tolerance']==(1e-4 if key=='azdes' else selected_settings()[1])
            assert sha256(path)==record['sha256']
            with np.load(path) as arrays:
                first, second = arrays['first_St'], arrays['second_St']
                matrix = arrays['similarity']
            np.testing.assert_array_equal(first, second)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
            np.testing.assert_allclose(matrix.diagonal(), 1, atol=1e-12, rtol=0)
            # Restrict displayed coordinates before rendering, without interpolation.
            keep = (first>=.1)&(first<=upper)
            coordinate = first[keep].copy()
            selected = matrix[np.ix_(keep,keep)].copy()
            plot_mode_similarity(torch.from_numpy(coordinate),torch.from_numpy(coordinate),
                                 torch.from_numpy(selected),triangle='lower',ax=ax,
                                 show_colorbar=False,cmap=cmap or 'viridis',
                                 mesh_kwargs={'rasterized':True,'vmin':0.,'vmax':1.})
            if levels:
                ax.collections[0].set_cmap(plt.get_cmap(cmap or 'viridis',levels))
                ax.collections[0].set_norm(BoundaryNorm(boundaries,levels))
            displayed = ax.collections[0].get_array()
            expected = selected.T
            invalid = np.triu(np.ones_like(expected,dtype=bool),k=1)
            np.testing.assert_array_equal(np.ma.getmaskarray(displayed),invalid)
            np.testing.assert_allclose(displayed.data[~invalid],expected[~invalid],rtol=0,atol=0)
            # Hide the trivial unit diagonal without changing any saved values.
            hidden = invalid | np.eye(len(coordinate),dtype=bool)
            ax.collections[0].set_array(np.ma.array(expected,mask=hidden))
            np.testing.assert_array_equal(np.ma.getmaskarray(ax.collections[0].get_array()),hidden)
            ax.set(xlim=(.1,upper),ylim=(.1,upper),aspect='equal',xlabel='',ylabel='')
            ticks=[.1,.3,.5,.7] if upper==.7 else [v for v in [.1,.15,.2,.25,.3,.35] if v<=upper]
            ax.set_xticks(ticks);ax.set_yticks(ticks)
            ax.grid(alpha=.4, linestyle=':', linewidth=.8)
            ax.set_title(rf'({chr(97+index)}) {label}, $\alpha={angle:.1f}^\circ$',fontsize=9)
            inputs.append({'case':key,'angle_deg':angle,'source_sha256':record['sha256'],
                           'record_sha256':sha256(path.with_suffix('.json')),
                           'displayed_frequency_bins':int(keep.sum()),
                           'symmetry_max_error':float(np.max(abs(matrix-matrix.T))),
                           'diagonal_max_error':float(np.max(abs(matrix.diagonal()-1)))})
        fig.supxlabel(r'$St_1$ [-]',fontsize=STYLE['axes.labelsize'])
        fig.supylabel(r'$St_2$ [-]',fontsize=STYLE['axes.labelsize'])
        fig.colorbar(axes.flat[0].collections[0],ax=list(axes.flat),label=r'$\rho$ [-]',shrink=.85,
                     ticks=boundaries if levels else [0,.2,.4,.6,.8,1])
        name='spod_mode_self_similarity' if upper==.7 else ('spod_mode_self_similarity_zoom_01_03' if upper==.3 else 'spod_mode_self_similarity_zoom_01_035')
        if levels:name+=f'_discrete_{levels}'
        stem=destination/name
        export(fig,stem)
        plt.close(fig)
    report = {'status':'pass','size_mm':[180,120],'inputs':inputs,
              'definition':'FlowTorch absolute normalized area-weighted inner product, not squared; leading mode against itself across frequencies',
              'display':'one triangular half excluding unit diagonal; no smoothing, interpolation or eigenvalue scaling; dotted grid at major ticks',
              'strouhal_limits':[.1,upper],'color_limits':[0,1],'colormap':cmap or 'viridis','color_normalization':'discrete uniform bins' if levels else 'linear','color_levels':levels,'color_boundaries':boundaries.tolist() if levels else None,
              'validation':'Input file hashes, matrix symmetry and unit diagonals checked; native plotted mask and every displayed value verified against saved matrices',
              'use_tex':not no_tex,'script_sha256':sha256(Path(__file__)),
              'flowtorch_plot_sha256':sha256(Path(inspect.getfile(inspect.unwrap(plot_mode_similarity))))}
    stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(stem.with_suffix('.pdf'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=int,help='Number of discrete color bands; default continuous')
    parser.add_argument('--cmap',default=None,help='Matplotlib colormap name; default viridis')
    parser.add_argument('--zoom-upper',type=float,choices=[.3,.35],default=.35)
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    if args.levels is not None and args.levels<2:parser.error('--levels must be at least 2')
    output=configured_path('OUTPUT_DIR')
    destination=args.output_dir or output/'paper/figures'
    for upper in [.7,args.zoom_upper]:
        draw(output,destination,upper,args.no_tex,args.levels,args.cmap)


if __name__=='__main__':
    main()
