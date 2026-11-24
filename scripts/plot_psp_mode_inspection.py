"""Compare physical leading PSP SPOD modes using native FlowTorch mode plots."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from scipy.ndimage import uniform_filter
from flowtorch.visualization import plot_spod_modes_2d
from convert_surface import ROOT, configured_path, sha256
from mode_similarity_study import read_case, prefix_basis, POLARS
from psp_settings import selected_settings
from plot_style import STYLE, export

CONFIG=ROOT/'config/psp_mode_inspection.json'


def prepare(output,working,config):
    records=[];fields=[];masks=[];coordinates=None
    for polar in config['polars']:
        case=read_case(output,polar);prefix_basis(case,working)
        st=case['St'];eligible=np.flatnonzero((st>=min(config['target_strouhal']))&(st<=max(config['target_strouhal'])))
        ids=np.array([eligible[np.argmin(abs(st[eligible]-target))] for target in config['target_strouhal']])
        assert len(np.unique(ids))==len(ids)
        physical=(case['basis'].to(torch.complex128)@case['coeff'][:,ids]).numpy()
        weight=case['weight'].numpy();np.testing.assert_allclose((weight[:,None]*abs(physical)**2).sum(0),1,atol=1e-8,rtol=0)
        source=output/'psp/working'/f'polar_{polar}'/'fields.npz'
        with np.load(source) as a:grid=a['coordinates'].copy()/1000
        if coordinates is None:coordinates=grid
        else:np.testing.assert_array_equal(coordinates,grid)
        shape=grid.shape[:2];keep=np.zeros(np.prod(shape),dtype=bool);keep[case['indices']]=True;keep=keep.reshape(shape)
        np.testing.assert_allclose(grid.reshape(-1,3)[case['indices']],case['coordinates']/1000,atol=1e-12,rtol=0)
        peak=abs(physical).max(0);assert (peak>0).all();normalized=physical/peak
        for column in range(len(ids)):
            field=np.zeros(shape,dtype=np.complex128);field.ravel()[case['indices']]=normalized[:,column]
            fields.append(field);masks.append(keep)
        p=working/f'polar_{polar}.npz'
        np.savez_compressed(p,physical_modes=physical,peak_magnitude=peak,native_frequency_indices=ids,
            target_St=config['target_strouhal'],actual_St=st[ids],retained_flat_indices=case['indices'],coordinates_m=grid,area_weights=weight)
        records.append({'polar':polar,'angle_deg':POLARS[polar],'source':case['source'],
            'fields_sha256':sha256(source),'cache_sha256':sha256(p),'actual_St':st[ids].tolist(),
            'target_St':config['target_strouhal'],'maximum_St_mismatch':float(abs(st[ids]-config['target_strouhal']).max())})
    # Select a single common 4x4 reference patch with a resolvable mean in every mode.
    common=np.logical_and.reduce(masks);score=np.full(common.shape,np.inf)
    for field in fields:
        patch_mean=uniform_filter(field.real,size=4)+1j*uniform_filter(field.imag,size=4)
        score=np.minimum(score,abs(patch_mean))
    patch_valid=uniform_filter(common.astype(float),size=4)>=1-1e-12
    patch_valid[:3]=False;patch_valid[-3:]=False;patch_valid[:,:3]=False;patch_valid[:,-3:]=False
    score[~patch_valid]=-np.inf
    index=tuple(int(i) for i in np.unravel_index(np.argmax(score),score.shape));assert score[index]>1e-8
    report={'status':'complete','rank':selected_settings()[0],'tolerance':selected_settings()[1],
        'mode_number':1,'inputs':records,'reference_index':index,'reference_xyz_m':coordinates[index].tolist(),
        'reference_selection':'common fully retained 4x4 patch maximizing the minimum normalized complex patch-mean magnitude across all modes',
        'minimum_reference_patch_mean_magnitude':float(score[index]),'normalization':'each physical mode divided by its own maximum magnitude; eigenvalues not included',
        'geometry':'original PSP planform X,-Y in metres, upward is outboard; no spatial interpolation or extrapolation',
        'config_sha256':sha256(CONFIG),'script_sha256':sha256(Path(__file__))}
    (working/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return fields,masks,coordinates,report


def load_prepared(working,config):
    report=json.loads((working/'preparation.json').read_text());assert report['config_sha256']==sha256(CONFIG)
    assert (report['rank'],report['tolerance'])==selected_settings()
    fields=[];masks=[];coordinates=None
    for record in report['inputs']:
        p=working/f"polar_{record['polar']}.npz";assert sha256(p)==record['cache_sha256']
        with np.load(p) as a:
            coordinates=a['coordinates_m'].copy();shape=coordinates.shape[:2];indices=a['retained_flat_indices'];m=a['physical_modes']/a['peak_magnitude']
            keep=np.zeros(np.prod(shape),bool);keep[indices]=True;keep=keep.reshape(shape)
            for column in range(m.shape[1]):
                f=np.zeros(shape,np.complex128);f.ravel()[indices]=m[:,column];fields.append(f);masks.append(keep)
    return fields,masks,coordinates,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quantity',choices=['absolute','real','phase'],action='append');p.add_argument('--plot-only',action='store_true');p.add_argument('--no-tex',action='store_true');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();load_dotenv(ROOT/'.env');torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    output=configured_path('OUTPUT_DIR');config=json.loads(CONFIG.read_text());working=output/'psp/working/mode_inspection';working.mkdir(parents=True,exist_ok=True)
    destination=args.output_dir or output/'psp/exploratory/figures/mode_inspection';destination.mkdir(parents=True,exist_ok=True)
    fields,masks,coordinates,report=load_prepared(working,config) if args.plot_only else prepare(output,working,config)
    i,j=report['reference_index'];aligned=[f*np.exp(-1j*np.angle(f[i-2:i+2,j-2:j+2].sum())) for f in fields]
    for f,g in zip(fields,aligned):np.testing.assert_allclose(abs(f),abs(g),atol=1e-14,rtol=0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=len(config['polars']);cols=len(config['target_strouhal']);titles=[rf'$St={target:.2f}$' for _ in range(rows) for target in config['target_strouhal']]
    for quantity in (args.quantity or ['absolute','real','phase']):
        with plt.rc_context({**STYLE,'text.usetex':not args.no_tex,'axes.titlesize':9}):
            fig,axes=plot_spod_modes_2d([torch.from_numpy(f.ravel()) for f in fields],titles,rows,cols,
                x=torch.from_numpy(coordinates[...,0].copy()),y=torch.from_numpy(-coordinates[...,1].copy()),
                show_real=quantity=='real',show_imaginary=False,show_absolute=quantity=='absolute',show_phase=quantity=='phase',
                reference_index=tuple(report['reference_index']),figsize=tuple(v/25.4 for v in config['size_mm']),
                plot_kwargs={quantity:{'rasterized':True}},colorbar_kwargs={'shrink':.65})
            for index,ax in enumerate(axes[0].flat):
                row,col=divmod(index,cols);mesh=ax.collections[0]
                expected={'absolute':abs(aligned[index]),'real':aligned[index].real,'phase':np.angle(aligned[index])}[quantity]
                if quantity=='phase':expected=np.angle(aligned[index])
                np.testing.assert_allclose(mesh.get_array(),expected,rtol=0,atol=1e-12)
                mask=~masks[index]
                if quantity=='phase':mask=mask|(abs(aligned[index])<config['phase_display_minimum_relative_magnitude'])
                mesh.set_array(np.ma.array(expected,mask=mask))
                if quantity=='real':mesh.set_clim(-1,1)
                if quantity=='absolute':mesh.set_clim(0,1)
                if row>0:ax.set_title('')
                if col==0:ax.text(-.05,.5,rf'$\alpha={POLARS[config["polars"][row]]:.1f}^\circ$',transform=ax.transAxes,rotation=90,ha='right',va='center',fontsize=10)
            fig.suptitle('Leading PSP mode: '+{'absolute':'relative magnitude','real':'phase-aligned real part','phase':r'phase (magnitude $\geq 0.05$ of peak)'}[quantity]+'; outboard upward',fontsize=10)
            colorbar=fig.axes[-1];colorbar.set_ylabel({'absolute':r'$|\phi_1|/\max|\phi_1|$','real':r'$\mathrm{Re}(\widetilde{\phi}_1)/\max|\phi_1|$','phase':r'$\arg(\widetilde{\phi}_1)$ [rad]'}[quantity])
            stem=destination/f'psp_leading_modes_{quantity}';export(fig,stem);plt.close(fig)
            stem.with_suffix('.json').write_text(json.dumps(report|{'quantity':quantity,'size_mm':config['size_mm'],'use_tex':not args.no_tex,
                'phase_visibility_threshold':config['phase_display_minimum_relative_magnitude'] if quantity=='phase' else None,
                'flowtorch_plot_sha256':sha256(Path(inspect.getfile(inspect.unwrap(plot_spod_modes_2d)))),'plot_script_sha256':sha256(Path(__file__))},indent=2)+'\n')
            print(stem.with_suffix('.pdf'),flush=True)

if __name__=='__main__':main()
