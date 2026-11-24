"""Animate cached PSP modes with native FlowTorch reconstruction and animation."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.analysis import AMSPOD
from flowtorch.visualization import animate_scalar_field
from convert_surface import ROOT, configured_path, sha256
from plot_psp_mode_inspection import load_prepared, CONFIG as INSPECTION_CONFIG
from plot_style import STYLE
from reference_scales import reference
from psp_settings import selected_result

CONFIG=ROOT/'config/psp_mode_animation.json'


def mode_name(polar, st):
    return f'polar_{polar}_St_{st:.2f}'.replace('.', 'p')


def reconstruct(mode, frequency, n_frames):
    """Read-only adapter for native mode_reconstruction, not a refitted SPOD.

    The repository stores physical modes separately from the live SPOD object.
    Only the cached attributes read by AMSPOD.mode_reconstruction are supplied.
    One extra endpoint verifies periodic closure before omitting the duplicate.
    """
    view=SimpleNamespace(_frequency=torch.tensor([frequency],dtype=torch.float64),
        _adaptive=False,_max_tapers=1,_n_keep=1,_dt=1/(frequency*n_frames),
        _modes=mode.reshape(1,-1,1),_complex=False)
    sequence=AMSPOD.mode_reconstruction(view,0,0,N=n_frames+1)
    phase=torch.arange(n_frames+1,dtype=torch.float64)*2*torch.pi/n_frames
    expected=mode.real[:,None]*torch.cos(phase)-mode.imag[:,None]*torch.sin(phase)
    np.testing.assert_allclose(sequence.numpy(),expected.numpy(),atol=2e-13,rtol=0)
    np.testing.assert_allclose(sequence[:,0].numpy(),sequence[:,-1].numpy(),atol=2e-13,rtol=0)
    return sequence[:,:-1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--polar',action='append');p.add_argument('--limit',type=int);p.add_argument('--no-tex',action='store_true');p.add_argument('--output-dir',type=Path);p.add_argument('--combine-only',action='store_true')
    args=p.parse_args();load_dotenv(ROOT/'.env');torch.set_default_dtype(torch.float64)
    config=json.loads(CONFIG.read_text());torch.set_num_threads(config['threads'])
    inspection=json.loads(INSPECTION_CONFIG.read_text());output=configured_path('OUTPUT_DIR')
    working=output/'psp/working/mode_inspection';fields,masks,coordinates,source=load_prepared(working,inspection)
    destination=args.output_dir or output/'psp/exploratory/animations/mode_inspection';destination.mkdir(parents=True,exist_ok=True)
    i,j=source['reference_index'];n=config['frames_per_period'];fps=config['frames_per_second'];records=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    columns=len(inspection['target_strouhal'])
    for row,polar in enumerate(inspection['polars']):
        if args.polar and polar not in args.polar:continue
        record=source['inputs'][row]
        current=selected_result(output,polar)
        assert sha256(current/'spectra.npz')==record['source']['spectra_sha256']
        chord,velocity=reference('psp',polar)
        for col,target in enumerate(inspection['target_strouhal']):
            if args.limit is not None and col>=args.limit:continue
            index=row*columns+col;stem=destination/mode_name(polar,target);video=stem.with_suffix('.mp4')
            expected={'layout_version':2,'source_mode_sha256':record['cache_sha256'],'target_St':target,'actual_St':record['actual_St'][col],
                'phase_reference_index':source['reference_index'],'frames':n,'fps':fps,'use_tex':not args.no_tex,'animation_config_sha256':sha256(CONFIG),
                'normalization':'same per-mode peak magnitude and common reference patch as static figures',
                'reconstruction':'native AMSPOD.mode_reconstruction on cached physical-mode attributes; Re(phi exp(+2 pi i f t)); no mean field or eigenvalue amplitude',
                'frequency_hz':record['actual_St'][col]*velocity/chord,
                'period_seconds':chord/(record['actual_St'][col]*velocity),
                'display':'one cycle, duplicate endpoint removed; 2-second slow-motion playback; common t/T across panels; retained native PSP geometry X,-Y; outboard upward',
                'mode_reconstruction_sha256':sha256(Path(inspect.getfile(inspect.unwrap(AMSPOD.mode_reconstruction)))),
                'animation_api_sha256':sha256(Path(inspect.getfile(inspect.unwrap(animate_scalar_field))))}
            if video.exists() and stem.with_suffix('.json').exists():
                saved=json.loads(stem.with_suffix('.json').read_text())
                if all(saved.get(k)==v for k,v in expected.items()) and saved.get('video_sha256')==sha256(video):
                    records.append(saved);print('Reusing',video,flush=True);continue
            if args.combine_only:raise ValueError(f'Missing or incompatible animation {video}')
            field=fields[index];aligned=field*np.exp(-1j*np.angle(field[i-2:i+2,j-2:j+2].sum()))
            sequence=reconstruct(torch.from_numpy(aligned.ravel()),expected['frequency_hz'],n).reshape(*field.shape,n).clone()
            sequence[~torch.from_numpy(masks[index])]=float('nan')
            assert sequence.dtype==torch.float64 and torch.isfinite(sequence[torch.from_numpy(masks[index])]).all()
            with plt.rc_context({**STYLE,'text.usetex':not args.no_tex,'axes.titlesize':10}):
                animation=animate_scalar_field(sequence,x=torch.from_numpy(coordinates[...,0].copy()),y=torch.from_numpy(-coordinates[...,1].copy()),
                    interval=round(1000/fps),center_zero=True,vmin=-1,vmax=1,cmap='coolwarm',colorbar=False,
                    figsize=tuple(config['single_size_inches']),layout='compressed',padding=.08,repeat=False,blit=False)
                fig=animation._fig;ax=fig.axes[0]
                ax.set_title(rf'$\alpha={record["angle_deg"]:.1f}^\circ,\quad St={target:.2f}$',pad=5)
                fig.suptitle(r'$t/T:\ 0\rightarrow1$; outboard upward',fontsize=9)
                writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-threads','2','-movflags','+faststart'])
                animation.save(str(video),writer=writer,dpi=config['video_dpi']);plt.close(fig)
            saved=expected|{'status':'complete','polar':polar,'angle_deg':record['angle_deg'],'video_sha256':sha256(video),'script_sha256':sha256(Path(__file__))}
            stem.with_suffix('.json').write_text(json.dumps(saved,indent=2)+'\n');records.append(saved);print(video,flush=True)
    if args.limit:return
    # Combine native-rendered videos; no frame interpolation or additional cycles.
    row_videos=[]
    for polar in inspection['polars']:
        if args.polar and polar not in args.polar:continue
        files=[destination/(mode_name(polar,st)+'.mp4') for st in inspection['target_strouhal']]
        if not all(p.exists() for p in files):continue
        strip=destination/f'polar_{polar}_one_period.mp4'
        command=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1']
        for file in files:command+=['-i',str(file)]
        command+=['-filter_complex',f'hstack=inputs={columns}', '-an','-c:v','libx264','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(strip)]
        subprocess.run(command,check=True);row_videos.append(strip)
        print(strip,flush=True)
    if len(row_videos)==len(inspection['polars']):
        overview=destination/'psp_leading_modes_one_period.mp4';command=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1']
        for file in row_videos:command+=['-i',str(file)]
        command+=['-filter_complex',f'vstack=inputs={len(row_videos)},scale=1800:-2','-an','-c:v','libx264','-crf','19','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(overview)]
        subprocess.run(command,check=True)
        gif=overview.with_suffix('.gif')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1','-i',str(overview),'-filter_complex','fps=24,scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4','-loop','0',str(gif)],check=True)
        print(overview,flush=True);print(gif,flush=True)
    (destination/'manifest.json').write_text(json.dumps({'status':'complete','records':records,'config':config,'inspection_config_sha256':sha256(INSPECTION_CONFIG),'script_sha256':sha256(Path(__file__))},indent=2)+'\n')

if __name__=='__main__':main()
