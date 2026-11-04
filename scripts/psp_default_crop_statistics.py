"""Extend the default automatic-mask/Mask 2 comparison with a five-pixel crop."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.data import PSPDataloader
from convert_surface import ROOT,configured_path,sha256
from psp_global_statistics import pooled,combined_masks,smoke
from psp_study import save_json


def main():
    load_dotenv(ROOT/'.env');torch.set_num_threads(4);assert smoke()['status']=='pass'
    working=configured_path('OUTPUT_DIR')/'psp/working'
    destination=working/'global_statistics/default_crop5';destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for directory in sorted(working.glob('polar_*')):
        start=time.perf_counter();prior=json.loads((directory/'run.json').read_text())
        target=destination/f'{directory.name}.json'
        source=configured_path('PSP_DATA_DIR')/prior['source_file']
        fingerprints={'source':sha256(source),'masks':sha256(directory/'masks.npz'),
            'fields':sha256(directory/'fields.npz'),'code':sha256(Path(__file__)),
            'pooling_code':sha256(ROOT/'scripts/psp_global_statistics.py')}
        assert fingerprints['source']==prior['source_sha256']
        if target.exists():
            record=json.loads(target.read_text());assert record['fingerprints']==fingerprints
        else:
            with np.load(directory/'masks.npz') as masks,np.load(directory/'fields.npz') as fields:
                auto=masks['p5_t10'];native=fields['native_Mask2']
                mask=combined_masks(auto,native,crop=5)[2][1]
                mask10=combined_masks(auto,native,crop=10)[2][1]
                assert (~mask10|mask).all() and (~mask|auto & native).all()
                expected_mean=fields['mean'][mask].mean()
            loader=PSPDataloader(str(source),dtype=torch.float64)
            try:
                loader.zone=prior['zone'];times=loader.write_times
                assert len(times)==prior['snapshots']==3639
                data=loader.load_snapshot('Cp',times).numpy()
                stats=pooled(data,mask)
                np.testing.assert_allclose(stats['mean'],expected_mean,rtol=1e-12,atol=1e-12)
                assert stats['samples']==int(mask.sum())*3639
                record={'polar':prior['polar'],'angle_deg':prior['angle_of_attack_deg'],
                    'mask':'p5_t10','policy':'automatic_mask2_crop5','pixels':int(mask.sum()),
                    'snapshots':3639,'dtype':'float64','status':'complete','fingerprints':fingerprints,
                    'definition':'Equal-weight pooled pixel-time Cp samples; exact linear-interpolation quantiles; five original-image pixels excluded on every edge',
                    'seconds':time.perf_counter()-start,**stats}
                save_json(target,record)
            finally:loader._file.close()
        rows.append(record)
        print(prior['polar'],*(f'{record[k]:.6f}' for k in ['minimum','maximum','q01','q99']),flush=True)
    assert len(rows)==8
    original=[r for p in sorted((working/'global_statistics').glob('polar_*.json')) if '_pilot' not in p.stem for r in json.loads(p.read_text())['rows'] if r['mask']=='p5_t10']
    lines=['# Default PSP mask and crop comparison','',
        'Automatic mask: patch 5 × 5, threshold 10, full-record maximum temporal reduction. All cases intersect Mask 2. Cropping excludes the stated number of pixels on every edge of the original image. Equal-weight pooled Cp statistics over all 3,639 snapshots per polar; Float64, exact linear-interpolation quantiles.','']
    for policy,title in [('automatic_mask2','Mask 2; no crop'),('automatic_mask2_crop5','Mask 2; 5-pixel crop'),('automatic_mask2_crop10','Mask 2; 10-pixel crop')]:
        selected=rows if policy.endswith('crop5') else [r for r in original if r['policy']==policy]
        assert len(selected)==8
        lines += [f'## {title}','','| Polar | Angle [deg] | Minimum | Maximum | Q1 (1%) | Q99 (99%) |','|---|---:|---:|---:|---:|---:|']
        for r in selected:lines.append(f'| {r["polar"]} | {r["angle_deg"]:g} | '+' | '.join(f'{r[k]:.4f}' for k in ['minimum','maximum','q01','q99'])+' |')
        lines.append('')
    (working/'global_statistics/default_mask_crop_comparison.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
