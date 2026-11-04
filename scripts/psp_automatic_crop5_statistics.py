"""Global default automatic-mask statistics with crop5 and no supplied Mask 2."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.data import PSPDataloader
from convert_surface import ROOT,configured_path,sha256
from psp_global_statistics import pooled,smoke
from psp_study import save_json


def main():
    load_dotenv(ROOT/'.env');torch.set_num_threads(4);assert smoke()['status']=='pass'
    working=configured_path('OUTPUT_DIR')/'psp/working'
    destination=working/'global_statistics/automatic_crop5';destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for directory in sorted(working.glob('polar_*')):
        started=time.perf_counter();prior=json.loads((directory/'run.json').read_text())
        source=configured_path('PSP_DATA_DIR')/prior['source_file'];target=destination/f'{directory.name}.json'
        fingerprints={'source':sha256(source),'masks':sha256(directory/'masks.npz'),
            'fields':sha256(directory/'fields.npz'),'code':sha256(Path(__file__)),
            'pooling_code':sha256(ROOT/'scripts/psp_global_statistics.py')}
        assert fingerprints['source']==prior['source_sha256']
        if target.exists():
            record=json.loads(target.read_text());assert record['fingerprints']==fingerprints and record['status']=='complete'
        else:
            with np.load(directory/'masks.npz') as masks,np.load(directory/'fields.npz') as fields:
                automatic=masks['p5_t10'];interior=np.zeros_like(automatic)
                interior[5:-5,5:-5]=True
                mask=automatic & interior
                assert not mask[:5].any() and not mask[-5:].any() and not mask[:,:5].any() and not mask[:,-5:].any()
                expected_mean=fields['mean'][mask].mean()
                with_mask2=mask & fields['native_Mask2']
                assert with_mask2.sum()<=mask.sum()<=automatic.sum()
                mask2_prior=json.loads((working/'global_statistics/default_crop5'/f'{directory.name}.json').read_text())
                assert int(with_mask2.sum())==mask2_prior['pixels']
            loader=PSPDataloader(str(source),dtype=torch.float64)
            try:
                loader.zone=prior['zone'];times=loader.write_times
                assert len(times)==prior['snapshots']==3639
                data=loader.load_snapshot('Cp',times).numpy()
                stats=pooled(data,mask)
                np.testing.assert_allclose(stats['mean'],expected_mean,rtol=1e-12,atol=1e-12)
                assert stats['samples']==int(mask.sum())*3639
                assert stats['minimum']<=mask2_prior['minimum'] and stats['maximum']>=mask2_prior['maximum']
                record={'polar':prior['polar'],'angle_deg':prior['angle_of_attack_deg'],
                    'mask':'p5_t10','policy':'automatic_crop5_without_mask2','pixels':int(mask.sum()),
                    'snapshots':3639,'dtype':'float64','status':'complete','fingerprints':fingerprints,
                    'definition':'Equal-weight pooled pixel-time Cp samples; exact linear-interpolation quantiles; default automatic mask intersected only with five-pixel crop on all four original-image edges; no native Mask 2',
                    'seconds':time.perf_counter()-started,**stats}
                save_json(target,record)
                del data
            finally:loader._file.close()
        rows.append(record)
        print(prior['polar'],*(f'{record[k]:.6f}' for k in ['minimum','maximum','q01','q99']),flush=True)
    assert len(rows)==8
    lines=['# Automatic PSP mask plus five-pixel crop, without Mask 2','',
        'Default automatic mask: patch 5 × 5, threshold 10, maximum reduction over all 3,639 snapshots. Five pixels excluded on every original-image edge. No supplied Mask 2. Equal-weight pooled pixel-time Cp statistics, Float64, exact linear-interpolation quantiles.','',
        '| Polar | Angle [deg] | Minimum | Maximum | Q1 (1%) | Q99 (99%) |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["polar"]} | {r["angle_deg"]:g} | '+' | '.join(f'{r[k]:.4f}' for k in ['minimum','maximum','q01','q99'])+' |')
    (working/'global_statistics/automatic_crop5_without_mask2.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
