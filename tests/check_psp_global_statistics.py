"""Cross-check pooled results against independently saved temporal statistics."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path
from psp_global_statistics import smoke,combined_masks

load_dotenv(ROOT/'.env')
base=configured_path('OUTPUT_DIR')/'psp/working'
assert smoke()['status']=='pass'
checked=0
for path in sorted((base/'global_statistics').glob('polar_*.json')):
    if '_pilot' in path.stem:continue
    record=json.loads(path.read_text());directory=base/f'polar_{record["polar"]}'
    assert record['status']=='complete' and record['dtype']=='float64' and record['snapshots']==3639
    with np.load(directory/'fields.npz') as fields,np.load(directory/'masks.npz') as masks:
        for row in record['rows']:
            domains=dict(combined_masks(masks[row['mask']],fields['native_Mask2']))
            domain=domains[row['policy']]
            np.testing.assert_allclose(row['mean'],fields['mean'][domain].mean(),rtol=1e-12,atol=1e-12)
            assert row['samples']==int(domain.sum())*3639
            if row['policy']=='automatic':
                with np.load(directory/f'{row["mask"]}.npz') as previous:
                    assert row['minimum']==float(previous['minimum'].min())
                    assert row['maximum']==float(previous['maximum'].max())
            checked+=1
print(f'PASS: sorted-reference fixture; {checked} global means/counts; automatic-mask extrema vs saved independent reductions')
