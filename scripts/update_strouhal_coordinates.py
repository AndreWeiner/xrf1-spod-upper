"""Add dimensionless coordinate sidecars without modifying cached numerical arrays."""
import ast
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from convert_surface import ROOT,configured_path,sha256
from reference_scales import CONFIG,reference,strouhal,frequency_hz


def main():
    load_dotenv(ROOT/'.env',override=False)
    output=configured_path('OUTPUT_DIR');reference_hash=sha256(CONFIG)
    numerical=[]
    for case in ['cp','cf_xy','cf_xyz']:
        for path in sorted((output/'analysis/working/spod'/case).glob('*/spectra.npz')):
            record=json.loads((path.parent/'run.json').read_text())
            assert sha256(path)==record['spectra_sha256']
            with np.load(path) as data:
                f=data['frequency'];st=strouhal(f,'numerical')
                width=strouhal(data['half_bandwidth'],'numerical')
                np.testing.assert_allclose(st,frequency_hz(f,'numerical')*reference('numerical')[0]/reference('numerical')[1],rtol=1e-14,atol=0)
                np.savez(path.parent/'strouhal_coordinates.npz',frequency_hz=frequency_hz(f,'numerical'),strouhal=st,
                    half_bandwidth_strouhal=width,reference_scales_sha256=reference_hash,
                    source_spectra_sha256=record['spectra_sha256'])
            numerical.append({'case':case,'variant':path.parent.name,'source_sha256':record['spectra_sha256']})
    psp=[]
    for directory in sorted((output/'psp/working').glob('polar_*')):
        record=json.loads((directory/'run.json').read_text());assert record['status']=='complete'
        frequency=None
        for item in record['cases']:
            if item['status']!='complete':continue
            with np.load(directory/f'{item["label"]}.npz') as data:
                if frequency is None:frequency=data['frequency_hz'].copy()
                else:np.testing.assert_array_equal(frequency,data['frequency_hz'])
        st=strouhal(frequency,'psp',record['polar'])
        np.savez(directory/'strouhal_coordinates.npz',frequency_hz=frequency,strouhal=st,
            reference_scales_sha256=reference_hash)
        psp.append({'polar':record['polar'],'reference_chord_m':reference('psp',record['polar'])[0],
            'reference_velocity_m_per_s':reference('psp',record['polar'])[1],
            'strouhal_bin_spacing':float(st[1]-st[0])})
    report={'status':'passed','definition':'St = f_Hz c_MAC/U_inf','reference_scales_sha256':reference_hash,
        'numerical_variants':numerical,'psp_polars':psp,'original_spectra':'Unmodified; numerical spectra hashes checked',
        'ordinate_policy':'SPOD bin eigenvalues and Fourier amplitudes unchanged; not converted into a density per Strouhal.'}
    (output/'strouhal_conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Added Float64 Strouhal coordinates for {len(numerical)} numerical variants and {len(psp)} PSP polars.')

if __name__=='__main__':main()
