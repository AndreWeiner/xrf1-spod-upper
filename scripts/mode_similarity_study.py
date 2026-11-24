"""Load the selected SPOD modes and compute their area-weighted self-similarities."""
from psp_settings import selected_settings, selected_result
import json

import numpy as np
import torch
from dotenv import load_dotenv
from flowtorch.analysis import mode_similarity
from convert_surface import ROOT, configured_path, sha256
from reference_scales import strouhal

POLARS = {'0255': 1.5, '0256': 3., '0257': 3.5, '0258': 4., '0259': 4.5,
          '0260': 5., '0261': 6., '0262': 4.}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def read_case(output, key):
    numerical = key == 'azdes'
    rank, tol = (200, 1e-4) if numerical else selected_settings()
    p = (output/'analysis/working/spod_rank_sensitivity/cp/rank_200/adaptive_0.0001' if numerical else
         selected_result(output, key))
    svd = (output/'analysis/working/svd/cp' if numerical else
           output/'psp/working/svd'/f'polar_{key}')
    run = json.loads((p/'run.json').read_text())
    source = json.loads((svd/'run.json').read_text())
    assert run['status'] == source['status'] == 'complete'
    assert run['rank'] == rank and run['tolerance'] == tol and run['adaptive']
    assert sha256(p/'spectra.npz') == run['spectra_sha256']
    with np.load(p/'spectra.npz') as a:
        coefficients = torch.from_numpy(a['reduced_modes'][:, :, 0].copy().T)
        frequency = a['frequency'].copy()
        assert coefficients.dtype == torch.complex128 and coefficients.shape == (rank, len(frequency))
        leading_eigenvalues = a['eigvals'][:, 0].copy()
        assert np.all(a['n_tapers'] >= 1) and np.isfinite(leading_eigenvalues).all()
    case = {'key':key, 'rank':rank, 'coeff':coefficients, 'svd':svd,
            'leading_eigenvalues':leading_eigenvalues,
            'St':strouhal(frequency, 'numerical' if numerical else 'psp', None if numerical else key),
            'label':'AZDES 4 deg' if numerical else f'PSP {POLARS[key]:g} deg'+(' repeat' if key=='0262' else ''),
            'source':{'spod_rank':rank,'adaptive_tolerance':tol,'spectra_sha256':run['spectra_sha256'], 'svd_run_sha256':sha256(svd/'run.json'),
                      'checkpoint_metadata_sha256':sha256(svd/'checkpoint/metadata.pt')}}
    assert case['source']['checkpoint_metadata_sha256'] == (run['svd_metadata_sha256'] if numerical else source['checkpoint_metadata_sha256'])
    if not numerical:
        assert run['subtract_snapshot_spatial_mean'] and run['subtract_pointwise_temporal_mean']
        assert run['domain']['automatic_mask']=='p5_t10' and run['domain']['crop_pixels_each_edge']==5
        assert run['domain']['native_mask'] is None
        with np.load(svd/'preprocessing.npz') as a:
            case.update(indices=a['retained_flat_indices'].copy(), weight=torch.from_numpy(a['area_weights'].copy()),
                        coordinates=a['coordinates'].copy())
        case['source']['preprocessing_sha256']=sha256(svd/'preprocessing.npz')
    return case


def prefix_basis(case, working):
    """Read only a rank prefix from the native NPY checkpoint shards.

    SVD.load copies the complete full-rank disk basis into owned temporary storage.
    Reading these self-contained physical U shards avoids copying ~2 GB per polar.
    """
    folder = case['svd']/'checkpoint'
    meta = torch.load(folder/'metadata.pt', weights_only=True)
    assert meta['format']=='flowtorch.svd' and meta['kind']=='source'
    assert len(meta['layout']['fields'])==1 and meta['layout']['normalization_factors'] in ({'Cp':1.}, {'cp':1.})
    shard = folder/'rank-0'
    index = torch.load(shard/'index.pt', weights_only=True)
    assert index['start']==0 and index['stop']==meta['rows']
    verification = case['svd']/'verification.json'
    manifest = json.loads(verification.read_text())['manifest_sha256'] if verification.exists() else {}
    basis = np.empty((meta['rows'], case['rank']), dtype=np.float64)
    weights = np.empty(meta['rows'], dtype=np.float64)
    fingerprints = {}
    stop = 0
    for left, right, names in index['chunks']:
        assert left == stop
        for name in ['U','weight']:
            file = shard/names[name]
            digest = sha256(file)
            relative = str(file.relative_to(case['svd']))
            if relative in manifest:
                assert digest == manifest[relative]
            fingerprints[relative] = digest
        array = np.load(shard/names['U'], mmap_mode='r')
        assert array.dtype==np.float64 and array.shape[0]==right-left
        basis[left:right] = array[:, :case['rank']]
        weights[left:right] = np.load(shard/names['weight']).reshape(-1)
        stop = right
    assert stop==meta['rows'] and np.isfinite(basis).all() and (weights>0).all()
    if 'weight' in case:
        np.testing.assert_array_equal(weights, case['weight'].numpy())
    u = torch.from_numpy(basis)
    w = torch.from_numpy(weights)
    gram = u.T @ (w[:, None]*u)
    error = float((gram-torch.eye(case['rank'])).abs().max())
    assert error < 1e-7, error
    case.update(basis=u, weight=w)
    save_json(working/f'basis_{case["key"]}.json', {'input':case['source'], 'physical_U_shards_sha256':fingerprints,
               'rank':case['rank'], 'weighted_orthogonality_max_error':error})
    print('Basis ready',case['key'],u.shape,'orthogonality',error,flush=True)


def checked_similarity(first, second):
    similarity = mode_similarity(first, second)
    assert similarity.dtype==torch.float64 and bool(torch.isfinite(similarity).all())
    assert float(similarity.min()) >= 0 and float(similarity.max()) <= 1
    return similarity


def store_matrix(working, name, a, b, matrix, details):
    path = working/(name+'.npz')
    np.savez_compressed(path, first_St=a['St'], second_St=b['St'], similarity=matrix.numpy())
    report = {'status':'complete','first':a['key'],'second':b['key'],'first_source':a['source'],
              'second_source':b['source'],'definition':'rho=abs(phi^H W psi)/(norm_W(phi)*norm_W(psi)); NOT squared',
              'shape':list(matrix.shape),'details':details,'sha256':sha256(path)}
    save_json(path.with_suffix('.json'),report)
    print('Saved',name,details,flush=True)


def compute(output, working):
    cases = {key:read_case(output,key) for key in ['0257','0258','azdes','0259','0260','0261']}
    for key, case in cases.items():
        matrix = checked_similarity(case['coeff'],case['coeff'])
        diagonal = float((matrix.diag()-1).abs().max())
        symmetry = float((matrix-matrix.T).abs().max())
        assert diagonal<1e-12 and symmetry<1e-12
        store_matrix(working,'self_'+key,case,case,matrix,
                     {'domain':'full retained domain; inherited area-weighted orthonormal POD coordinates',
                      'diagonal_max_error':diagonal,'symmetry_max_error':symmetry})


def main():
    load_dotenv(ROOT/'.env')
    torch.set_num_threads(4)
    output = configured_path('OUTPUT_DIR')
    working = output/'analysis/working/mode_similarity'
    working.mkdir(parents=True, exist_ok=True)
    compute(output, working)


if __name__ == '__main__':
    main()
