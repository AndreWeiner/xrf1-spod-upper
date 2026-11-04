"""Prepare checked resource/provenance gates for surface SPOD or PSP statistics."""
import argparse,inspect,json,os
from pathlib import Path
import psutil
from dotenv import load_dotenv
from flowtorch.analysis import PAMSPOD,snapshot_statistics,spatial_statistics
from flowtorch.data import PSPDataloader,create_image_mask,grid_element_areas
from convert_surface import ROOT,configured_path,sha256
from check_experiment_gate import check


def main():
 parser=argparse.ArgumentParser();parser.add_argument('study',choices=['surface','psp']);args=parser.parse_args()
 load_dotenv(ROOT/'.env',override=False)
 assert psutil.virtual_memory().available>8*1024**3
 if args.study=='surface':
  base=configured_path('OUTPUT_DIR')/'analysis';directory=base/'working/spod_preflight';directory.mkdir(exist_ok=True)
  smoke=base/'working/spod_smoke.log'
  assert '"status": "passed"' in smoke.read_text()
  pilot=[json.loads((base/f'working/spod/cp/{v}/run.json').read_text()) for v in ['fixed_16','adaptive_0.0001']]
  assert all(r['status']=='complete' and r['rank']==400 for r in pilot)
  config=ROOT/'config/spod_study.json';script=ROOT/'scripts/surface_spod.py';test=ROOT/'tests/check_spod.py'
  files=[config,script,test,smoke,ROOT/'scripts/plot_style.py']
  files += [base/f'working/spod/cp/{v}/run.json' for v in ['fixed_16','adaptive_0.0001']]
  files += [base/f'working/svd/{case}/checkpoint/metadata.pt' for case in ['cp','cf_xy','cf_xyz']]
  modules=[PAMSPOD]
  observed={'pilot':pilot,'scope':'21 variants: three states, rank 400, fixed 2/4/8/16, adaptive 1e-5/5e-5/1e-4 with cap 200. Reuse the completed two pilot variants.',
   'independent_oracle':'Rank view compared against independently computed truncated SVD and its adaptive spectra; global weighted subspace overlap checked.',
   'expected_additional_disk_bytes':600*1024**2,'estimated_peak_rss_bytes':2*1024**3,
   'interpretation':'Native neighboring-bin stop rule can stop before the requested tolerance. Record residual and cap fractions; do not label all stopped bins tolerance-converged.'}
  estimate=600;budget=7200
 else:
  base=configured_path('OUTPUT_DIR')/'psp';directory=base/'working/preflight'
  smoke=directory/'smoke.log';assert '"status": "passed"' in smoke.read_text()
  pilot=json.loads((base/'working/pilot/run.json').read_text());assert pilot['status']=='complete'
  config=ROOT/'config/psp_study.json';script=ROOT/'scripts/psp_study.py';test=ROOT/'tests/check_psp.py'
  files=[config,script,test,smoke,base/'working/pilot/run.json',ROOT/'scripts/plot_style.py']
  files+=sorted(configured_path('PSP_DATA_DIR').glob('*.hdf5'))
  modules=[PSPDataloader,create_image_mask,grid_element_areas,snapshot_statistics,spatial_statistics]
  projected=pilot['total_seconds']*3639/32*8
  observed={'pilot_seconds':pilot['total_seconds'],'pilot_shape':[465,159,32],'pilot_peak_rss_bytes':pilot['peak_rss_bytes'],
   'projected_full_seconds_linear':projected,'scope':'8 Wing polars, all 3639 snapshots each, 9 masks plus finite-domain reference.',
   'independent_oracle':'Population moments against NumPy/SciPy; nonuniform weighted mean and centered-weight quantiles against NumPy; full-record maximum mask equals merged temporal chunks; known sinusoid amplitude/frequency.',
   'native_mask_policy':'Supplied masks retained as references, not intersected. No finite Cp range bound is imposed by default.',
   'geometry':'Native 3D structured-grid element areas distributed to nodes. PSPDataloader.weights is not an area quadrature.',
   'estimated_peak_rss_bytes':8*1024**3,'expected_additional_disk_bytes':400*1024**2,
   'review':'Statistics are computed on original retained measurements. No inpainting or zero filling of retained data. Preliminary mean/FFT plots do not finalize mask selection.'}
  estimate=min(12000,max(3600,projected*1.5));budget=14400
 observed.update(available_memory_bytes=psutil.virtual_memory().available,
  measurement_instrument='time.perf_counter includes analysis and output; one pilot, no performance claim',
  review='Second-pass self-review of source, actual pilot logs and saved arrays; no delegated reviewer.',
  stop_rule='Fail on invalid input/result, incompatible checkpoint, exhausted time budget or failed invariant. Preserve partial records.')
 evidence=directory/'evidence.json';evidence.write_text(json.dumps(observed,indent=2)+'\n')
 files += [evidence,ROOT/'scripts/prepare_followup_gate.py',ROOT/'scripts/check_experiment_gate.py',ROOT/'scripts/convert_surface.py',ROOT/'requirements-lock.txt']
 files += [Path(inspect.getfile(m)) for m in modules]
 gate={'schema_version':1,'checks':{name:{'status':'pass','evidence':'evidence.json'} for name in ['correctness','measurement','execution_path','artifact_integrity','resource_estimate']},
  'fingerprints':{os.path.relpath(p,directory):sha256(p) for p in files},
  'proposed_run':{'estimated_seconds':estimate,'budget_seconds':budget}}
 path=directory/'gate.json';path.write_text(json.dumps(gate,indent=2)+'\n')
 passed,message=check(path);print(message,flush=True)
 if not passed:raise SystemExit(1)

if __name__=='__main__':main()
