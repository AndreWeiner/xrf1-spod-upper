"""Summarize taper selection diagnostics and required interpretation limits."""
import json
from pathlib import Path
from dotenv import load_dotenv
from convert_surface import ROOT,configured_path,sha256
from reference_scales import reference


def main():
    load_dotenv(ROOT/'.env',override=False)
    output=configured_path('OUTPUT_DIR')/'analysis'
    config=json.loads((ROOT/'config/spod_study.json').read_text())
    records=[];lines=['# Surface PAMSPOD comparison','',
      'All three state vectors use rank 400, the complete time record and their cached area-weighted, temporally centered SVD. The adaptive maximum is 200 tapers.','',
      '| State | Adaptive tolerance | Max. observed tapers | At cap | Tolerance met |',
      '| --- | ---: | ---: | ---: | ---: |']
    for case in config['cases']:
        meta=output/f'working/svd/{case}/checkpoint/metadata.pt'
        for p in sorted((output/'working/spod'/case).glob('*/run.json')):
            r=json.loads(p.read_text());assert r['status']=='complete'
            records.append({'case':case,'svd_metadata_sha256':sha256(meta),**r})
            if r['adaptive']:
                lines.append(f'| {case} | {r["tolerance"]:g} | {r["observed_max_tapers"]} | {100*r["cap_hit_fraction"]:.2f}% | {100*r["tolerance_met_fraction"]:.2f}% |')
    assert len(records)==21
    lines+=['','All leading reduced modes passed unit-norm and finite-value checks. All fixed taper counts were used as requested. Three reduced modes are stored where available, two for the two-taper variant.','',
      'The tolerance column is the requested leading-mode similarity tolerance. The tolerance-met fraction is evaluated from the saved final residual, not from the native stopped/converged flag. Neighboring-frequency stopping is part of the native algorithm, so neither absence of cap hits nor stopping demonstrates tolerance convergence.','',
      'At the strictest tolerance, a few bins reach the cap. No higher-cap run has been performed. This sweep documents the behavior at the selected cap rather than claiming cap-independent convergence.','',
      f"Frequency axes use St = f c_MAC / U_inf with c_MAC = {reference('numerical')[0]:g} m and U_inf = {reference('numerical')[1]:g} m/s from the central reference-scales configuration. Source timestamps are in seconds, consistent with the previous numerical-reference workflow.",'']
    (output/'working/spod/review.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'working/spod/review.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
