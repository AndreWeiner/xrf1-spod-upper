"""Central SI reference scales and Float64 Strouhal coordinates."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/reference_scales.json'


def reference(dataset,polar=None):
    config=json.loads(CONFIG.read_text())
    block=config[dataset]
    chord=float(block['mean_aerodynamic_chord_m'])
    if dataset=='psp':
        if polar is None:raise ValueError('A PSP polar ID is required; no common-velocity fallback')
        velocity=float(block['inflow_velocity_m_per_s_by_polar'][str(polar)])
    else:velocity=float(block['inflow_velocity_m_per_s'])
    if not np.isfinite(chord) or not np.isfinite(velocity) or min(chord,velocity)<=0:
        raise ValueError('Reference chord and velocity must be finite and positive')
    return chord,velocity


def frequency_hz(frequency,dataset):
    values=np.asarray(frequency,dtype=np.float64)
    if dataset=='numerical':
        unit=float(json.loads(CONFIG.read_text())['numerical']['seconds_per_source_time_unit'])
        if not np.isfinite(unit) or unit<=0:raise ValueError('Invalid source-time conversion')
        values=values/unit
    return values


def strouhal(frequency,dataset,polar=None):
    chord,velocity=reference(dataset,polar)
    return frequency_hz(frequency,dataset)*(chord/velocity)


def numerical_zoom_max():
    return float(json.loads(CONFIG.read_text())['plots']['numerical_peak_zoom_strouhal_max'])
