"""Selected PSP settings shared by final-result readers."""
import json
from convert_surface import ROOT


def selected_settings():
    config = json.loads((ROOT/'config/psp_spod_study.json').read_text())
    return config['selected_rank'], config['selected_tolerance']


def selected_result(output, polar):
    rank, tolerance = selected_settings()
    return output/'psp/working/spod'/f'polar_{polar}'/f'rank_{rank}'/f'adaptive_{tolerance:g}'
