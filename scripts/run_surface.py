"""Run the upper-wing conversion followed by storage verification."""
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from convert_surface import configured_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / '.env', override=False)
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/convert_surface.py'),
                    '--zones', 'wing-upper', '--fields', 'cp', 'cfx', 'cfy', 'cfz'],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/verify_conversion.py'),
                    str(configured_path('OUTPUT_DIR') / 'surface')],
                   check=True, cwd=ROOT)
    print('Conversion and verification completed successfully.', flush=True)


if __name__ == '__main__':
    main()
