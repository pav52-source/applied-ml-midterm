"""Assemble a new Gradescope midterm folder, refusing to overwrite any destination."""
import argparse
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def assemble(destination):
    destination=Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing destination: {destination}')
    if ROOT == destination or ROOT in destination.parents:
        raise ValueError('Choose a staging destination outside the repository to avoid duplicated source/data in Git.')
    destination.mkdir(parents=True)
    for source in (ROOT/'deliverables/notebooks').glob('*.ipynb'):
        shutil.copy2(source,destination/source.name)
    for folder in ('part1-data','part2-data','part1-results','part2/outputs','part2/calibration_outputs','integration/outputs','reports'):
        shutil.copytree(ROOT/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__','*.aux','*.log','*.out','.DS_Store'))
    # Source modules are supplemental; substantive code already exists in notebooks.
    for source in (ROOT/'part2').glob('*.py'):
        shutil.copy2(source,destination/'part2'/source.name)
    for name in ('__init__.py','reconcile_reasons.py'):
        shutil.copy2(ROOT/'integration'/name,destination/'integration'/name)
    shutil.copy2(ROOT/'deliverables/README.md',destination/'README.md')
    shutil.copy2(ROOT/'deliverables/requirements.txt',destination/'requirements.txt')
    shutil.copy2(ROOT/'PART1_NOTES.md',destination/'PART1_NOTES.md')
    shutil.copy2(ROOT/'Midterm_Project.tex',destination/'Midterm_Project.tex')
    shutil.copy2(ROOT/'integration/provenance.json',destination/'integration/provenance.json')
    print(f'Assembled {destination}; nothing uploaded.')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',help='A new staging path, e.g. /tmp/cs5785-final/midterm')
    assemble(parser.parse_args().destination)
