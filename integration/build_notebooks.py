"""Package authoritative source into final notebooks without model fitting."""
import ast
import copy
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/notebooks'


def md(s): return dict(cell_type='markdown',metadata={},source=s+'\n')
def code(s): return dict(cell_type='code',metadata={},source=s+'\n',outputs=[],execution_count=None)

ROOT_FINDER = '''ROOT = next((p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (p / 'part1-data').is_dir() and (p / 'part2-data').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Run inside the repository or assembled midterm folder with both data directories.')'''
BOOT = '''from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
''' + ROOT_FINDER + '''
RUN_FULL = False
REPRO_ROOT = ROOT / 'deliverables/reproduced'
REPRO_ROOT.mkdir(parents=True, exist_ok=True)
print('Mode:', 'full fresh reproduction' if RUN_FULL else 'read-only artifact verification; no training')
'''


def save(name,cells):
    for i,c in enumerate(cells):
        c['source']=''.join(c['source'])
        c['id']=hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
    (OUT / name).write_text(json.dumps(notebook,indent=1)+'\n')


def source_defs(path,names):
    source=(ROOT/path).read_text(); tree=ast.parse(source)
    return [code(ast.get_source_segment(source,node)) for node in tree.body
            if isinstance(node,ast.FunctionDef) and node.name in names]


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'CS5785_Midterm_Starter.ipynb').read_text())
    cells=[md('# Final Part 1: teammate’s K-Means/K-NN hybrid\n\nAuthoritative source: origin/main at the commit in integration/provenance.json. All original Part 1 implementation cells are copied below. No Michael-only root Part 1 modules are used. Default mode checks saved final predictions; set RUN_FULL=True and restart/run all to reproduce the existing 99-configuration research. Outputs are isolated; no original artifacts are overwritten.'),code(BOOT),code("PROJECT_DIR = ROOT\nWORK = REPRO_ROOT / 'part1_workflow'\nWORK.mkdir(parents=True,exist_ok=True)\n(WORK / 'writeup').mkdir(exist_ok=True)")]
    for i,c in enumerate(original['cells'][1:23],1):
        c=copy.deepcopy(c);s=''.join(c['source'])
        if c['cell_type']=='code':
            if i==2:s='PROJECT_DIR = ROOT\n'+s[s.index('PART1_DIR ='):]
            for path in ('part1-results','writeup','part1_submission.csv'):
                s=s.replace(f'PROJECT_DIR / "{path}"',f'WORK / "{path}"')
            c['source']='if RUN_FULL:\n'+textwrap.indent(s,'    ')+'\nelse:\n    print("Part 1 original cell '+str(i)+' not retrained in verification mode.")\n'
            c['metadata']={'source_cell':i,'source_notebook':'CS5785_Midterm_Starter.ipynb'}
        cells.append(c)
    cells += [md('## Verify the final hybrid and distinguish the earlier submission\n\nThe final submission is the knn-experiment hybrid, not the root K-Means submission. Reported public F1 0.87927 is recorded in PART1_NOTES.md and was not re-queried.'),code('''import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
reasons = ['Safety','Funding','Efficacy','Recruitment','Other']
truth = pd.read_csv(ROOT / 'part1-data/part1_val.csv')
artifacts = WORK / 'part1-results' if RUN_FULL else ROOT / 'part1-results'
validation = pd.read_csv(artifacts / 'knn-experiment/validation_kmeans_knn_hybrid_k3_uniform.csv')
assert validation.ID.equals(truth.ID)
metrics = pd.Series({'sample_f1':f1_score(truth[reasons],validation[reasons],average='samples'),
                     'macro_f1':f1_score(truth[reasons],validation[reasons],average='macro')})
assert np.isclose(metrics.sample_f1,.8867521367521366)
assert np.isclose(metrics.macro_f1,.7792343604108309)
display(metrics.to_frame('validation'))
submission = pd.read_csv(artifacts / 'knn-experiment/best_augmented_candidate_submission.csv')
test = pd.read_csv(ROOT / 'part1-data/part1_test.csv')
assert len(submission)==312 and submission.ID.equals(test.ID)
assert submission.Predicted.notna().all() and submission.Predicted.str.strip().ne('').all()
assert all(set(s.split()) <= set(reasons) for s in submission.Predicted)
display(pd.read_csv(artifacts / 'knn-experiment/results.csv').groupby('method',sort=False).head(1))'''),md('## Findings\n\nInitial cluster augmentation underperformed the supervised control. The equal K-Means/local-neighbor blend reaches validation sample F1 0.88675 and macro F1 0.77923; its 99-setting selection and rare labels limit confidence. PCA is diagnostic only. These results belong to Part 1; no reason embedding or reason prediction enters the Part 2 classifier.')]
    save('part1_final.ipynb',cells)
    for old,new in [('part2_modeling.ipynb','part2_modeling_final.ipynb'),('part2_calibration.ipynb','part2_calibration_final.ipynb')]:
        n=json.loads((ROOT/'part2'/old).read_text());cells=n['cells']
        cells.insert(1,md('## Integrated deliverable provenance\n\nSource: Michael’s origin/michael implementation. Model features, hyperparameters, splits and calibration procedure are unchanged. Part 1 references in preservation hashes are administrative only. For the authoritative cross-part error analysis use part2_error_analysis_final.ipynb. All new output paths below use deliverables/reproduced/.'))
        for c in cells:
            s=''.join(c['source'])
            oldroot="ROOT = Path.cwd().resolve()\nif ROOT.name == 'part2':\n    ROOT = ROOT.parent\nif not all((ROOT / name).is_dir() for name in ('part1-data', 'part2-data')):\n    raise FileNotFoundError('Start the notebook from the repository root or its part2/ directory.')"
            s=s.replace(oldroot,ROOT_FINDER)
            s=s.replace("ROOT / 'part2' / 'notebook_reproduction'","ROOT / 'deliverables/reproduced'")
            s=s.replace('part2/notebook_reproduction/','deliverables/reproduced/')
            c['source']=s
        save(new,cells)
    # Self-contained actual analysis implementations, not module-import wrappers.
    cells=[md('# Final Part 2 stopping-reason analysis with teammate predictions\n\nRecompute the reason-dependent analysis without fitting any model. True labels override predictions. All 131 old unresolved phase rows are resolved as hybrid pseudo-labels; absent Part 1 records remain separate. Multi-label categories overlap. Inputs are the supplied datasets and saved authoritative Part 1/Part 2 predictions. The complete matching, aggregation, verification and figure code is embedded.'),code(BOOT),code('''import hashlib
import json
import numpy as np
import pandas as pd
REASONS=('Safety','Funding','Efficacy','Recruitment','Other')
PHASES=('I','II','III')
THRESHOLD=.5
NO_REASON='no recorded stopping reason'
UNAVAILABLE='matched Part 1 record; reasons unavailable'
OUT=REPRO_ROOT / 'integrated_error_analysis'
OUT.mkdir(parents=True,exist_ok=True)''')]
    cells += [md('## Matching and summary implementation (Michael’s unchanged evaluation logic)')]
    cells += source_defs('part2/error_analysis.py',{'match_unsuccessful','summarize','summarize_groups'})
    cells += source_defs('part2/evaluation.py',{'evaluate'})
    cells.insert(3,code('from sklearn.metrics import log_loss, f1_score, roc_auc_score, average_precision_score'))
    cells += source_defs('part2/report.py',{'markdown_table'})
    cells += [md('## Authoritative Part 1 provenance, recomputation and old/new comparison\n\nUse observed train/validation labels, hybrid pseudo-labels for masked train rows, and hybrid test predictions. No labels are inferred from outcome predictions. Original results remain in part2/outputs/; new tables are written separately.')]
    cells += source_defs('integration/reconcile_reasons.py',{'authoritative_reasons','reconcile','figures'})
    cells += [code('groups,coverage,comparison=reconcile(ROOT,OUT)\nfigures(groups,OUT)\ndisplay(coverage)\ndisplay(groups)\ndisplay(comparison[["phase","reason_group","n_trials_old","n_trials_integrated","false_success_rate_old","false_success_rate_integrated"]])'),code('from IPython.display import display, Image\ndisplay(Image(filename=str(OUT / "integrated_reason_errors.png")))'),md('## Interpretation\n\nThe 113 previously unresolved trials now have predictions, not ground truth. Across 452 unsuccessful phase rows there are 27 true-reason rows, 171 predicted-reason rows and 254 no-record rows; none remain unresolved. Phase II Recruitment changes from 18/93 false successes to 15/93 after pooled calibration. Phase III Efficacy remains 7/8 (all reason labels predicted); Phase III no record changes from 53/82 to 52/82. Empty Phase III Funding rates are undefined. No classifier or calibrator was retrained, and aggregate outcome metrics remain unchanged.')]
    save('part2_error_analysis_final.ipynb',cells)
    print('Created four final self-contained notebooks.')


if __name__=='__main__':build()
