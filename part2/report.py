"""Generate readable tables from saved experiments without rerunning training."""
import json
import pandas as pd
from .config import OUTPUT, PHASES


def markdown_table(frame):
    def fmt(v):
        return f'{v:.5f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]])


def main():
    results = pd.read_csv(OUTPUT / 'experiment_results.csv')
    metadata = json.loads((OUTPUT / 'run_metadata.json').read_text())
    best = results.loc[results.experiment.eq(metadata['best_experiment'])].iloc[0]
    lines = ['# Part 2 baseline experiment report', '',
        'Selection uses minimum overall validation binary log loss. All fits use only outcome_train; '
        'validation is used for evaluation and model selection, never fitting preprocessing or refitting the winner. '
        'F1 uses probability >= 0.5, with label=1 positive. PR-AUC means average precision, not trapezoidal area. '
        'Phase-macro log loss is also provided because the test set has equal phase sizes.', '',
        f"Selected: **{metadata['best_experiment']}**, log loss **{best.val_overall_log_loss:.5f}**. "
        f"The phase prior scores {results.iloc[0].val_overall_log_loss:.5f}.", '',
        '## All completed experiments', '',
        'clinical = diseases_cleaned + drugs + icdcodes; criteria = eligibility criteria; all_text = all four. '
        'Each text field has its own TF-IDF vocabulary and normalization. Pooled models include one-hot phase; '
        'phase-specific models fit independent vocabularies, year transformations, and classifiers. '
        'The same C is used for all three independent phase models. No class weighting or threshold tuning.', '']
    cols = ['experiment', 'train_overall_log_loss', 'val_overall_log_loss', 'val_phase_macro_log_loss',
            *[f'val_{p}_log_loss' for p in PHASES]]
    lines += [markdown_table(results[cols]), '']
    for phase in PHASES:
        lines += [f'## Phase {phase} validation metrics', '', markdown_table(results[
            ['experiment', f'val_{phase}_n', f'val_{phase}_log_loss', f'val_{phase}_f1',
             f'val_{phase}_roc_auc', f'val_{phase}_average_precision']]), '']
    lines += ['## Interpretation and limits', '',
        'The pooled criteria-only C=1 model wins this grid. Adding disease/drug/ICD text does not beat it; '
        'those fields may add sparse trial-specific associations that do not generalize. '
        'Larger C consistently widens the train/validation gap. Year does not improve the winning configuration. '
        'Pooled models generally benefit from sharing text coefficients across phases, especially with only 1,012 Phase I training rows. '
        'These are validation observations, not evidence of test performance.', '',
        'The random-by-trial validation split is earlier than the test distribution: train years 1993–2013, '
        'validation 1995–2013, test 2014–2021 (nonmissing). There are 21/6/0 missing years. '
        'A future trial-grouped temporal holdout drawn from training would be useful before more complex models. '
        'Repeated model selection on 1,013 validation rows can be optimistic; Phase I has only 149 rows. '
        'Combined-phase trials are correlated across phase rows, so row metrics are not independent-trial uncertainty estimates.', '',
        'There are no shared trial IDs or duplicate trial-phase rows. Training/validation share seven exact nonempty '
        'normalized eligibility texts and one full four-field text signature across distinct IDs; related protocols may '
        'make validation optimistic. Test has no such exact matches. The provided split is retained. '
        'No identifiers, phase_detail, reason labels, embeddings, drug-knowledge fields, or external data are predictive inputs.', '',
        'Reason-category error analysis and the assignment’s subsequent classifier improvement are intentionally deferred '
        'until review of these baselines. See ../ERROR_ANALYSIS.md. The assignment also requests notebook-format source; '
        '../baseline_workflow.ipynb runs this modular workflow, but a final report and completed analysis remain to be assembled.', '',
        '## Reproducibility', '',
        'run_metadata.json records input SHA-256 hashes, library versions, seed, vectorizer and solver settings. '
        'experiment_results.csv contains all configuration choices and train/validation metrics, including overall metrics. '
        'validation_predictions.csv preserves every experiment’s validation probabilities; best_validation_predictions.csv '
        'supports the later error analysis. best_model.pkl is a trusted local pickle; use the recorded environment to reload it. '
        'Fit times include feature fitting for each configuration even though features are reused across C values.', '']
    (OUTPUT / 'experiment_report.md').write_text('\n'.join(lines))
    print(markdown_table(results[['experiment', 'val_overall_log_loss']]))


if __name__ == '__main__':
    main()
