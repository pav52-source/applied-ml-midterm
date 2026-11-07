# Deferred reason-category error analysis

This is the explicit integration point for the next stage, after baseline review.
No Part 1 data is used in the current classifier. Do not feed reason labels or
reason embeddings into features when adding this analysis.

Input: `outputs/best_validation_predictions.csv`, containing validation ID,
trial_id, phase, true outcome, probability, and predicted_label at threshold 0.5.
Join Part 1 trial `ID` to Part 2 `trial_id`, never to the phase-suffixed ID.

Build a separate analysis-only reason table with unique trial_id, Safety, Funding,
Efficacy, Recruitment, Other, and a source field identifying true labels versus
Part 1 predictions. Masked labels (-100) are not true negatives. Use available
true labels where permitted and the established Part 1 predictions otherwise.
The absence of a reason in a matched trial is different from no Part 1 record;
reserve “no recorded stopping reason” for trials absent from all Part 1 records.

Filter to outcome label=0. For each phase and each of the five categories plus
the no-recorded-reason group, report:

- phase, reason_group, n_trials, n_predicted_success, false_success_rate;
- counts by reason-label source (true versus predicted).

Deduplicate trial_id within each phase. Multi-label trials contribute to every
applicable category; counts need not sum to the total. Use probability >= 0.5 for
false successes. Report an undefined rate for empty groups rather than zero.
Do not interpret rate differences without group sizes and reason-prediction
uncertainty. Save baseline rates before evaluating a proposed improvement using
the same groups, split, and threshold.

The current implementation deliberately does not create invented category rates.
The analysis and subsequent improvement required for the full assignment remain
pending, as requested for this initial baseline stage.
