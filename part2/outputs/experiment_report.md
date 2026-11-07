# Part 2 baseline experiment report

Selection uses minimum overall validation binary log loss. All fits use only outcome_train; validation is used for evaluation and model selection, never fitting preprocessing or refitting the winner. F1 uses probability >= 0.5, with label=1 positive. PR-AUC means average precision, not trapezoidal area. Phase-macro log loss is also provided because the test set has equal phase sizes.

Selected: **criteria_year0_pooled_C1**, log loss **0.57375**. The phase prior scores 0.67199.

## All completed experiments

clinical = diseases_cleaned + drugs + icdcodes; criteria = eligibility criteria; all_text = all four. Each text field has its own TF-IDF vocabulary and normalization. Pooled models include one-hot phase; phase-specific models fit independent vocabularies, year transformations, and classifiers. The same C is used for all three independent phase models. No class weighting or threshold tuning.

| experiment | train_overall_log_loss | val_overall_log_loss | val_phase_macro_log_loss | val_I_log_loss | val_II_log_loss | val_III_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| phase_prior | 0.67323 | 0.67199 | 0.67172 | 0.68322 | 0.69154 | 0.64040 |
| clinical_year0_pooled_C0.1 | 0.57172 | 0.60076 | 0.60399 | 0.63296 | 0.62527 | 0.55372 |
| clinical_year0_pooled_C1 | 0.45687 | 0.61318 | 0.62062 | 0.66481 | 0.63448 | 0.56258 |
| clinical_year0_pooled_C10 | 0.31548 | 0.73561 | 0.74967 | 0.81404 | 0.74624 | 0.68874 |
| clinical_year0_phase_specific_C0.1 | 0.57844 | 0.60732 | 0.60965 | 0.63651 | 0.63416 | 0.55828 |
| clinical_year0_phase_specific_C1 | 0.46068 | 0.61252 | 0.61427 | 0.64023 | 0.64170 | 0.56088 |
| clinical_year0_phase_specific_C10 | 0.31436 | 0.76272 | 0.77839 | 0.84352 | 0.76438 | 0.72728 |
| clinical_year1_pooled_C0.1 | 0.56823 | 0.60181 | 0.60378 | 0.62969 | 0.62951 | 0.55214 |
| clinical_year1_pooled_C1 | 0.45550 | 0.61691 | 0.62376 | 0.66691 | 0.64034 | 0.56404 |
| clinical_year1_pooled_C10 | 0.31476 | 0.74134 | 0.75549 | 0.82142 | 0.75385 | 0.69120 |
| clinical_year1_phase_specific_C0.1 | 0.57347 | 0.60726 | 0.60791 | 0.62843 | 0.63497 | 0.56034 |
| clinical_year1_phase_specific_C1 | 0.45859 | 0.61555 | 0.61582 | 0.63587 | 0.64494 | 0.56665 |
| clinical_year1_phase_specific_C10 | 0.31289 | 0.76891 | 0.78390 | 0.84632 | 0.77064 | 0.73476 |
| criteria_year0_pooled_C0.1 | 0.59689 | 0.60898 | 0.60766 | 0.61321 | 0.62600 | 0.58376 |
| criteria_year0_pooled_C1 | 0.46903 | 0.57375 | 0.56816 | 0.56316 | 0.60146 | 0.53986 |
| criteria_year0_pooled_C10 | 0.23523 | 0.62325 | 0.60947 | 0.58409 | 0.67127 | 0.57304 |
| criteria_year0_phase_specific_C0.1 | 0.60972 | 0.62679 | 0.63094 | 0.65598 | 0.63932 | 0.59753 |
| criteria_year0_phase_specific_C1 | 0.46823 | 0.58720 | 0.58576 | 0.59492 | 0.61057 | 0.55178 |
| criteria_year0_phase_specific_C10 | 0.20749 | 0.62275 | 0.60343 | 0.55502 | 0.67021 | 0.58506 |
| criteria_year1_pooled_C0.1 | 0.59483 | 0.60963 | 0.60724 | 0.61031 | 0.62953 | 0.58189 |
| criteria_year1_pooled_C1 | 0.46863 | 0.57484 | 0.56897 | 0.56281 | 0.60257 | 0.54152 |
| criteria_year1_pooled_C10 | 0.23508 | 0.62398 | 0.61029 | 0.58466 | 0.67099 | 0.57523 |
| criteria_year1_phase_specific_C0.1 | 0.60444 | 0.62424 | 0.62635 | 0.64485 | 0.63952 | 0.59469 |
| criteria_year1_phase_specific_C1 | 0.46696 | 0.58850 | 0.58675 | 0.59388 | 0.61062 | 0.55576 |
| criteria_year1_phase_specific_C10 | 0.20701 | 0.62580 | 0.60706 | 0.55892 | 0.66998 | 0.59227 |
| all_text_year0_pooled_C0.1 | 0.54239 | 0.58635 | 0.58721 | 0.60629 | 0.61050 | 0.54484 |
| all_text_year0_pooled_C1 | 0.36763 | 0.58768 | 0.59011 | 0.61615 | 0.61264 | 0.54155 |
| all_text_year0_pooled_C10 | 0.14301 | 0.71206 | 0.71279 | 0.72941 | 0.73323 | 0.67572 |
| all_text_year0_phase_specific_C0.1 | 0.55202 | 0.59702 | 0.59921 | 0.62408 | 0.62169 | 0.55186 |
| all_text_year0_phase_specific_C1 | 0.36858 | 0.59298 | 0.59009 | 0.59897 | 0.62506 | 0.54625 |
| all_text_year0_phase_specific_C10 | 0.13360 | 0.71754 | 0.70441 | 0.67736 | 0.75885 | 0.67703 |
| all_text_year1_pooled_C0.1 | 0.54120 | 0.58858 | 0.58868 | 0.60599 | 0.61475 | 0.54531 |
| all_text_year1_pooled_C1 | 0.36749 | 0.58948 | 0.59158 | 0.61646 | 0.61477 | 0.54350 |
| all_text_year1_pooled_C10 | 0.14297 | 0.71303 | 0.71363 | 0.72945 | 0.73375 | 0.67769 |
| all_text_year1_phase_specific_C0.1 | 0.54915 | 0.59821 | 0.59907 | 0.61869 | 0.62321 | 0.55531 |
| all_text_year1_phase_specific_C1 | 0.36798 | 0.59575 | 0.59236 | 0.59803 | 0.62602 | 0.55304 |
| all_text_year1_phase_specific_C10 | 0.13344 | 0.72058 | 0.70729 | 0.67777 | 0.75908 | 0.68502 |

## Phase I validation metrics

| experiment | val_I_n | val_I_log_loss | val_I_f1 | val_I_roc_auc | val_I_average_precision |
| --- | --- | --- | --- | --- | --- |
| phase_prior | 149 | 0.68322 | 0.72650 | 0.50000 | 0.57047 |
| clinical_year0_pooled_C0.1 | 149 | 0.63296 | 0.71823 | 0.67904 | 0.73341 |
| clinical_year0_pooled_C1 | 149 | 0.66481 | 0.69767 | 0.67537 | 0.69915 |
| clinical_year0_pooled_C10 | 149 | 0.81404 | 0.67066 | 0.65349 | 0.68802 |
| clinical_year0_phase_specific_C0.1 | 149 | 0.63651 | 0.72362 | 0.69228 | 0.77811 |
| clinical_year0_phase_specific_C1 | 149 | 0.64023 | 0.66298 | 0.67390 | 0.74642 |
| clinical_year0_phase_specific_C10 | 149 | 0.84352 | 0.65116 | 0.59651 | 0.64940 |
| clinical_year1_pooled_C0.1 | 149 | 0.62969 | 0.71658 | 0.68456 | 0.73956 |
| clinical_year1_pooled_C1 | 149 | 0.66691 | 0.70930 | 0.67482 | 0.70069 |
| clinical_year1_pooled_C10 | 149 | 0.82142 | 0.67836 | 0.65018 | 0.68545 |
| clinical_year1_phase_specific_C0.1 | 149 | 0.62843 | 0.71875 | 0.70184 | 0.78109 |
| clinical_year1_phase_specific_C1 | 149 | 0.63587 | 0.68182 | 0.68364 | 0.75121 |
| clinical_year1_phase_specific_C10 | 149 | 0.84632 | 0.65517 | 0.60368 | 0.65282 |
| criteria_year0_pooled_C0.1 | 149 | 0.61321 | 0.74490 | 0.73143 | 0.79667 |
| criteria_year0_pooled_C1 | 149 | 0.56316 | 0.77174 | 0.78327 | 0.83880 |
| criteria_year0_pooled_C10 | 149 | 0.58409 | 0.71264 | 0.76857 | 0.82847 |
| criteria_year0_phase_specific_C0.1 | 149 | 0.65598 | 0.75676 | 0.76066 | 0.80672 |
| criteria_year0_phase_specific_C1 | 149 | 0.59492 | 0.78351 | 0.78603 | 0.82267 |
| criteria_year0_phase_specific_C10 | 149 | 0.55502 | 0.78889 | 0.78842 | 0.82839 |
| criteria_year1_pooled_C0.1 | 149 | 0.61031 | 0.73469 | 0.72096 | 0.80185 |
| criteria_year1_pooled_C1 | 149 | 0.56281 | 0.75410 | 0.77665 | 0.83897 |
| criteria_year1_pooled_C10 | 149 | 0.58466 | 0.72000 | 0.76857 | 0.83061 |
| criteria_year1_phase_specific_C0.1 | 149 | 0.64485 | 0.71498 | 0.69026 | 0.77588 |
| criteria_year1_phase_specific_C1 | 149 | 0.59388 | 0.78534 | 0.78199 | 0.82141 |
| criteria_year1_phase_specific_C10 | 149 | 0.55892 | 0.77095 | 0.78750 | 0.83401 |
| all_text_year0_pooled_C0.1 | 149 | 0.60629 | 0.74157 | 0.72739 | 0.77058 |
| all_text_year0_pooled_C1 | 149 | 0.61615 | 0.68639 | 0.72739 | 0.76188 |
| all_text_year0_pooled_C10 | 149 | 0.72941 | 0.69091 | 0.72500 | 0.75001 |
| all_text_year0_phase_specific_C0.1 | 149 | 0.62408 | 0.72165 | 0.73125 | 0.80449 |
| all_text_year0_phase_specific_C1 | 149 | 0.59897 | 0.70330 | 0.74301 | 0.79823 |
| all_text_year0_phase_specific_C10 | 149 | 0.67736 | 0.71765 | 0.72004 | 0.76527 |
| all_text_year1_pooled_C0.1 | 149 | 0.60599 | 0.73514 | 0.72482 | 0.77667 |
| all_text_year1_pooled_C1 | 149 | 0.61646 | 0.70588 | 0.72610 | 0.76253 |
| all_text_year1_pooled_C10 | 149 | 0.72945 | 0.70238 | 0.72610 | 0.75255 |
| all_text_year1_phase_specific_C0.1 | 149 | 0.61869 | 0.73684 | 0.72482 | 0.79904 |
| all_text_year1_phase_specific_C1 | 149 | 0.59803 | 0.71186 | 0.74044 | 0.79589 |
| all_text_year1_phase_specific_C10 | 149 | 0.67777 | 0.70115 | 0.72279 | 0.76881 |

## Phase II validation metrics

| experiment | val_II_n | val_II_log_loss | val_II_f1 | val_II_roc_auc | val_II_average_precision |
| --- | --- | --- | --- | --- | --- |
| phase_prior | 501 | 0.69154 | 0.00000 | 0.50000 | 0.47106 |
| clinical_year0_pooled_C0.1 | 501 | 0.62527 | 0.59009 | 0.70243 | 0.68714 |
| clinical_year0_pooled_C1 | 501 | 0.63448 | 0.60444 | 0.70580 | 0.66545 |
| clinical_year0_pooled_C10 | 501 | 0.74624 | 0.63830 | 0.69421 | 0.65320 |
| clinical_year0_phase_specific_C0.1 | 501 | 0.63416 | 0.59735 | 0.69414 | 0.68664 |
| clinical_year0_phase_specific_C1 | 501 | 0.64170 | 0.58874 | 0.68426 | 0.66602 |
| clinical_year0_phase_specific_C10 | 501 | 0.76438 | 0.58174 | 0.66288 | 0.64425 |
| clinical_year1_pooled_C0.1 | 501 | 0.62951 | 0.62084 | 0.69594 | 0.66762 |
| clinical_year1_pooled_C1 | 501 | 0.64034 | 0.62252 | 0.70098 | 0.65477 |
| clinical_year1_pooled_C10 | 501 | 0.75385 | 0.63866 | 0.69010 | 0.64778 |
| clinical_year1_phase_specific_C0.1 | 501 | 0.63497 | 0.59424 | 0.68862 | 0.66994 |
| clinical_year1_phase_specific_C1 | 501 | 0.64494 | 0.58772 | 0.68290 | 0.65566 |
| clinical_year1_phase_specific_C10 | 501 | 0.77064 | 0.57203 | 0.66050 | 0.63788 |
| criteria_year0_pooled_C0.1 | 501 | 0.62600 | 0.64557 | 0.72167 | 0.70079 |
| criteria_year0_pooled_C1 | 501 | 0.60146 | 0.64655 | 0.73556 | 0.71390 |
| criteria_year0_pooled_C10 | 501 | 0.67127 | 0.64344 | 0.71855 | 0.67780 |
| criteria_year0_phase_specific_C0.1 | 501 | 0.63932 | 0.63248 | 0.71471 | 0.68920 |
| criteria_year0_phase_specific_C1 | 501 | 0.61057 | 0.61404 | 0.72099 | 0.70027 |
| criteria_year0_phase_specific_C10 | 501 | 0.67021 | 0.60475 | 0.69220 | 0.66910 |
| criteria_year1_pooled_C0.1 | 501 | 0.62953 | 0.62420 | 0.70046 | 0.67133 |
| criteria_year1_pooled_C1 | 501 | 0.60257 | 0.64668 | 0.73459 | 0.71113 |
| criteria_year1_pooled_C10 | 501 | 0.67099 | 0.65306 | 0.71839 | 0.67951 |
| criteria_year1_phase_specific_C0.1 | 501 | 0.63952 | 0.61803 | 0.68929 | 0.65702 |
| criteria_year1_phase_specific_C1 | 501 | 0.61062 | 0.62693 | 0.72004 | 0.70051 |
| criteria_year1_phase_specific_C10 | 501 | 0.66998 | 0.60385 | 0.69178 | 0.66914 |
| all_text_year0_pooled_C0.1 | 501 | 0.61050 | 0.62281 | 0.72538 | 0.70762 |
| all_text_year0_pooled_C1 | 501 | 0.61264 | 0.65188 | 0.72987 | 0.69600 |
| all_text_year0_pooled_C10 | 501 | 0.73323 | 0.65272 | 0.71660 | 0.67552 |
| all_text_year0_phase_specific_C0.1 | 501 | 0.62169 | 0.60832 | 0.71254 | 0.70052 |
| all_text_year0_phase_specific_C1 | 501 | 0.62506 | 0.61024 | 0.70307 | 0.69248 |
| all_text_year0_phase_specific_C10 | 501 | 0.75885 | 0.59740 | 0.68622 | 0.67483 |
| all_text_year1_pooled_C0.1 | 501 | 0.61475 | 0.63415 | 0.71519 | 0.69141 |
| all_text_year1_pooled_C1 | 501 | 0.61477 | 0.65495 | 0.72832 | 0.69191 |
| all_text_year1_pooled_C10 | 501 | 0.73375 | 0.64718 | 0.71599 | 0.67426 |
| all_text_year1_phase_specific_C0.1 | 501 | 0.62321 | 0.62084 | 0.70764 | 0.69009 |
| all_text_year1_phase_specific_C1 | 501 | 0.62602 | 0.61062 | 0.70221 | 0.69016 |
| all_text_year1_phase_specific_C10 | 501 | 0.75908 | 0.59307 | 0.68575 | 0.67412 |

## Phase III validation metrics

| experiment | val_III_n | val_III_log_loss | val_III_f1 | val_III_roc_auc | val_III_average_precision |
| --- | --- | --- | --- | --- | --- |
| phase_prior | 363 | 0.64040 | 0.79602 | 0.50000 | 0.66116 |
| clinical_year0_pooled_C0.1 | 363 | 0.55372 | 0.81331 | 0.75898 | 0.86607 |
| clinical_year0_pooled_C1 | 363 | 0.56258 | 0.80464 | 0.73516 | 0.82689 |
| clinical_year0_pooled_C10 | 363 | 0.68874 | 0.76000 | 0.69804 | 0.79430 |
| clinical_year0_phase_specific_C0.1 | 363 | 0.55828 | 0.79702 | 0.75246 | 0.86324 |
| clinical_year0_phase_specific_C1 | 363 | 0.56088 | 0.79365 | 0.73962 | 0.83582 |
| clinical_year0_phase_specific_C10 | 363 | 0.72728 | 0.75654 | 0.68840 | 0.79021 |
| clinical_year1_pooled_C0.1 | 363 | 0.55214 | 0.80669 | 0.75318 | 0.85972 |
| clinical_year1_pooled_C1 | 363 | 0.56404 | 0.80777 | 0.73384 | 0.82201 |
| clinical_year1_pooled_C10 | 363 | 0.69120 | 0.76096 | 0.70088 | 0.79662 |
| clinical_year1_phase_specific_C0.1 | 363 | 0.56034 | 0.79623 | 0.74085 | 0.84979 |
| clinical_year1_phase_specific_C1 | 363 | 0.56665 | 0.79600 | 0.73692 | 0.82317 |
| clinical_year1_phase_specific_C10 | 363 | 0.73476 | 0.75806 | 0.68764 | 0.78691 |
| criteria_year0_pooled_C0.1 | 363 | 0.58376 | 0.80690 | 0.75596 | 0.86629 |
| criteria_year0_pooled_C1 | 363 | 0.53986 | 0.82657 | 0.76707 | 0.86074 |
| criteria_year0_pooled_C10 | 363 | 0.57304 | 0.80240 | 0.74631 | 0.83370 |
| criteria_year0_phase_specific_C0.1 | 363 | 0.59753 | 0.79727 | 0.74370 | 0.85728 |
| criteria_year0_phase_specific_C1 | 363 | 0.55178 | 0.81181 | 0.75474 | 0.85889 |
| criteria_year0_phase_specific_C10 | 363 | 0.58506 | 0.80934 | 0.73496 | 0.83807 |
| criteria_year1_pooled_C0.1 | 363 | 0.58189 | 0.80139 | 0.72686 | 0.84980 |
| criteria_year1_pooled_C1 | 363 | 0.54152 | 0.83209 | 0.76511 | 0.86093 |
| criteria_year1_pooled_C10 | 363 | 0.57523 | 0.79678 | 0.74624 | 0.83503 |
| criteria_year1_phase_specific_C0.1 | 363 | 0.59469 | 0.79859 | 0.68449 | 0.81681 |
| criteria_year1_phase_specific_C1 | 363 | 0.55576 | 0.80907 | 0.74448 | 0.85381 |
| criteria_year1_phase_specific_C10 | 363 | 0.59227 | 0.80312 | 0.73039 | 0.83643 |
| all_text_year0_pooled_C0.1 | 363 | 0.54484 | 0.81455 | 0.77266 | 0.87375 |
| all_text_year0_pooled_C1 | 363 | 0.54155 | 0.81382 | 0.75627 | 0.84092 |
| all_text_year0_pooled_C10 | 363 | 0.67572 | 0.78486 | 0.71711 | 0.80311 |
| all_text_year0_phase_specific_C0.1 | 363 | 0.55186 | 0.80374 | 0.76226 | 0.86896 |
| all_text_year0_phase_specific_C1 | 363 | 0.54625 | 0.79688 | 0.75549 | 0.84987 |
| all_text_year0_phase_specific_C10 | 363 | 0.67703 | 0.78800 | 0.73045 | 0.82496 |
| all_text_year1_pooled_C0.1 | 363 | 0.54531 | 0.80443 | 0.76626 | 0.86887 |
| all_text_year1_pooled_C1 | 363 | 0.54350 | 0.81905 | 0.75539 | 0.83757 |
| all_text_year1_pooled_C10 | 363 | 0.67769 | 0.78486 | 0.71755 | 0.80259 |
| all_text_year1_phase_specific_C0.1 | 363 | 0.55531 | 0.80226 | 0.74817 | 0.85444 |
| all_text_year1_phase_specific_C1 | 363 | 0.55304 | 0.80000 | 0.75000 | 0.84070 |
| all_text_year1_phase_specific_C10 | 363 | 0.68502 | 0.78557 | 0.72954 | 0.82083 |

## Interpretation and limits

The pooled criteria-only C=1 model wins this grid. Adding disease/drug/ICD text does not beat it; those fields may add sparse trial-specific associations that do not generalize. Larger C consistently widens the train/validation gap. Year does not improve the winning configuration. Pooled models generally benefit from sharing text coefficients across phases, especially with only 1,012 Phase I training rows. These are validation observations, not evidence of test performance.

The random-by-trial validation split is earlier than the test distribution: train years 1993–2013, validation 1995–2013, test 2014–2021 (nonmissing). There are 21/6/0 missing years. A future trial-grouped temporal holdout drawn from training would be useful before more complex models. Repeated model selection on 1,013 validation rows can be optimistic; Phase I has only 149 rows. Combined-phase trials are correlated across phase rows, so row metrics are not independent-trial uncertainty estimates.

There are no shared trial IDs or duplicate trial-phase rows. Training/validation share seven exact nonempty normalized eligibility texts and one full four-field text signature across distinct IDs; related protocols may make validation optimistic. Test has no such exact matches. The provided split is retained. No identifiers, phase_detail, reason labels, embeddings, drug-knowledge fields, or external data are predictive inputs.

Reason-category error analysis and the assignment’s subsequent classifier improvement are intentionally deferred until review of these baselines. See ../ERROR_ANALYSIS.md. The assignment also requests notebook-format source; ../baseline_workflow.ipynb runs this modular workflow, but a final report and completed analysis remain to be assembled.

## Reproducibility

run_metadata.json records input SHA-256 hashes, library versions, seed, vectorizer and solver settings. experiment_results.csv contains all configuration choices and train/validation metrics, including overall metrics. validation_predictions.csv preserves every experiment’s validation probabilities; best_validation_predictions.csv supports the later error analysis. best_model.pkl is a trusted local pickle; use the recorded environment to reload it. Fit times include feature fitting for each configuration even though features are reused across C values.
