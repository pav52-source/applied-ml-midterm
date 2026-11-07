# Initial data audit

Text: criteria, diseases_cleaned, drugs, icdcodes. Additional available text/categorical fields: phase_detail, matched_drugs, drug_targets, drug_actions, drug_pathways, drug_smiles. Phase is categorical; start_year is numerical. ID and trial_id are matching keys, label is the binary target. Optional pathway_catalog has 2,501 rows and five columns; it is unused.

All splits have no duplicate rows or duplicate trial-phase records. Multiple phase rows of the same trial are retained within splits; no trial crosses splits. Cross-split exact text matches are reported as a possible related-protocol leakage risk. Year counts include missing values. Class balance and success rates are row-weighted, consistent with the assignment metric.

## schema

```text
split  rows           column   dtype
train  8037               ID     str
train  8037         trial_id     str
train  8037            phase     str
train  8037     phase_detail     str
train  8037         criteria     str
train  8037 diseases_cleaned     str
train  8037            drugs     str
train  8037         icdcodes     str
train  8037    matched_drugs     str
train  8037     drug_targets     str
train  8037     drug_actions     str
train  8037    drug_pathways     str
train  8037      drug_smiles     str
train  8037       start_year float64
train  8037            label   int64
  val  1013               ID     str
  val  1013         trial_id     str
  val  1013            phase     str
  val  1013     phase_detail     str
  val  1013         criteria     str
  val  1013 diseases_cleaned     str
  val  1013            drugs     str
  val  1013         icdcodes     str
  val  1013    matched_drugs     str
  val  1013     drug_targets     str
  val  1013     drug_actions     str
  val  1013    drug_pathways     str
  val  1013      drug_smiles     str
  val  1013       start_year float64
  val  1013            label   int64
 test  1500               ID     str
 test  1500         trial_id     str
 test  1500            phase     str
 test  1500     phase_detail     str
 test  1500         criteria     str
 test  1500 diseases_cleaned     str
 test  1500            drugs     str
 test  1500         icdcodes     str
 test  1500    matched_drugs     str
 test  1500     drug_targets     str
 test  1500     drug_actions     str
 test  1500    drug_pathways     str
 test  1500      drug_smiles     str
 test  1500       start_year float64
```

## missing_values

```text
split                               column  missing
train                                   ID        0
train                             trial_id        0
train                                phase        0
train                         phase_detail        0
train                             criteria        2
train                     diseases_cleaned        0
train                                drugs        0
train                             icdcodes        0
train                        matched_drugs     4361
train                         drug_targets     4361
train                         drug_actions     4394
train                        drug_pathways     4361
train                          drug_smiles     4361
train                           start_year       21
train                                label        0
train         criteria_empty_after_parsing        2
train diseases_cleaned_empty_after_parsing        0
train            drugs_empty_after_parsing        0
train         icdcodes_empty_after_parsing        0
  val                                   ID        0
  val                             trial_id        0
  val                                phase        0
  val                         phase_detail        0
  val                             criteria        0
  val                     diseases_cleaned        0
  val                                drugs        0
  val                             icdcodes        0
  val                        matched_drugs      526
  val                         drug_targets      526
  val                         drug_actions      529
  val                        drug_pathways      526
  val                          drug_smiles      526
  val                           start_year        6
  val                                label        0
  val         criteria_empty_after_parsing        0
  val diseases_cleaned_empty_after_parsing        0
  val            drugs_empty_after_parsing        0
  val         icdcodes_empty_after_parsing        0
 test                                   ID        0
 test                             trial_id        0
 test                                phase        0
 test                         phase_detail        0
 test                             criteria        0
 test                     diseases_cleaned        0
 test                                drugs        0
 test                             icdcodes        0
 test                        matched_drugs      910
 test                         drug_targets      910
 test                         drug_actions      910
 test                        drug_pathways      910
 test                          drug_smiles      910
 test                           start_year        0
 test         criteria_empty_after_parsing        0
 test diseases_cleaned_empty_after_parsing        0
 test            drugs_empty_after_parsing        0
 test         icdcodes_empty_after_parsing        0
```

## class_balance

```text
split   phase  rows  unique_trials  successes  success_rate
train overall  8037           7535     4466.0      0.555680
train       I  1012           1012      573.0      0.566206
train      II  3950           3950     1884.0      0.476962
train     III  3075           3075     2009.0      0.653333
  val overall  1013            901      561.0      0.553801
  val       I   149            149       85.0      0.570470
  val      II   501            501      236.0      0.471058
  val     III   363            363      240.0      0.661157
 test overall  1500           1442        NaN           NaN
 test       I   500            500        NaN           NaN
 test      II   500            500        NaN           NaN
 test     III   500            500        NaN           NaN
```

## start_year_counts

```text
split  start_year  rows
train      1993.0     2
train      1994.0     5
train      1995.0     8
train      1996.0    12
train      1997.0    24
train      1998.0    41
train      1999.0    44
train      2000.0    56
train      2001.0   114
train      2002.0   214
train      2003.0   349
train      2004.0   587
train      2005.0   779
train      2006.0  1194
train      2007.0  1285
train      2008.0   925
train      2009.0   362
train      2010.0   884
train      2011.0   707
train      2012.0   349
train      2013.0    75
train         NaN    21
  val      1995.0     2
  val      1996.0     2
  val      1997.0     4
  val      1998.0     5
  val      1999.0     8
  val      2000.0     6
  val      2001.0    17
  val      2002.0    22
  val      2003.0    42
  val      2004.0    68
  val      2005.0   101
  val      2006.0   160
  val      2007.0   141
  val      2008.0   109
  val      2009.0    45
  val      2010.0   121
  val      2011.0    93
  val      2012.0    53
  val      2013.0     8
  val         NaN     6
 test      2014.0   484
 test      2015.0   457
 test      2016.0   296
 test      2017.0   200
 test      2018.0    54
 test      2019.0     6
 test      2020.0     2
 test      2021.0     1
```

## duplicates_and_years

```text
split  exact_duplicate_rows  duplicate_trial_phase  multi_phase_trials  extra_phase_rows  year_min  year_median  year_max
train                     0                      0                 502               502    1993.0       2007.0    2013.0
  val                     0                      0                 112               112    1995.0       2007.0    2013.0
 test                     0                      0                  58                58    2014.0       2015.0    2021.0
```

## cross_split_overlap

```text
split_a split_b                                      key  shared_keys
  train     val                                 trial_id            0
  train     val                                 criteria            7
  train     val criteria+diseases_cleaned+drugs+icdcodes            1
  train    test                                 trial_id            0
  train    test                                 criteria            0
  train    test criteria+diseases_cleaned+drugs+icdcodes            0
    val    test                                 trial_id            0
    val    test                                 criteria            0
    val    test criteria+diseases_cleaned+drugs+icdcodes            0
```
