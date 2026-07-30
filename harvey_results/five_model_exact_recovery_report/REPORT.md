# Harvey exact-deduplication recovery report

Policy: preserve valid postprocessed arrays; for malformed rows, recover complete location evidence values from the generated array region, then remove exact repeated evidence lists within each row. A constrained fallback also accepts complete scalar evidence items when only the evidence-list closer is missing immediately before an object boundary. Matching uses the Harvey evaluator's global greedy RapidFuzz ratio at threshold 0.75.

## Overall results

Native-run results are shown here for provenance and before/after validity reporting.

| Model | Rows | Gold | JSON valid before | JSON valid after | Nonempty after | Duplicates removed | Original P/R/F1 | Recovered P/R/F1 | F1 change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | 1080 | 1871 | 85.83% (927) | 100.00% (1080) | 99.17% (1071) | 407 | 0.589714 / 0.539284 / 0.563372 | 0.568999 / 0.625869 / **0.596080** | +0.032708 |
| Gemma-2-9B | 1080 | 1871 | 81.67% (882) | 100.00% (1080) | 100.00% (1080) | 184 | 0.569584 / 0.424372 / 0.486371 | 0.571429 / 0.549439 / **0.560218** | +0.073847 |
| Gemma-2-27B | 1000 | 1738 | 5.90% (59) | 100.00% (1000) | 98.40% (984) | 7697 | 0.784314 / 0.023015 / 0.044718 | 0.439269 / 0.830265 / **0.574557** | +0.529839 |
| Llama-3.2-3B | 1080 | 1871 | 80.00% (864) | 100.00% (1080) | 99.91% (1079) | 1353 | 0.331264 / 0.541956 / 0.411192 | 0.304607 / 0.681988 / **0.421122** | +0.009930 |
| Qwen2.5-3B | 1080 | 1871 | 54.63% (590) | 100.00% (1080) | 99.91% (1079) | 157 | 0.566731 / 0.313201 / 0.403442 | 0.483755 / 0.501336 / **0.492388** | +0.088946 |

## Common 1000-row comparison

This is the primary like-for-like comparison: every model is rescored on the same first 1000 prompts and gold rows.

| Model | JSON valid before | Nonempty after | Duplicates removed | Original P/R/F1 | Recovered P/R/F1 | F1 change |
|---|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | 86.60% (866) | 99.10% (991) | 339 | 0.588750 / 0.542002 / 0.564410 | 0.571053 / 0.624281 / **0.596482** | +0.032072 |
| Gemma-2-9B | 81.10% (811) | 100.00% (1000) | 179 | 0.579528 / 0.423475 / 0.489362 | 0.577805 / 0.551208 / **0.564193** | +0.074831 |
| Gemma-2-27B | 5.90% (59) | 98.40% (984) | 7697 | 0.784314 / 0.023015 / 0.044718 | 0.439269 / 0.830265 / **0.574557** | +0.529839 |
| Llama-3.2-3B | 80.20% (802) | 99.90% (999) | 1242 | 0.331573 / 0.542002 / 0.411444 | 0.305879 / 0.679517 / **0.421861** | +0.010417 |
| Qwen2.5-3B | 54.30% (543) | 99.90% (999) | 146 | 0.569182 / 0.312428 / 0.403418 | 0.486018 / 0.500000 / **0.492910** | +0.089492 |

## Validity and recovery detail

| Model | Valid nonempty before | Valid empty before | Malformed | Malformed recovered nonempty | Missing-list-close fields salvaged | Malformed unrecoverable | Empty after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | 927 | 0 | 153 | 144 | 25 | 9 | 9 |
| Gemma-2-9B | 882 | 0 | 198 | 198 | 24 | 0 | 0 |
| Gemma-2-27B | 44 | 15 | 941 | 940 | 33 | 1 | 16 |
| Llama-3.2-3B | 863 | 1 | 216 | 216 | 0 | 0 | 1 |
| Qwen2.5-3B | 590 | 0 | 490 | 489 | 510 | 1 | 1 |

## Overall TP/FP/FN

| Model | Original TP/FP/FN | Recovered TP/FP/FN |
|---|---:|---:|
| Gemma-2-2B | 1009 / 702 / 862 | 1171 / 887 / 700 |
| Gemma-2-9B | 794 / 600 / 1077 | 1028 / 771 / 843 |
| Gemma-2-27B | 40 / 11 / 1698 | 1443 / 1842 / 295 |
| Llama-3.2-3B | 1014 / 2047 / 857 | 1276 / 2913 / 595 |
| Qwen2.5-3B | 586 / 448 / 1285 | 938 / 1001 / 933 |

## Deduplication sensitivity

| Model | No dedup F1 | Exact dedup F1 | Normalized dedup F1 |
|---|---:|---:|---:|
| Gemma-2-2B | 0.540129 | **0.596080** | 0.596080 |
| Gemma-2-9B | 0.534510 | **0.560218** | 0.560371 |
| Gemma-2-27B | 0.227516 | **0.574557** | 0.576739 |
| Llama-3.2-3B | 0.344260 | **0.421122** | 0.421609 |
| Qwen2.5-3B | 0.472901 | **0.492388** | 0.492776 |

## Common 1000-row recovered gold-category F1

| Category | Gemma-2-2B | Gemma-2-9B | Gemma-2-27B | Llama-3.2-3B | Qwen2.5-3B |
|---|---:|---:|---:|---:|---:|
| C1 | 0.785388 | 0.646310 | 0.946535 | 0.877637 | 0.855914 |
| C2 | 0.668605 | 0.577640 | 0.840506 | 0.676301 | 0.478405 |
| C3 | 0.368421 | 0.176471 | 0.680851 | 0.410256 | 0.176471 |
| C4 | 0.000000 | 0.400000 | 0.400000 | 0.400000 | 0.400000 |
| C5 | 0.732558 | 0.732558 | 0.870466 | 0.840426 | 0.710059 |
| C6 | 0.764706 | 0.615385 | 0.875000 | 0.864865 | 0.673684 |
| C7 | 0.807292 | 0.680115 | 0.877451 | 0.828645 | 0.660819 |
| C8 | 0.725490 | 0.602151 | 0.859649 | 0.818182 | 0.522727 |
| C9 | 0.804959 | 0.808896 | 0.945295 | 0.816694 | 0.653631 |
| C10 | 0.000000 | 0.000000 | 0.250000 | 0.727273 | 0.727273 |
| C11 | 0.909091 | 0.666667 | 0.956522 | 0.909091 | 0.956522 |

## Interpretation notes

- JSON-valid-after is 100% because every custom prediction is serialized as an array; an unrecoverable row becomes `[]`. Use the nonempty-after column to see remaining coverage loss.
- Exact deduplication is a recovery policy, not the untouched raw-model score. It removes only identical recovered evidence lists within a row; already-valid postprocessed arrays are preserved.
- True `pred_raw` is available for Gemma-2-27B. In the four older artifacts it is blank, so invalid `pred_postproc_raw` is the recovery source; valid postprocessed arrays are still preserved.
- Category metrics reproduce the evaluator's gold-category attribution. Unmatched predictions are overall false positives but are not assigned to category-level false positives.
- The common-prefix table equalizes rows, gold labels, recovery, and scoring. It does not equalize generation protocol: Gemma-2-27B came from the newer 4-bit evaluator, while the four smaller-model artifacts came from the older non-4-bit evaluator.
- Test-set fingerprints:
  - `bb730e92f161b14f`: Gemma-2-2B, Gemma-2-9B, Llama-3.2-3B, Qwen2.5-3B
  - `cd3db006255208c0`: Gemma-2-27B
- Native runs with different fingerprints, row counts, or gold totals are not strictly comparable as a model ranking.
- The common-prefix table is directly comparable: all five models use the same `cd3db006255208c0` fingerprint.
