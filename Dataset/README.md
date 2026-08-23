# Dataset folder

## Pipeline (what feeds into what)

```
Reviewer Dataset(2000-2014).csv ───────┐
                                        ├──► reviewer_dataset.csv
Amazom Reviwe Dataset (2015-2023).csv ─┴──► Reviewer Dataset(2015-2023).csv ──┘
```

## File descriptions

| File | Role |
|---|---|
| `reviewer_dataset.csv` | **Final merged dataset.** This is the only file loaded by the behavioral model notebooks (`behavorial_model_01/02/03/final.ipynb`). |
| `Reviewer Dataset(2000-2014).csv` | Intermediate output, merged into `reviewer_dataset.csv`. Produced in `notebooks/feature_engg_01.ipynb` (source raw file has been moved to the archive folder). |
| `Reviewer Dataset(2015-2023).csv` | Intermediate output — produced from `Amazom Reviwe Dataset (2015-2023).csv` in `notebooks/feature_enng_02.ipynb`, merged into `reviewer_dataset.csv`. |
| `Amazom Reviwe Dataset (2015-2023).csv` | Raw source file, review-level (pre-aggregation), used to build `Reviewer Dataset(2015-2023).csv`. |
| `Amazon Review Dataset(2000-2014).csv` | **Not referenced by any notebook.** Appears to be an unused/superseded file — confirm whether it's safe to remove before final submission. |

If you only need to run the behavioral model, `reviewer_dataset.csv` is the single file that matters — everything else here is provenance/intermediate data.
