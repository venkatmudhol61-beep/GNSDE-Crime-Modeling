# GN-SDE: Graph Neural Stochastic Differential Equations for Crime Forecasting

GN-SDE models spatio-temporal crime dynamics across urban regions by combining Graph Neural Networks with Stochastic Differential Equations solved via [`torchsde`](https://github.com/google-research/torchsde).

**Datasets:** Chicago Beats · Chicago Districts · NYC Precincts

---

## Datasets

| Dataset | Source | Link |
|---|---|---|
| NYPD Complaint Data Historic | NYC Open Data | [qgea-i56i](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i) |
| NYPD Arrests Data Historic | NYC Open Data | [8h9b-rp9u](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u) |
| Chicago Police Beats | Chicago Data Portal | [aerh-rz74](https://data.cityofchicago.org/Public-Safety/Boundaries-Police-Beats-current-/aerh-rz74) |
| Chicago Crimes 2001-Present | Chicago Data Portal | [ijzp-q8t2](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Repository Structure

```
data/
├── raw/
└── processed/

src/
├── data/
│   ├── build_crime.py
│   ├── build_arrest.py
│   └── build_regions_order.py
├── graph/
│   ├── build_adjsp_chicago_districts.py
│   ├── build_adjsp_chicago_beats.py
│   └── build_adjsp_nyc_precincts.py
├── model/
│   └── gnsde_new32_final_paper.py
├── train_gnsde.py
└── train_gnode.py
```

---

## Data Pipeline

Run in order:

```bash
python src/data/build_crime.py            # → data/processed/*_crime_timeseries.csv
python src/data/build_arrest.py           # → data/processed/*_arrest_timeseries.csv
python src/data/build_regions_order.py    # → data/raw/graph_*_order.csv

python src/graph/build_adjsp_chicago_districts.py
python src/graph/build_adjsp_chicago_beats.py
python src/graph/build_adjsp_nyc_precincts.py
# → data/processed/*_adjacency_spatial.npy
# → data/processed/*_adjacencyfc.npy
```

---

## Model Variants

All variants use `hidden_dim=32`.

| Variant | Description |
|---|---|
| `fc` | Fully-connected baseline |
| `spatial` | Spatial smoothing via Tobler `rho` |
| `spatial_attn` | Graph attention with enforcement beta modulator `m_beta` |
| `latent` | Latent-memory GN-SDE with `obs_gate` enforcement blending |

---

## Training

Set `DATASET` and `VARIANT` at the top of `train_gnsde.py`:

```python
DATASET = "chicago_beats"   # chicago_beats | chicago_districts | nyc_precincts
VARIANT = "latent"          # fc | spatial | spatial_attn | latent
```

```bash
python train_gnsde.py
```

Training runs in two phases:

- **Phase 1 — Huber warmup.** Drift network only; `log_sigma_pred` frozen.
- **Phase 2 — Huber + NLL calibration.** Best Phase 1 checkpoint reloaded; `log_sigma_pred` unfrozen; drift LR reduced to `LR × 0.1`.

Early stopping is applied in both phases.

---

## Inference

After training, 100 MC forward passes are run to estimate predictive uncertainty:

```
total variance = MC variance across SDE paths + sigma_pred²
```

A 90% prediction interval is calibrated empirically on the validation set.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/{DATASET}_{VARIANT}_final_best.pt` | Best model checkpoint |
| `data/processed/{DATASET}_{VARIANT}_final_predictions.csv` | `month, region_id, C_gnsde, C_std, C_p05, C_p95, split` |
| `data/processed/{DATASET}_{VARIANT}_final_history.csv` | `epoch, phase, train, val, sigma_pred` |
| `data/processed/{DATASET}_latent_final_enforcement.csv` | `month, region_id, L_latent` — latent variant only |

---

## Metrics

Test-set metrics reported after inference: MAE, RMSE, R², sMAPE, Coverage@90%, Interval Width.

---

## Diagnostics

Variant-specific diagnostics printed after evaluation:

- **`latent`** — `obs_gate` balance, `Corr(L_latent, L_obs)`, per-region `L_latent` variation
- **`spatial_attn`** — `m_beta` range over time and regions, `rho`, collapse warning
- **`spatial`** — `rho`, pseudo-Moran's I
- **`fc`** — `Corr(C, AC)`, edge count

---

## Reproducibility

Seeds are fixed in `train_gnsde.py`:

```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## Citation

```bibtex
@article{yourpaper2026,
  title   = {Graph Neural Stochastic Differential Equations for Crime Forecasting},
  author  = {Author Names},
  journal = {Under Review},
  year    = {2026}
}
```

*Replace with final published reference.*

---

## License

Released for academic and research purposes.