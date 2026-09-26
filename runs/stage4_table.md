| Metric | Raw model* | Fine-tuned (ours, no help) | Change |
|---|---|---|---|
| Overall height error (MAE) | 1.72 m | **1.02 m** | 41% better |
| RMSE | 2.67 m | **1.96 m** | 27% better |
| Buildings/trees error | 2.62 m | **2.15 m** | 18% better |
| Ground error | 1.13 m | **0.27 m** | 76% better |
| Pixels within 1 m | 48.91 % | **69.16 %** | +20 pts |
| Pixels within 2 m | 71.06 % | **82.52 %** | +11 pts |

*Raw model got a per-tile scale fit using the true heights (best case for it). Like-for-like, fine-tuned with the same fit: MAE 0.97 m.
Validation: 620 tiles from 4 ISPRS Potsdam patches never used in training. Scale 0.1013 m per nDSM unit, derived from the official DSM.
