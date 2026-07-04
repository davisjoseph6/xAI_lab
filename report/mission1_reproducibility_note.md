# Mission 1 reproducibility note

Mission 1 was run successfully after retraining the XGBoost classifier on a dense numerical one-hot encoded matrix.

Key validation:
- Model input shape: (800, 61)
- Model input dtype: float32
- XGBoost native categorical handling: disabled
- Native categorical splits detected: no
- SHAP additivity error:
  - representative background: approximately 2.58e-08
  - good-only background: approximately 2.62e-08

The classifier selected original applicant row 11.
The applicant was truly class 1 and predicted class 1.
Predicted bad-risk probability: approximately 0.918.
