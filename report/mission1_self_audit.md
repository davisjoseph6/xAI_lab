# Mission 1 self-audit

Ma became unavailable due to illness, so Da performed a self-audit.

## Checks performed

1. Verified that the selected applicant was truly and predicted bad risk.
2. Verified that the bad-risk probability column corresponds to model class 1.
3. Verified that XGBoost native categorical handling was disabled.
4. Verified that the trained booster contains no native categorical splits.
5. Verified SHAP additivity under both backgrounds.
6. Compared SHAP results under representative and good-only backgrounds.
7. Ran LIME with five different random seeds.
8. Checked LIME top-feature stability and sign stability.
9. Checked LIME surrogate score.
10. Compared SHAP and LIME at the original-feature level.
11. Ran feature sweeps to test whether disputed features actually changed the model output.

## Main audit findings

The implementation appears valid for the Mission 1 experiment. SHAP additivity errors were approximately 2.6e-08. SHAP was stable for the top three features across backgrounds. LIME was stable in top-three feature names, but its local surrogate score was low, so a single LIME plot should not be trusted as the main explanation. Direct feature sweeps supported the importance of duration_months, checking_status, and savings_status. Credit_amount was important in SHAP but showed a non-monotonic local response in the sweep.

## Remaining limitation

Because Ma could not independently inspect the results, this is a self-audit rather than a true peer audit. The report should state this explicitly.
