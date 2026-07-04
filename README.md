# Explainable AI Lab Report

## Team and scope

This report covers Mission 1 and Mission 2 of the Explainable AI lab.

The original team was Da and Ma. Ma became unavailable due to illness during the project, so Da completed the remaining experiments, self-audits, and final report writing. This is disclosed because the intended two-person audit structure could not be fully completed.

The central rule used throughout the work was:

> An explanation is a claim about a model, not a property of the data or the world. Therefore, it must be tested for faithfulness rather than accepted because it looks plausible.

---

## Mission 1: Why was she denied credit?

### Claim being tested

The claim tested in Mission 1 was:

> For one applicant rejected by the trained credit-risk model, SHAP and LIME identify which recorded input features most increased the model’s predicted probability of bad credit risk.

This is a claim about the trained model’s prediction, not a claim about the applicant’s real moral character, real financial reliability, or true causal reasons for default.

### Precommitment

The intended precommitment step was to predict, before viewing the explanations, which three features would most drive rejection and which explanation method would be more stable.

The exact precommitment text should be preserved in `report/precommitments.md`. Because Ma became unavailable and this final write-up is based on Da’s completed logs, this report does not invent a second independent precommitment after the fact.

The working expectation used for the analysis was that credit duration, checking-account status, and credit amount would be important drivers, and that SHAP would be more stable than LIME because LIME uses a stochastic local surrogate.

### Setup

The model was trained on the UCI Statlog German Credit dataset.

The raw target mapping was:

- `1` = good credit risk
- `2` = bad credit risk

The model target mapping was:

- `0` = good credit risk
- `1` = bad credit risk

Categorical variables were one-hot encoded before training. XGBoost native categorical handling was explicitly disabled. This was important because the SHAP experiment used interventional TreeSHAP with explicit background datasets.

Validation confirmed:

- model input shape: `(800, 61)`
- model input dtype: `float32`
- XGBoost native categorical handling: disabled
- native categorical split findings: none
- validated as purely numerical: true

The train/test split was stratified, using an 80/20 split with random seed `42`.

The selected applicant was chosen by the predeclared deterministic rule:

> Select the rejected test applicant with the lowest original row identifier.

This selected original applicant row `11`.

For this applicant:

- true class: `1`
- predicted class: `1`
- predicted bad-risk probability: `0.9179913997650146`

The model performance was:

- balanced accuracy: `0.7130952380952381`
- ROC-AUC for bad-credit class: `0.7994047619047618`
- confusion matrix, rows = actual and columns = predicted:

| Actual / Predicted | Predicted good | Predicted bad |
|---|---:|---:|
| Actual good | 125 | 15 |
| Actual bad | 28 | 32 |

The model is not perfect. Therefore, the explanation is an explanation of this imperfect classifier’s output, not proof that the applicant truly deserved rejection.

### Hidden choices

The main hidden choices were:

- the target class explained was the bad-credit-risk class;
- the output explained by SHAP was predicted bad-class probability;
- SHAP used interventional perturbation;
- SHAP was run with two different background datasets;
- LIME was run five times with different seeds;
- one-hot SHAP values were grouped back to original feature names before comparison with LIME.

The two SHAP backgrounds were:

1. a representative sample from the training data;
2. a good-risk-only background.

This matters because SHAP explanations are relative to a reference distribution. An attribution can move when the background changes.

### SHAP results

SHAP additivity was validated under both backgrounds.

For the representative background:

- base value: `0.29263533431318495`
- sum of grouped SHAP values: `0.62535603963048`
- reconstructed bad probability: `0.9179913739436649`
- actual bad probability: `0.9179913997650146`
- absolute additivity error: approximately `2.58e-08`

For the good-risk-only background:

- base value: `0.17721928860220174`
- sum of grouped SHAP values: `0.7407720849531044`
- reconstructed bad probability: `0.9179913735553061`
- actual bad probability: `0.9179913997650146`
- absolute additivity error: approximately `2.62e-08`

The top local SHAP features under the representative background were:

| Rank | Feature | SHAP value |
|---:|---|---:|
| 1 | duration_months | 0.160899 |
| 2 | checking_status | 0.134371 |
| 3 | credit_amount | 0.067631 |
| 4 | employment_since | 0.061700 |
| 5 | housing | 0.054793 |
| 6 | savings_status | 0.045696 |

All of these top features increased the predicted bad-credit probability for the selected applicant under the representative background.

Under the good-risk-only background, the top features were very similar:

| Rank | Feature | SHAP value |
|---:|---|---:|
| 1 | duration_months | 0.160758 |
| 2 | checking_status | 0.145943 |
| 3 | credit_amount | 0.104066 |
| 4 | employment_since | 0.069595 |
| 5 | savings_status | 0.066852 |
| 6 | housing | 0.059555 |

The top three features were identical across the two SHAP backgrounds:

1. `duration_months`
2. `checking_status`
3. `credit_amount`

The background-sensitivity metrics were:

- absolute-attribution Spearman correlation: `0.9624060150375938`
- top-three Jaccard overlap: `1.0`
- top-five Jaccard overlap: `0.6666666666666666`
- sign agreement fraction: `1.0`
- maximum absolute attribution change: `0.036435328406970405`

Therefore, SHAP was highly stable for the top three features, though the fifth-ranked feature changed depending on the background.

### LIME results

LIME was run five times with seeds:

- `11`
- `22`
- `33`
- `44`
- `55`

The top-three LIME features were perfectly stable across the five runs:

| Feature | Runs present | Top-three count | Top-five count | Median rank | Mean weight | Sign consistency |
|---|---:|---:|---:|---:|---:|---:|
| checking_status | 5 | 5 | 5 | 1.0 | 0.177213 | 1.0 |
| duration_months | 5 | 5 | 5 | 2.0 | 0.134704 | 1.0 |
| savings_status | 5 | 5 | 5 | 3.0 | 0.101784 | 1.0 |

The pairwise top-three Jaccard overlap across LIME seeds was always `1.0`.

However, the top-five Jaccard overlap was less stable, ranging from approximately `0.4286` to `1.0`.

The LIME surrogate scores were low:

- minimum: approximately `0.3363`
- mean: approximately `0.3518`
- maximum: approximately `0.3658`

This means that although LIME’s top-three feature names were stable, the local linear surrogate was a weak approximation of the model near this applicant. Therefore, a single LIME plot should not be treated as the legal or definitive reason for rejection.

### Background and seed tests

The SHAP background test asked whether the explanation changed when the reference population changed.

The answer was mixed but mostly reassuring:

- the top three SHAP features were unchanged;
- the signs were fully consistent;
- the rank correlation was high;
- lower-ranked features moved.

The LIME seed test asked whether the local explanation changed when only the random seed changed.

The answer was also mixed:

- the top three LIME features were completely stable;
- the top-five feature set was less stable;
- the surrogate score was low.

Therefore, SHAP was more defensible as the main explanation, while LIME was useful as a stability and disagreement check.

### Feature-sweep intervention

To test whether the disputed and top features actually changed the model output, I performed feature sweeps by replacing one feature at a time while holding the rest of the applicant fixed.

For `duration_months`, the bad-risk probability increased as duration increased:

| duration_months | bad probability |
|---:|---:|
| 9 | 0.659817 |
| 12 | 0.722891 |
| 18 | 0.805719 |
| 24 | 0.847863 |
| 36 | 0.911843 |

This supports SHAP and LIME identifying duration as an important driver.

For `checking_status`, the applicant’s current category had the highest bad-risk probability:

| checking_status | bad probability |
|---|---:|
| A11 | 0.917991 |
| A12 | 0.864023 |
| A13 | 0.796381 |
| A14 | 0.691536 |

This supports both SHAP and LIME identifying checking-account status as important.

For `savings_status`, the current category also had the highest bad-risk probability:

| savings_status | bad probability |
|---|---:|
| A61 | 0.917991 |
| A62 | 0.876582 |
| A63 | 0.843222 |
| A64 | 0.701028 |
| A65 | 0.769027 |

This supports LIME’s emphasis on savings status and SHAP’s lower but still positive attribution.

For `credit_amount`, the response was non-monotonic:

| credit_amount | bad probability |
|---:|---:|
| 914.4 | 0.928426 |
| 1353.0 | 0.909209 |
| 2317.0 | 0.864099 |
| 3933.0 | 0.834352 |
| 7064.0 | 0.901774 |

This means that although SHAP ranked credit amount highly, its local effect is not simply “larger loan amount always means higher risk.” The model has nonlinear interactions.

### Verdict

For the simulated rejection explanation, I would write:

> In this simulated model decision, the main recorded factors increasing the model’s estimated bad-credit-risk probability for this applicant were the credit duration, the checking-account status, and the credit amount, relative to the declared SHAP background population. LIME also consistently identified checking-account status and credit duration, but because the LIME surrogate score was low, I would not rely on a single LIME explanation as the definitive reason.

I trust SHAP more than LIME for the final explanation, because:

1. SHAP additivity was numerically validated;
2. SHAP’s top three features were stable across two backgrounds;
3. SHAP and LIME agreed on two of the strongest drivers;
4. LIME’s top-three features were stable across seeds, but its surrogate score was low;
5. feature sweeps supported the importance of duration and checking status.

The strongest caution is that this remains a model-relative explanation. It does not prove that these features are fair, legally sufficient, or real-world causal reasons for denying credit.

---

## Mission 2: Is the model looking at the dog or the snow?

### Claim being tested

The claim tested in Mission 2 was:

> A saliency map showing highlighted regions on the dog is faithful evidence that the trained ResNet-50 model used the dog, rather than only the snowy background, to classify the image as a husky-related ImageNet class.

This claim was tested using:

1. vanilla saliency;
2. Integrated Gradients with two baselines;
3. Grad-CAM;
4. progressive model-randomisation sanity checks;
5. dog-region and background-region interventions.

### Precommitment

The intended precommitment was to predict which saliency methods would survive the model-randomisation sanity check before viewing the results.

The exact precommitment text should be preserved in `report/precommitments.md`. Because the current logs do not show the full precommitment text, this final report does not invent an after-the-fact detailed prediction.

The working expectation was that visually persuasive maps might not all be faithful, and that at least one method could retain image-like structure even after the model weights were randomised. Therefore, the model-randomisation test and region intervention were treated as more important than visual inspection.

### Setup

The final image was a full-body husky sitting in snow.

The earlier image was replaced because it showed mainly the dog’s face and head, making the dog-versus-snow question less meaningful.

Final image metadata:

- file path: `images/husky.jpg`
- source URL: `https://pixabay.com/photos/siberian-husky-snow-dog-husky-291721/`
- licence / usage note: Pixabay Content License; attribution is not required by Pixabay, but the source URL was retained for transparency
- image SHA-256: `48348d41080157af57aafd5e234cb62074619a1a61fca606e90036ce554769ea`

The model was torchvision pretrained ResNet-50 with ImageNet weights.

The model’s top-five predictions were:

| Rank | Class | Probability |
|---:|---|---:|
| 1 | Eskimo dog | 0.2705 |
| 2 | Siberian husky | 0.1513 |
| 3 | malamute | 0.0890 |
| 4 | dogsled | 0.0105 |
| 5 | timber wolf | 0.0038 |

The fixed target class for all explanations and randomisation checks was:

- target class: `Eskimo dog`
- target class index: `248`
- target quantity: pre-softmax class logit

The target class was fixed during randomisation. I did not explain each randomised model’s new top class.

### Hidden choices

The main hidden choices were:

- target output: pre-softmax logit, not softmax probability;
- saliency map convention: signed gradients were saved, while maximum absolute channel value was used for spatial display;
- Integrated Gradients baselines: black RGB image and Gaussian-blurred original image;
- Integrated Gradients steps: `128`;
- Grad-CAM layer: `layer4[-1].conv3`;
- Grad-CAM ReLU attribution: enabled;
- Grad-CAM interpolation: bilinear;
- randomisation order: `fc`, `layer4`, `layer3`, `layer2`, `layer1`, `stem`;
- randomisation seeds: `101`, `202`, `303`;
- BatchNorm running statistics were reset;
- correlations were computed from raw numerical maps, not screenshots;
- dog/background intervention used a manually created `224 x 224` dog mask.

The final dog mask had a white-pixel fraction of approximately `0.237`, meaning the dog region covered about 23.7% of the model input crop.

### Saliency maps

The initial saliency maps were generated successfully:

- vanilla saliency;
- Integrated Gradients with black baseline;
- Integrated Gradients with blurred baseline;
- Grad-CAM.

However, visual inspection alone was not treated as evidence. The maps were tested using model randomisation.

### Model-randomisation sanity check

The sanity check progressively randomised the trained ResNet-50 from the classifier layer down to the stem. If a saliency method is faithful to the trained model, its map should change when the model weights are destroyed.

The final-stage correlations after randomisation through the stem were:

| Method | Mean Spearman correlation | Standard deviation |
|---|---:|---:|
| saliency | 0.161915 | 0.006797 |
| Grad-CAM | 0.174339 | 0.183966 |
| Integrated Gradients, black baseline | 0.330609 | 0.004688 |
| Integrated Gradients, blurred baseline | 0.738993 | 0.001377 |

The shuffled-map null correlations were approximately centred around zero for all methods, with 97.5th percentiles around `0.008` to `0.009`.

Therefore:

- vanilla saliency changed substantially, but did not go fully to the shuffled-map null level;
- Grad-CAM had low mean correlation after full randomisation, but high variance across randomisation seeds;
- Integrated Gradients with black baseline retained moderate similarity;
- Integrated Gradients with blurred baseline retained very high similarity even after full model randomisation.

This means Integrated Gradients with the blurred baseline should not be trusted as a faithful explanation of the trained model in this experiment. It appears to retain strong image- or baseline-driven structure even when the model has been randomised.

### Dog/background intervention

The dog/background intervention was the strongest test of the actual business question.

The original model output was:

- target logit: `6.355390`
- target probability: `0.270475`
- predicted class: `248`, Eskimo dog

The intervention results were:

| Condition | Replacement | Target probability | Probability change | Predicted class |
|---|---|---:|---:|---:|
| original | none | 0.270475 | 0.000000 | 248 |
| dog region replaced | blurred image | 0.105733 | -0.164742 | 250 |
| background region replaced | blurred image | 0.448048 | +0.177573 | 248 |
| dog region replaced | black RGB image | 0.004842 | -0.265633 | 223 |
| background region replaced | black RGB image | 0.179960 | -0.090515 | 248 |
| dog region replaced | normalised channel mean | 0.015351 | -0.255124 | 174 |
| background region replaced | normalised channel mean | 0.351740 | +0.081265 | 248 |

Replacing the dog region with black or channel-mean baselines sharply reduced the target probability and changed the predicted class. This is strong evidence that the dog region mattered to the model.

Replacing the background did not destroy the Eskimo-dog prediction. In two replacement settings, it actually increased the target probability. This suggests that the snowy background was not the sole basis for the prediction.

However, the background cannot be called irrelevant. With the black RGB replacement, replacing the background reduced the target probability from `0.270475` to `0.179960`. Therefore, the background still influenced the model to some extent.

### Verdict

The original statement:

> “The saliency map glows over the dog, therefore the model recognises the dog.”

is too strong if based only on a heatmap.

After the sanity check and interventions, the better conclusion is:

> The trained ResNet-50 prediction depended strongly on the dog region, but visual saliency alone was insufficient evidence. The dog-region intervention gave the strongest support: replacing the dog region sharply reduced the Eskimo-dog probability and changed the predicted class. The snowy background was not the sole reason for the prediction, although it may still have influenced the output.

I would trust the dog/background intervention more than the saliency maps.

Among the saliency methods, I would distrust Integrated Gradients with the blurred baseline most, because it retained a high correlation of approximately `0.739` even after full model randomisation. Vanilla saliency and Grad-CAM showed more model dependence, but they still do not by themselves prove that the model “understands” the dog.

---

## Self-audit note

The intended workflow was that Da and Ma would audit each other’s code and interpretation. Because Ma became unavailable due to illness, Da performed a self-audit instead.

The report therefore distinguishes completed checks from peer-validated checks. The experimental artefacts were generated, saved, and validated, but the independent second-person audit was not fully possible.

For Mission 1, the self-audit verified:

- selected applicant and class mapping;
- XGBoost categorical handling;
- SHAP additivity;
- SHAP background sensitivity;
- LIME seed stability;
- LIME surrogate score;
- SHAP–LIME comparison;
- feature-sweep results.

For Mission 2, the self-audit verified:

- final image replacement and hash;
- target class and target logit;
- saliency outputs;
- model randomisation order;
- original model not modified in place;
- BatchNorm reset;
- raw-map correlation calculation;
- dog-mask size and binary values;
- dog/background intervention outputs.

---

## Limitations

### Mission 1 limitations

The credit-risk model is imperfect, with balanced accuracy around `0.713`. Therefore, even faithful explanations only explain an imperfect classifier.

SHAP explanations depend on the chosen background distribution. The top three features were stable across backgrounds, but the lower-ranked features moved.

LIME’s top-three features were stable across seeds, but the local surrogate score was low. Therefore, a single LIME explanation should not be used as a definitive rejection reason.

The feature-sweep tests change one feature at a time and may create unrealistic combinations of applicant features. They are useful model probes, not real-world causal interventions.

### Mission 2 limitations

The dog mask was manually created. The intervention evidence depends on the quality of that mask.

Replacing image regions with black, blur, or channel mean can create out-of-distribution images. Therefore, the intervention results are stronger than visual inspection but still not perfect causal evidence about natural images.

The ResNet-50 ImageNet classes separate `Eskimo dog`, `Siberian husky`, and `malamute`, but the image is treated broadly as husky-related for this lab.

Because Ma became unavailable, the audit was a self-audit rather than a true independent peer audit.

---

## Files needed for reproduction

Important source files:

- `src/mission1_credit.py`
- `src/mission1_explanations.py`
- `src/mission2_saliency.py`
- `src/mission2_randomisation.py`
- `src/utilities.py`
- `tools/create_dog_mask.py`

Important Mission 1 outputs:

- `outputs/mission1/model_metrics.json`
- `outputs/mission1/selected_applicant.json`
- `outputs/mission1/xgboost_categorical_validation.json`
- `outputs/mission1/shap_additivity_checks.json`
- `outputs/mission1/shap_background_sensitivity.json`
- `outputs/mission1/shap_representative_background_beeswarm.png`
- `outputs/mission1/shap_representative_background_waterfall.png`
- `outputs/mission1/shap_good_only_background_beeswarm.png`
- `outputs/mission1/shap_good_only_background_waterfall.png`
- `outputs/mission1/lime_stability_by_feature.csv`
- `outputs/mission1/lime_pairwise_overlap.csv`
- `outputs/mission1/shap_lime_comparison.csv`
- `outputs/mission1/top_driver_feature_sweeps.csv`
- `outputs/mission1/disputed_feature_sweeps.csv`

Important Mission 2 outputs:

- `outputs/mission2/saliency_metadata.json`
- `outputs/mission2/original_saliency.png`
- `outputs/mission2/original_integrated_gradients_black.png`
- `outputs/mission2/original_integrated_gradients_blurred.png`
- `outputs/mission2/original_gradcam.png`
- `outputs/mission2/randomisation_summary.csv`
- `outputs/mission2/randomisation_curves.png`
- `outputs/mission2/shuffled_map_null_correlations.json`
- `outputs/mission2/dog_background_interventions.csv`
- `outputs/mission2/randomisation_methodology.json`

Important image files:

- `images/husky.jpg`
- `images/model_input_224.png`
- `images/dog_mask_224.png`
- `images/dog_mask_224_preview.png`

Environment file:

- `environment.txt`

The final submission should include the report, source code, environment file, selected outputs, and figures needed to support the claims.
