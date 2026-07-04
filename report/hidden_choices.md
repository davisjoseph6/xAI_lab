# Hidden choices and defaults

## Mission 1

- Dataset: UCI Statlog German Credit dataset.
- Raw target mapping: 1 = good credit, 2 = bad credit.
- Model target mapping: 0 = good credit, 1 = bad credit.
- Train/test split: stratified 80/20 split with seed 42.
- Model: XGBoost binary classifier.
- Categorical handling: original categorical variables were one-hot encoded before training.
- Native XGBoost categorical splitting: disabled.
- Explained class: bad-credit-risk class.
- Selected applicant: rejected test applicant with the lowest original row identifier.
- Selected applicant: original row 11.
- Decision threshold: 0.5.
- SHAP output: predicted bad-class probability.
- SHAP perturbation mode: interventional.
- SHAP backgrounds: representative training background and good-risk-only training background.
- LIME seeds: 11, 22, 33, 44, 55.
- LIME sample count: 5000 per run.
- LIME discretizer: quartile.
- LIME explanation level: original features after grouping.

## Mission 2

- Model: torchvision pretrained ResNet-50, ImageNet weights.
- Final image: full-body husky sitting in snow.
- Image path: images/husky.jpg.
- Image SHA-256: 48348d41080157af57aafd5e234cb62074619a1a61fca606e90036ce554769ea.
- Target class: Eskimo dog, class index 248.
- Target quantity: pre-softmax class logit.
- Saliency convention: signed gradients saved; maximum absolute channel value used for spatial map.
- Integrated Gradients baselines: black RGB image and Gaussian-blurred original image.
- Integrated Gradients steps: 128.
- Grad-CAM layer: layer4[-1].conv3.
- Grad-CAM interpolation: bilinear.
- Grad-CAM ReLU attribution: enabled.
- Randomisation order: fc, layer4, layer3, layer2, layer1, stem.
- Randomisation seeds: 101, 202, 303.
- BatchNorm running statistics: reset during randomisation.
- Correlations: Spearman correlations computed from raw spatial arrays, not PNG files.
- Region mask: images/dog_mask_224.png, white = dog, black = background.
- Region replacements: blurred image, black RGB image, normalised channel mean.
