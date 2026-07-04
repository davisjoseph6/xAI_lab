# Mission 2 self-audit

Ma became unavailable due to illness, so Da performed a self-audit.

## Checks performed

1. Replaced the earlier face/head-only dog image with a full-body husky-in-snow image.
2. Recorded the final image SHA-256 hash.
3. Verified that ResNet-50 classified the image as a husky-related class.
4. Fixed the target class as Eskimo dog, class index 248.
5. Verified that the target quantity was the pre-softmax class logit.
6. Generated vanilla saliency, Integrated Gradients with black and blurred baselines, and Grad-CAM.
7. Ran progressive model randomisation from the classifier layer down to the stem.
8. Verified that the original model was not modified in place.
9. Verified that BatchNorm running statistics were reset during randomisation.
10. Verified that correlations were computed from raw numerical maps, not PNG screenshots.
11. Created a 224x224 dog mask aligned to the model input crop.
12. Verified that the mask was binary and had plausible coverage.
13. Ran dog-region and background-region interventions using blurred, black RGB, and normalised-channel-mean replacements.

## Main audit findings

The model-randomisation sanity check showed that not all saliency methods were equally faithful. Integrated Gradients with the blurred baseline retained high similarity even after full model randomisation, so it should not be trusted as a faithful explanation of the trained model. Vanilla saliency and Grad-CAM showed more model dependence, although neither should be treated as conclusive on visual evidence alone.

The dog/background intervention was the strongest evidence. Replacing the dog region with black or channel-mean baselines sharply reduced the target probability, while replacing the background had a smaller or even positive effect depending on the replacement. This supports the conclusion that the dog region mattered strongly to the model’s prediction, but the background was not irrelevant.

## Remaining limitations

Because Ma could not independently inspect the result, this is a self-audit rather than a true peer audit. The dog mask was manually created, so the intervention evidence depends on the quality of that mask. The replacement methods may also create out-of-distribution images, so the intervention results should be interpreted cautiously.
