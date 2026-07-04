# Mission 2 summary

## Target

- Target class: Eskimo dog (248)
- Target quantity: pre-softmax class logit
- Image SHA-256: 48348d41080157af57aafd5e234cb62074619a1a61fca606e90036ce554769ea
- IG steps: 128
- IG black convergence delta: -1.4601373672485352
- IG blurred convergence delta: -0.03254079818725586

## Top-five predictions

- Eskimo dog: 0.2705
- Siberian husky: 0.1513
- malamute: 0.0890
- dogsled: 0.0105
- timber wolf: 0.0038

## Final-stage randomisation correlations

| method                       |   mean_spearman |   standard_deviation |   minimum_spearman |   maximum_spearman |
|:-----------------------------|----------------:|---------------------:|-------------------:|-------------------:|
| saliency                     |        0.161915 |           0.00679656 |          0.156948  |           0.169661 |
| gradcam                      |        0.174339 |           0.183966   |          0.0284486 |           0.381002 |
| integrated_gradients_black   |        0.330609 |           0.00468835 |          0.326042  |           0.33541  |
| integrated_gradients_blurred |        0.738993 |           0.0013765  |          0.737558  |           0.740302 |

## Dog/background interventions

| condition                  | replacement             |   target_logit |   target_probability |   target_logit_change |   target_probability_change |   predicted_class_index |
|:---------------------------|:------------------------|---------------:|---------------------:|----------------------:|----------------------------:|------------------------:|
| original                   | none                    |        6.35539 |           0.270475   |              0        |                   0         |                     248 |
| dog_region_replaced        | blurred_image           |        5.69346 |           0.105733   |             -0.661927 |                  -0.164742  |                     250 |
| background_region_replaced | blurred_image           |        7.04462 |           0.448048   |              0.689228 |                   0.177573  |                     248 |
| dog_region_replaced        | black_rgb_image         |        2.60814 |           0.00484158 |             -3.74725  |                  -0.265633  |                     223 |
| background_region_replaced | black_rgb_image         |        5.72189 |           0.17996    |             -0.633496 |                  -0.0905151 |                     248 |
| dog_region_replaced        | normalised_channel_mean |        3.6035  |           0.0153512  |             -2.75189  |                  -0.255124  |                     174 |
| background_region_replaced | normalised_channel_mean |        6.63689 |           0.35174    |              0.281503 |                   0.0812651 |                     248 |
