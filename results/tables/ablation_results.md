# Vec2Tec ablation study

Single-seed run (seed = 2024). Underlying Word2Vec model and corpus lexicon are held fixed; only the enhancement stack is varied.

Deltas are vs. `word2vec_plain` (mean of word vectors, no enhancement) on the SAME dataset.


## ASTD

| variant        |   accuracy |   f1_macro |   f1_weighted |   train_time_s |   predict_time_s |   delta_accuracy_pp |   delta_f1_macro_pp |
|:---------------|-----------:|-----------:|--------------:|---------------:|-----------------:|--------------------:|--------------------:|
| v2t_synonym    |   0.712551 |   0.682491 |      0.717692 |         0.5151 |           0.002  |              7.6923 |              9.5811 |
| v2t_lexicon    |   0.676113 |   0.626529 |      0.675562 |         0.2615 |           0.003  |              4.0485 |              3.9849 |
| v2t_full       |   0.621457 |   0.602824 |      0.633822 |         0.2329 |           0.003  |             -1.4171 |              1.6144 |
| word2vec_plain |   0.635628 |   0.58668  |      0.637931 |         0.1715 |           0.0054 |              0      |              0      |
| v2t_contextual |   0.635628 |   0.58668  |      0.637931 |         0.182  |           0.002  |              0      |              0      |

## IMDB

| variant        |   accuracy |   f1_macro |   f1_weighted |   train_time_s |   predict_time_s |   delta_accuracy_pp |   delta_f1_macro_pp |
|:---------------|-----------:|-----------:|--------------:|---------------:|-----------------:|--------------------:|--------------------:|
| v2t_full       |    0.86932 |   0.86932  |      0.86932  |         5.4238 |           0.144  |               1.108 |              1.1083 |
| v2t_lexicon    |    0.86588 |   0.86588  |      0.86588  |         1.9777 |           0.0421 |               0.764 |              0.7643 |
| v2t_contextual |    0.8594  |   0.859398 |      0.859398 |         6.4856 |           0.1255 |               0.116 |              0.1161 |
| word2vec_plain |    0.85824 |   0.858237 |      0.858237 |         2.0431 |           0.0594 |               0     |              0      |
| v2t_synonym    |    0.84424 |   0.844225 |      0.844225 |         6.0288 |           0.1224 |              -1.4   |             -1.4012 |

## LABR

| variant        |   accuracy |   f1_macro |   f1_weighted |   train_time_s |   predict_time_s |   delta_accuracy_pp |   delta_f1_macro_pp |
|:---------------|-----------:|-----------:|--------------:|---------------:|-----------------:|--------------------:|--------------------:|
| v2t_full       |   0.789705 |   0.789705 |      0.789705 |         4.8774 |           0.0148 |              1.1927 |              1.1932 |
| v2t_lexicon    |   0.787508 |   0.787504 |      0.787503 |         6.1664 |           0.0235 |              0.973  |              0.9731 |
| word2vec_plain |   0.777778 |   0.777773 |      0.777775 |         5.0572 |           0.012  |              0      |              0      |
| v2t_contextual |   0.776836 |   0.776832 |      0.776834 |         5.1262 |           0.013  |             -0.0942 |             -0.0941 |
| v2t_synonym    |   0.763026 |   0.763017 |      0.76302  |         5.3782 |           0.0176 |             -1.4752 |             -1.4756 |
