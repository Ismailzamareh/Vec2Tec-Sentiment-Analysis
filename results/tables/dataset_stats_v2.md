# Dataset statistics (extended)

| dataset   | language   | text_type   |   n_classes |   train_size |   validation_size |   test_size |   vocab_size_train |   avg_doc_len_tokens |   median_doc_len_tokens |   max_doc_len_tokens | class_distribution_train         |   imbalance_ratio_train |
|:----------|:-----------|:------------|------------:|-------------:|------------------:|------------:|-------------------:|---------------------:|------------------------:|---------------------:|:---------------------------------|------------------------:|
| IMDB      | English    | long review |           2 |        25000 |               nan |       25000 |             159209 |                231.4 |                     173 |                 2473 | {0: 12500, 1: 12500}             |                   1     |
| LABR      | Arabic     | long review |           2 |        12755 |               nan |        3186 |              86677 |                 59.9 |                      31 |                 2904 | {0: 6383, 1: 6372}               |                   1.002 |
| ASTD      | Arabic     | short tweet |           2 |         1976 |               nan |         494 |              12273 |                 15.5 |                      16 |                   29 | {0: 1344, 1: 632}                |                   2.127 |
| ASTD4     | Arabic     | short tweet |           4 |         1924 |               636 |         636 |              12034 |                 15.6 |                      16 |                   29 | {0: 481, 1: 481, 2: 481, 3: 481} |                   1     |
