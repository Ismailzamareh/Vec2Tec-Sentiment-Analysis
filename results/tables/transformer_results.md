# Transformer baseline (frozen multilingual MiniLM + LR head)

Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (no fine-tuning).

| dataset   | model          | encoder                                                     |   dim |   n_train |   n_test |   accuracy |   precision_macro |   recall_macro |   f1_macro |   f1_weighted |   encode_train_s |   encode_test_s |   train_time_s |   predict_time_s |   seed |
|:----------|:---------------|:------------------------------------------------------------|------:|----------:|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|-----------------:|----------------:|---------------:|-----------------:|-------:|
| astd      | transformer_lr | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |   384 |      1976 |      494 |   0.777328 |          0.751484 |       0.777633 |   0.75848  |      0.782791 |          103.114 |         26.8336 |         0.2101 |           0.004  |   2024 |
| astd4     | transformer_lr | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |   384 |      1924 |      636 |   0.462264 |          0.46774  |       0.462264 |   0.463464 |      0.463464 |          105.25  |         34.8266 |         0.6593 |           0.0076 |   2024 |
| imdb      | transformer_lr | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |   384 |      5000 |     5000 |   0.7754   |          0.775403 |       0.775404 |   0.7754   |      0.7754   |          889.406 |        841.386  |         0.2344 |           0.0294 |   2024 |
| labr      | transformer_lr | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |   384 |     12755 |     3186 |   0.751726 |          0.751761 |       0.751714 |   0.751711 |      0.751715 |         1255.74  |        328.918  |         1.064  |           0.0209 |   2024 |
