# ASTD 4-class dataset card

## Source files
- Tweets.txt (reference corpus) — `C:\Users\TECH ON\Desktop\Master\Second semester\NLP\New folder\AR Data\Tweets.txt`, 10,006 lines
- train ID list — `C:\Users\TECH ON\Desktop\Master\Second semester\NLP\New folder\AR Data\4class-balanced-train.txt`
- validation ID list — `C:\Users\TECH ON\Desktop\Master\Second semester\NLP\New folder\AR Data\4class-balanced-validation.txt`
- test ID list — `C:\Users\TECH ON\Desktop\Master\Second semester\NLP\New folder\AR Data\4class-balanced-test.txt`

## Indexing decision
- 0-based (verified via explore_astd4.py).
- Each line in a `4class-balanced-*.txt` file is treated as a 0-based line number into `Tweets.txt`. The 1-based reading was ruled out empirically in `src/explore_astd4.py`: only 0-based produced the documented `{POS, NEG, NEUTRAL, OBJ}` × 481 / 159 balance.

## Label mapping
| raw label | numeric id |
|:----------|-----------:|
| NEG | 0 |
| POS | 1 |
| NEUTRAL | 2 |
| OBJ | 3 |

## Final split sizes
| split | size | class distribution |
|:------|-----:|:-------------------|
| train | 1924 | {0: 481, 1: 481, 2: 481, 3: 481} |
| validation | 636 | {0: 159, 1: 159, 2: 159, 3: 159} |
| test | 636 | {0: 159, 1: 159, 2: 159, 3: 159} |

## Coverage and balancing
- Total tweets used across the three splits: **3,196** out of 10,006 available in Tweets.txt.
- Balancing rationale: the official splits are downsampled per class so that all four classes are equally represented in every split. The minority class in the raw corpus is **POS = 799**, which is the binding constraint on the size of the balanced design (train + val + test ≈ 4 × 799 minus held-out reserve).
- IDs that failed to map to a labelled tweet: **0** ({'train': 0, 'validation': 0, 'test': 0}). Zero is expected and required by the runtime assertions in `load_astd_4class`.

## Provenance of this file
Written automatically by `src/preprocess.py --astd4`. Re-run that command to regenerate.
