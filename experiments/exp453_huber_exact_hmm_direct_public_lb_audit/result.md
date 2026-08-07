# exp453_huber_exact_hmm_direct_public_lb_audit 結果

## 状態

設計確定、実装未着手。Kaggle実行、submission生成、competition submissionはない。

## 仮説

fixed `delta=1.345` Huber emissionのfold-stable OOF改善が、少数wellの
tail悪化よりPublic LBへ強く現れる可能性を記述評価する。

## 設定

- 親: `exp389_exp209_huber_exact_hmm_emission`
- 候補: `huber_delta1p345_on_exp209_absolute_tvt`
- 検証: frozen train-side evidenceからdirect Public LBへのcensus
- メトリック: RMSE
- シード: no RNG
- 学習 / model / booster: `0 / 0 / 0`

## 既存train-side証拠

| メトリック | 値 |
| --- | ---: |
| Gaussian exact HMM RMSE | 11.938287235 |
| Huber exact HMM RMSE | 11.852741130 |
| 改善 | 0.085546105 |
| 改善fold | 5/5 |
| by-well delta p95 | +0.002234351 |
| worst-well delta | +1.750248202 |
| Public LB | - |

## 再現性

- deterministic anchor: まだ主張しない
- seed policy: no RNG、fixed sorted well/row/grid/rate order
- kernel version: 未実行
- feature content SHA: 未実行
- model SHA / manifest SHA: 非該当
- prediction SHA: 未実行
- submission SHA: 未実行
- rerun result: 未実行

## 解釈

結果はまだない。Public LBはexp434 `exact_hmm`との記述比較にだけ使い、
train-side tail FAILを自動で覆さない。

## 次

別承認後にのみ、凍結済み設計のinference Notebookを実装する。
