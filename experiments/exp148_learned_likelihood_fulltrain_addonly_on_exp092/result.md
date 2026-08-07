# exp148_learned_likelihood_fulltrain_addonly_on_exp092 結果

## 状態

Kaggle train v1 / inference v7 完了。GPU inference v7 は Public LB 7.960。CPU runtime inference v1 の提出 ref `54183122` は Public LB 7.921 で、現行の ML route submitted anchor として扱う。

## 仮説

exp127 の 155 wells subset で改善した learned likelihood confidence feature は、exp145 full-train feature cache を使えば exp092 full-row surface でも add-only feature として改善する可能性がある。

## 評価設計

- `learned_likelihood_confidence_addonly`: exp092 U-projection correction / disagreement feature surface に exp145 learned likelihood confidence feature を追加して学習する。
- `exp092_fulltrain_control`: 再学習しない。保存済み exp092 metrics を historical baseline として参照する。
- GroupKFold 5 folds、well group、metric は RMSE。
- 比較対象は exp092 full CV `lgb1` 9.322479896 / Public LB 8.350 と、exp127 subset delta `lgb_mean` -0.119735177。

## 結果

Kaggle train v1 は `learned_likelihood_confidence_addonly` のみを 15 boosters で学習した。rows は 3,783,989、wells は 773、features は 294。exp072/exp092 base rows と exp145 full-train learned likelihood features の join coverage は pass で、drop rows / wells は 0。

| model | pooled RMSE |
|---|---:|
| `lgb0` | 8.59978585937889 |
| `lgb1` | 8.563971121229669 |
| `lgb2` | 8.509819718794075 |
| `lgb_mean` | 8.50128118189582 |

best は `lgb_mean` の 8.50128118189582。保存済み exp092 `lgb1` CV 9.322479895503927 との historical 比較では -0.821198713608107 改善した。ただし control 再学習はしていないため、同一実行 ablation ではない。

Inference v5 は public raw-test cache を使って完了したが、code submission hidden rerun では hidden test が public test と異なるため `Notebook Threw Exception` になった。v7 では exp148 inference 内で current-test learned likelihood features を生成する形に修正し、public run は完了した。14,151 rows の `submission.csv` を生成し、fallback rows は 0。prediction SHA256 は `9a5f5d1030c357d8059c3c9ee2ba3a0578563ce11b9d02fe07906aa8b235d50b`、submission SHA256 は `45a8b1787fd80213c158d9af04fb596750d8025802d1328ab9d075432bcb6e4b`。`check_submission.py` は sample submission 互換で PASS。

提出 ref `54124882` は `SubmissionStatus.COMPLETE`、Public LB は 7.960。保存済み exp092 Public LB 8.350 から -0.390 改善した。

その後の CPU runtime inference v1 由来の提出 ref `54183122` は `SubmissionStatus.COMPLETE`、Public LB 7.921。GPU inference v7 の 7.960 からさらに -0.039 改善し、exp193 Public LB 7.946 も -0.025 上回る。ref の exp148 CPU runtime attribution はユーザー確認と Kaggle submissions table に基づく。

## 解釈

Full-row CV では learned likelihood confidence feature の add-only 効果は強く、exp127 subset / exp144 hidden-like stress で見えた signal は exp092 full-row surface でも残った。現時点では train-side positive。

Public LB は exp092 を明確に上回ったため、ML route submitted anchor を exp148 に更新する。現行の exp148 anchor 値は CPU runtime inference v1 の Public LB 7.921 とする。Control 再学習なしの historical 比較だが、ユーザー判断により追加 trust audit は不要とする。
