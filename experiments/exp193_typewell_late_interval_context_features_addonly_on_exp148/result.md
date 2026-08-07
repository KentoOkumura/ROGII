# exp193_typewell_late_interval_context_features_addonly_on_exp148 結果

## 状態

Kaggle train v1 完了。train-side supported。Kaggle inference v2 完了、submit-check PASS。competition submit ref `54347471` は Public LB 7.946 で完了。exp148 GPU inference v7 の 7.960 は上回ったが、exp148 CPU runtime submission の 7.921 には届かないため非採用。

## 仮説

typewell 後半区間の min/max/span と observed prefix の `known_last_pct` は、exp176 で positive だった late-range prior を candidate 別 signal なしで弱く表現できる可能性がある。

## 評価設計

- `typewell_late_interval_context_addonly`: exp148 の feature surface に `typewell_late_interval_context` feature group を追加する。
- `exp148_fulltrain_control`: 再学習しない。保存済み exp148 metrics を historical baseline として参照する。
- GroupKFold 5 folds、well group、metric は RMSE。
- GPU runtime、3 LightGBM configs、5 folds、15 boosters。

## 結果

Kaggle train v1 は `KernelWorkerStatus.COMPLETE`。

- kernel: `kentookumura/exp193-typewell-late-context-exp148-train`
- URL: <https://www.kaggle.com/code/kentookumura/exp193-typewell-late-context-exp148-train>
- 実行内容: `typewell_late_interval_context_addonly` 1 variant、GPU LightGBM 3 configs x 5 folds = 15 boosters
- control / parent retraining: なし
- rows / wells / features: 3,783,989 / 773 / 313
- output: `kaggle/output/train_v1`

| model | pooled RMSE | exp148 同 config 差分 |
| --- | ---: | ---: |
| `lgb0` | 8.553543817 | -0.046242042 |
| `lgb1` | 8.475340902 | -0.088630219 |
| `lgb2` | 8.510015021 | +0.000195302 |
| `lgb_mean` | 8.456665439 | -0.044615743 |

best は `lgb_mean`。exp148 historical `lgb_mean` 8.501281182 から -0.044615743 改善した。

追加 `tlic_` feature は 19 個で、missing rate max は 0.0。feature importance では `tlic_known_last_pct` が rank 46 / 313、`tlic_known_last_to_late70_min_delta` が rank 89、late50/60 deltas も rank 109 / 114 に入り、context-only signal はモデルに使われている。

主要 SHA:

- prediction decompressed SHA256: `c171a0655ff3011e198d8b5ad1c74c5d3a8b9f086b09cd47dd613385133721dc`
- model manifest SHA256: `e8336b7a2058e584219750b26b30cb582802cf9b030e9ed728346230cd7d1e67`
- feature schema SHA256: `c762e6987be934ce6145ba954e2f60e68c5c36c7a54b2912253d610aac16fd80`

## 解釈

train-side では supported。exp160 と同じく CV positive が Public LB に転移しない可能性があるため、この結果だけで submit しない。

2026-07-05 に同じ exp193 内で inference port を作成し、Kaggle inference v2 を完了した。kernel id は `kentookumura/exp193-typewell-late-context-exp148-inference`。current test の horizontal/typewell input から `tlic_` 19 features を再生成し、exp193 train v1 の 15 saved boosters を `lgb_mean` として平均した。

v1 は `generator.candidates` が exp193 config に無く失敗したため、exp145/exp148 と同じ generator block を追加して v2 を再実行した。v2 は `KernelWorkerStatus.COMPLETE`、elapsed 116.88 sec。output は `kaggle/output/inference_v2`。

Inference v2 metrics:

- rows: test 14,151 / submission 14,151 / predicted 14,151
- feature count: 313 = base 196 + projection 69 + learned likelihood 54 + typewell context 19
- fallback rows: 0
- train manifest と inference feature schema: exact match
- prediction range: 11590.3720703125 - 12240.1171875
- prediction mean/std: 11905.43199423296 / 278.79483926833177
- prediction SHA256: `3567ebd4e48b1ab08e3b2ebf05dfa5061c65303e6f9081be262edd7940cbd0f8`
- submission SHA256: `9265e3e19e7eea20c6e0097b3b581b4a15c29353ebb77875d09ac30475502695`

Submit-check:

- PASS。fail 0、warn 0。
- `submission.csv` は sample と header / row count が一致し、ID 順も一致。重複 ID、NaN、Inf はなし。

Competition submit:

- ref: `54347471`
- submitted at: 2026-07-05 02:12:58.030000 UTC / 2026-07-05 11:12:58.030000 JST
- kernel: `kentookumura/exp193-typewell-late-context-exp148-inference` v2
- Public LB: 7.946
- Private LB: 未表示

exp148 GPU inference v7 Public LB 7.960 からは -0.014 改善した。一方、ユーザー確認済みの exp148 CPU runtime submission Public LB 7.921 には +0.025 届かないため、exp193 は ML route submitted anchor には採用しない。CV 改善量 -0.044615743 より LB 改善量は小さく、CV-to-LB 転移は控えめだった。アンサンブル route anchor の exp082 Public LB 7.601 は引き続き全体最良。
