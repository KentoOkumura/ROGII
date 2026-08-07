# exp198_exact_replacement_prune_on_exp148 結果

## 現在の状態

Kaggle train v1、inference v4、scoring が完了。Public LB は 7.930。exp148 GPU inference v7 の 7.960 は上回ったが、現 ML route anchor の exp148 CPU runtime 7.921 には届かないため未採用。

## 実験内容

exp148 の active model feature list から、`corr_prune_sanity_readout_on_exp148` で高信頼の exact replacement / sign-flip / constant duplicate と判断した 17 列だけを削る。新規特徴量、direct TVT replacement、blend、postprocess、submit はこの初回実装の範囲外。

## 評価

比較基準は exp148 GPU train `lgb_mean` CV 8.50128118189582 / Public LB 7.960。Kaggle train package の pushed config SHA は loose package / bootstrap で `3f2972a11b64971923186aa301e011904d17a260e08550f8296008fadb457e5f` に一致した。Kaggle train v1 は `kentookumura/exp198-exact-replacement-prune-exp148-train` version 1 として完了した。

| model | pooled RMSE | exp148 同一 model 比 |
|---|---:|---:|
| `lgb0` | 8.525098952 | -0.074686907 |
| `lgb1` | 8.531602621 | -0.032368501 |
| `lgb2` | 8.476691203 | -0.033128516 |
| `lgb_mean` | 8.457923653 | -0.043357529 |

feature count は 277 で、exp148 の 294 features から指定 17 列だけが落ちた。feature join coverage は 3,783,989 rows / 773 wells、dropped rows 0 / dropped wells 0 で pass。削除対象 17 列が feature schema に残っていないことも確認した。

distance bucket では `000_050` が -0.019642711、`050_100` が -0.017495751、`1000_plus` が -0.050682068 改善した。一方 `100_250`、`250_500`、`500_1000` はそれぞれ +0.002599955、+0.016456604、+0.010155678 と小幅悪化。well 単位では 423 wells 改善、350 wells 悪化、最大悪化は `b37fd114` の +1.022149086 RMSE、最大改善は `86454a6f` の -1.425857544 RMSE。

生成物は `/tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/train_v1` に取得済み。feature schema SHA256 は `c9827f1a2fbec34e039035cab121b56077e56be8e1c3a74a7624ba205566c833`、model manifest SHA256 は `f286ae46c6e47a66793ea2e4668e8569ef79fa6f896c19610c311ed1ff1c54d8`、OOF prediction gzip SHA256 は `60e0756f6c137de676afac20686c5fa326214898cb4599f66d0d2f6690dc238e`、decompressed prediction SHA256 は `816dc0883b4920d7ece1ed63cc719dc11ae88dcb72d80a51ee2644076b41d381`。

## 推論と提出前チェック

Kaggle inference v4 `kentookumura/exp198-exact-replacement-prune-exp148-inference` は CPU runtime / internet off で完了した。実行時間は 155.668 秒、current-test feature replay は 101.484 秒。likelihood-PF replay は 14,151 / 14,151 rows を生成し、current-test learned likelihood features は 14,151 rows / 3 wells / 51 columns。LightGBM は `drop_exact_replacements_17` / `gpu_repro_guard_dp_threads8` / `lgb_mean` の 15 boosters を読み、277 features で推論した。fallback rows は 0。

提出ファイルは `/tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/inference_v4/submission.csv` に取得済み。`sample_submission.csv` と header、行数、id 順序が完全一致し、重複 ID、欠損、NaN、Inf-like value はなかった。prediction range は 11590.4658203125 から 12240.234375、mean 11905.457642403806、std 278.82843661424334。prediction SHA256 は `e23bd8f8e59b56fe188833849075e1ce146ced28c2810ab0bd1ea0b42948944c`、submission SHA256 は `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`。

## 次アクション

## Scoring

submission ref `54354847` は `SubmissionStatus.COMPLETE`、Public LB は 7.930。submission SHA256 は `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`。exp148 GPU inference v7 の 7.960 からは -0.030 改善し、exp193 の 7.946 からも -0.016 改善した。一方で exp148 CPU runtime inference の 7.921 からは +0.009 悪化した。

## 次アクション

drop-only exact replacement prune は CV 改善が LB に一部転移したが、現 ML route submitted anchor は更新しない。次は exp198 の exact-prune 効果を維持しつつ、CPU runtime anchor との差を埋められる follow-up を検討する。
