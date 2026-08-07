# exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 結果

## 結論

Kaggle train v1 は COMPLETE。`lgb_mean` pooled RMSE は 8.496204218351805 で、exp148 GPU train `lgb_mean` 8.50128118189582 から -0.005076963544015 の小改善だった。

ただし、この実験は `exp196 base + exp145/exp072-derived ll_*` の混合 provenance 診断であり、直接 inference / submit には使わない。改善幅が小さいため、exp196 pct40 surface から learned-likelihood `ll_*` も再生成する clean replacement follow-up は実施しない方針に変更した。

## 実験内容

- exp196 pct40 hard-window full replay cache を base 196 features として読む。
- projection_correction と u_disagreement は exp196 base candidate columns から再計算する。
- exp145 full-train learned-likelihood `ll_*` は exp148 と同じ cache から読み、active feature group に残す。
- active variant は `pct40_base_surface_keep_exp145_ll_mixed_provenance` のみ。
- control / parent 再学習はしない。exp148 の保存済み CV / Public LB を historical baseline として参照する。

## 注意点

これは `exp196 base + exp145/exp072-derived ll_*` の混合 provenance 診断であり、clean replacement ではない。`ll_candidate_tvt_*_minus_likpf_mean_tvt` などは exp145 candidate TVT と exp196 `likpf_mean` の差分になるため、feature meaning が exp148 と完全には一致しない。

OOF はわずかに改善したが、この実験を直接 inference / submit には使わない。clean replacement 実験も、追加実装コストに対する期待値が低いとしてバックログから外した。

## Kaggle 実行

- kernel: `kentookumura/exp199-pct40-base-keep-ll-train`
- version: 1
- status: `KernelWorkerStatus.COMPLETE`
- GPU: true
- internet: false
- kernel sources: `kentookumura/exp145-train`, `kentookumura/exp196-typewell-hard-window-pct40-train`
- output 確認先: `/tmp/kaggle-output/exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148/train_v1`

## CV

| model | pooled RMSE TVT | exp148 同 config 差分 | prediction SHA256 |
|---|---:|---:|---|
| `lgb0` | 8.551067730689416 | -0.048718128689474 | `ee91943810b9617de0d070113703d48df8d5aa07ee6b94c397674be49ed28703` |
| `lgb1` | 8.533458031963196 | -0.030513089266472 | `32c7d58d9a23617f2f6db1d58f9bddcf5c8a747db09abb29c0e5a010338bdad6` |
| `lgb2` | 8.570960612427667 | +0.061140893633592 | `bdbf7c76685f0436ec2cca82989908241f4db670c67d7849e05ce0f1cce4b527` |
| `lgb_mean` | 8.496204218351805 | -0.005076963544015 | `4e4ba51f815a8d64939c8b4acf4c91ef52af0666565b33a3b1de14f00fdf8585` |

`lgb0` / `lgb1` は改善したが、`lgb2` は悪化した。ensemble 平均はわずかに改善したため、base surface 差し替えの signal はあるが、単独で提出候補にする強さではない。

## Coverage / SHA

- rows / wells: 3,783,989 / 773
- features: 294
- boosters: 15
- elapsed_seconds: 12568.078
- feature join: base 3,783,989 rows、learned 3,783,989 rows、joined 3,783,989 rows、dropped rows 0、coverage pass
- exp196 base cache SHA256: gzip `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`, decompressed `106cdfb266f93a0e45f25b281d3238c1fab0a24a84dac4c23187044022b5127e`
- exp145 learned-likelihood cache SHA256: gzip `298fdafd7376d0d551083cac26491901658001ed717b4caa7a9f8b32103886ff`, decompressed `e1c276d69e9355f6c03c18ac51a0883ee99ec6d80d040a5c62e5d55048bb7456`
- model manifest SHA256: `516fe14fabd30c34dab8c85da2166e9ce0d0bc9ce629537975b2a9194f62ad21`
- feature schema SHA256: `da85f659658d3b50bb88aa863ceb89546dd99e361c669f79aa3dfe131259d944`
- summary SHA256: `9fd4d991255d64b38b9b1e969de929c6b24afac55139fe0412ea7e2266e54f63`

## Bucket / Worst Well

`lgb_mean` distance bucket RMSE は `000_050` 0.973144770、`050_100` 1.283747077、`100_250` 2.042684555、`250_500` 3.290816307、`500_1000` 4.804614067、`1000_plus` 9.319353104。

worst wells は `1b1eba53` 48.148292542、`86454a6f` 45.207756042、`fb03ae90` 42.261505127。feature importance 上位には通常の base geometry 系に加えて `ll_learned_pred_abs_error_beam_mean` と `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` が入り、混合 provenance の `ll_*` もモデルに使われている。

## 判断

exp199 は `completed_train_side_supported_no_submit` として閉じる。混合 provenance のまま raw/current-test inference port や submit には進めない。2026-07-05 のユーザー判断により、clean regeneration follow-up はバックログから削除した。
