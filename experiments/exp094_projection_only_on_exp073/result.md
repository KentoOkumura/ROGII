# exp094_projection_only_on_exp073 結果

## 仮説

exp073 full replay prediction に `TVT + Z - prefix_anchor` projection を後処理として入れると、再学習なしで OOF RMSE を改善できる可能性がある。一方で、真の急変や short tail を平滑化するリスクがあるため、全体 RMSE だけでは採用しない。

## 設定

- 親: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 検証: exp073 OOF prediction に対する target-free projection postprocess audit
- メトリック: RMSE
- シード: 42。ただし exp094 自体に RNG はない。
- 入力: exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` OOF prediction
- grid: degree 3/4/5、beta 0.25/0.50/0.75、robust C 1.25/1.5/2.0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 9.399456024 |
| Public LB | - |
| Private LB | - |

Kaggle train v1:

- Kernel: `kentookumura/exp094-projection-only-on-exp073-train` v1
- Output: `/tmp/kaggle-output/exp094_projection_only_on_exp073/train_v1`
- Rows / wells: 3,783,989 / 773
- Runtime: 3412.934 秒
- Baseline exp073 OOF RMSE: 9.526374817
- Best variant: `degree4_beta0.75_c2`
- Best RMSE: 9.399456024
- Delta vs baseline: -0.126918725
- Guard: failed

## 再現性

- deterministic anchor: false。train-side postprocess audit であり submission anchor ではない。
- seed policy: `no_new_rng_projection_postprocess`
- kernel version: `kentookumura/exp094-projection-only-on-exp073-train` v1
- input content SHA: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- model SHA / manifest SHA: 対象外
- prediction SHA: best `bc4f02808ae1fd1cc0a174ee558cfee462734961ae51ef9a65a1018c38889200`
- submission SHA: inference 未選択
- rerun result: 未実施

## 解釈

Projection-only は global OOF では強く、best は exp073 から -0.126919 RMSE 改善した。一方で採用 guard は落ちた。best variant の fold delta はすべて改善側だが、distance 0-50 ft が +1.439466、tail rank 0-99 が +1.130416 悪化しており、prefix 直後の continuity を壊している。

全 variant を確認しても `passes_guard` は 0 件だったため、固定 projection-only policy として inference port しない。projection の価値は long-tail 側に偏っており、続けるなら near-prefix を除外した long-tail-only / confidence-gated projection として切り直す。

## 次

1. exp094 は `completed_no_inference_guard_failed` として閉じる。
2. `projection_only_on_exp073` backlog は完了扱いで削除する。
3. follow-up は `projection_confidence_error_map` または long-tail-only projection gate に限定し、global projection submit はしない。
