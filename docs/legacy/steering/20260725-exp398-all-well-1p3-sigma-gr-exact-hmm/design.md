# 設計

## アプローチ

exp397では`rho_gr < 0.50`の8 wellsだけを`1.3`倍するselectorが非退化・安定性gateを
FAILした。本実験はそのFAILを再分類せず、selector自体を使わない独立仮説として、
全wellのGR Gaussian evidenceを一定に弱める。

well `w`についてexp209のscaleをそのまま再現し、

```text
sigma_base = clip(std_population(fillna(GR_prefix, 0) - GR_typewell), 10, 60)
sigma_eff = 1.3 * sigma_base
log_emission = -0.5 * min(((GR_eval - GR_state) / sigma_eff)^2, 600)
```

とする。`sigma_eff`は再clipしない。保存済みexp209 predictionはload-only controlとし、
候補だけを773 wellsで再計算する。

## 実験範囲

- 対象実験: `exp398_all_well_1p3_sigma_gr_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: 全well共通`GR sigma multiplier = 1.3`
- 固定する変数: exp209 absolute-TVT state/rate grammar、transition、prior、
  Gaussian `z²` clip 600、missing-GR処理、Type Well補間、posterior mean
- 実行量: 1 variant / 773 HMM runs / 5 reporting folds /
  model・trained fold・booster・PF・Beam・parent control rerun各0
- baseline: saved exp209 direct RMSE `11.938287234887435`、
  fixed LikPF 50:50 RMSE `10.269696146642758`
- runtime見積: 過去の同一kernel familyからCPU約5〜8.5時間

## 再現性設計

- seed policy: RNGなし、well ID / raw row / grid / rate / variant順を固定
- stochastic処理: なし
- PF/Beam / seed bagging: 新規実行なし。saved LikPFはreporting controlのみ
- 並列処理: outer worker 1、Numba threads 2、乱数なし
- runtime: Kaggle CPU、GPU/internet off、上限30,600秒
- input: raw identity、exp209、exp072、exp226 fold、exp115 roleのSHAを確認
- prediction: truth join前にgzip raw/decompressed/logical SHAをfreeze
- model/submission: modelなし、inference/submission無効
- package: canonical metadataとbootstrap内config/sourceの一致をpush前に確認

## リスク

- リークリスク: unknown-suffix truth、error、hidden-like roleはprediction freeze後だけ読む。
- 科学リスク: 全wellでGRを弱めるため、GRが有効なwellのmode識別も悪化し得る。
- CV/LB不一致: train wellsで一律倍率が良くてもhidden testのGR品質分布は異なり得る。
- ランタイム: 773 exact-HMM runsは約5〜8.5時間。saved controlは再実行しない。
- 再現性: float reduction差を考慮し、input/prediction content SHAとruntime versionを保存する。
- 事後選択: `1.1/1.2/1.4`、再clip、well selector、emission family、blendは追加しない。
