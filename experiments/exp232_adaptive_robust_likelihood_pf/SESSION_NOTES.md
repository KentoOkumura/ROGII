# exp232_adaptive_robust_likelihood_pf セッションノート

## 目的

`adaptive_robust_likelihood_pf` を temperature-only の本実験として実装する。exp072 likelihood-PF の観測モデルを、target-free gate が発火した row だけ `T=2/4` に緩める。outlier mixture は独立バックログ `adaptive_outlier_mixture_likelihood_pf` に分離する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_rejected_no_inference_no_submit`
- 親: `exp072_exp063_full_replay_feature_cache`
- 補助根拠: `exp214_public_raw_gr_residual_scale_control`、`exp200_pf_step_delta_soft_prior_full_replay_replacement`、`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: train-side full pseudo-tail audit 完了 / LB 未実行
- inference / submit: 不採用のため実施しない

## 実行設計（完了）

- 設定済み variant 数: 2（`temp_t2`、`temp_t4`）。各 parallel kernel の active variant 数は1で、全 eligible well を評価する。
- LightGBM config 数 / fold 数 / booster 数: 0 / 0 / 0
- PF: active variant を 500 particles x 128 seeds、全 eligible exp072 pseudo-tail wells で実行する。
- control: exp209 enriched cache の `hmm_mean_tvt - hmm_minus_likpf_mean` から exp072 `likpf_mean` を復元する。親/control の再生成なし。
- gate: high innovation に加え、GR change-point / GR novelty / pre-update low ESS / pre-update high max-weight の一つ以上を要求する。
- coverage audit: stable な 64 wells で 64 row stride と gate row の weighted particle p05-p95 を保存する。
- runtime: Kaggle CPU、GPU disabled、internet disabled、`num_workers=1`。`train_variant0` / `train_variant1` を並列実行する。

## 実装内容

- `adaptive_robust_likelihood_pf.py` に Numba の adaptive temperature likelihood-PF を実装。
  - 通常 row: `L = exp(-0.5 * residual^2)`。
  - gate row: `L = exp(-0.5 * residual^2 / T)`。
  - `T=2/4` 以外の尤度モデル、transition、resampling条件、particle数、seed数、seed mean aggregation は変更しない。
- 予測、ESS、resampling、gate seed fraction、innovation、change-point、novelty、pre-update collapse diagnostics、sampled interval、first sampled loss を保存する。
- exp115 fold assignment が利用可能なら hidden-like subgroup metrics を保存する。
- train notebook は入力・control・variant・禁止事項・出力をセル単位で確認する。inference notebook は明示的に停止する。

## 静的検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp232_adaptive_robust_likelihood_pf/adaptive_robust_likelihood_pf.py \
  experiments/exp232_adaptive_robust_likelihood_pf/exp232_adaptive_robust_likelihood_pf_train.py \
  experiments/exp232_adaptive_robust_likelihood_pf/exp232_adaptive_robust_likelihood_pf_inference.py \
  experiments/exp232_adaptive_robust_likelihood_pf/settings.py
.venv/bin/ruff check experiments/exp232_adaptive_robust_likelihood_pf --select F821
```

結果: 実装時点で `py_compile` と `ruff --select F821` は PASS。Kaggle 上の初回実行はまだ行っていない。

## 再現性メモ

- stochastic components: particle propagation / resampling。
- stable seed: `experiment + well + variant + public_likpf + seed index` を SHA256 で固定する。
- 単一 worker を使い、並列実行順序に乱数を依存させない。
- exp072 cache、exp209 enriched control、exp115 artifact、metric CSV、row candidate gzip の SHA（row candidate は decompressed content SHA を主証拠）を記録する。

## 次のアクション

1. temperature-only direct likelihood update は終了し、raw-test regeneration、inference、submit を行わない。
2. robust likelihood を再検討する場合だけ、gate event 後の cumulative path divergence を読む containment audit を先に実施する。

## 2026-07-11 Kaggle train 起動

- ユーザーから Kaggle 実行の明示依頼を受けた。
- 実行対象: `temp_t2` / `temp_t4` の 2 PF variant。
- LightGBM config / fold / booster: `0 / 0 / 0`。
- GPU: 無効、CPU-only。parent/control は再学習・再生成せず、v1 時点では exp072 saved `likpf_mean` を比較 control として読もうとした。
- canonical kernel: `kentookumura/exp232-adaptive-robust-likelihood-pf-train`
- title: `exp232 adaptive robust likelihood pf train`
- 実行前 static validation: `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp EXP=exp232_adaptive_robust_likelihood_pf` は PASS。
- package preflight: metadata の kernel id/title slug 一致、CPU/internet disabled、competition source、exp072/exp115 kernel source、bootstrap 内の helper/config/train script を確認した。
- runtime 見積り: exp072 full cache の likelihood-PF 生成約 15,380 秒を基準に、2 variant と interval readout を含めて約 9〜11 時間を見込む。Kaggle 12時間上限に近づく場合は、結果を見ずに同一exp内で variant を分割する判断を別途行う。
- push: `make push-kaggle-train EXP=exp232_adaptive_robust_likelihood_pf` は成功。Kaggle kernel version `1`、id_no `126646228`。
- metadata pull: canonical id/title、private、CPU (`enable_gpu=false`)、internet false、competition source、exp072/exp115 kernel source を確認した。
- initial status: `KernelWorkerStatus.RUNNING`。初期 `kaggle kernels logs` は空であり、実行中に空を返す既知挙動として扱う。

## 2026-07-11 Kaggle train v1 ERROR

- final status: `KernelWorkerStatus.ERROR`。
- first meaningful traceback: notebook `In [4]` の exp072 control contract check。
  `RuntimeError: The saved exp072 likpf_mean control is required for this experiment.`
- PF generation は開始していない（bootstrap 完了後、約155秒で停止）。GPU、memory、Numba、runtime timeout の失敗ではない。
- 原因: `exp063_full_replay_feature_cache_feature_schema.csv` に `likpf_mean` が存在しない。exp072 feature-cache 実装は ML model feature surface を保存する際、`likpf_scale_*` と `likpf_mean` を除外する設計である。そのため cache を row-level `T=1` PF control として読む今回の前提が誤っていた。
- 代替 control 候補: `kentookumura/exp209-joint-exact-parity-train` の `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz` は、`hmm_mean_tvt` と `hmm_minus_likpf_mean` を持つため `likpf_mean = hmm_mean_tvt - hmm_minus_likpf_mean` と復元できる。ただし exp209 cache は exp072 v2 との full artifact exact parity が未証明であり、その採用は設計判断を要する。
- 別案は同一 exp 内で T=1 / T=2 / T=4 をすべて再生成する方法だが、exp072単体約4.3時間の3倍に interval readout が加わり、Kaggle 12時間上限を超えるリスクが高い。
- ユーザー判断なしに control source / variant 数を変更して再 push しない。

## 2026-07-11 exp209 復元による v2 復旧

- ユーザー指定: 「exp209から復元する方法としてください」。
- 採用 control: `kentookumura/exp209-joint-exact-parity-train` の `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz`。
- 復元式: `likpf_mean = hmm_mean_tvt - hmm_minus_likpf_mean`。exp209 の direct comparison が保存した row-level difference を逆算する。
- 実装 guard: exp072 evaluation cache と exp209 control の `id` を one-to-one で結合し、`well`、`target`、`last_known_tvt`、`md_since` の一致を検査する。不足 ID、重複 ID、非有限の復元値、主要列不一致は実行前に例外にする。
- 使用範囲: 復元 `likpf_mean` は `exp072_likpf_mean` の比較列のみ。target-free gate、particle observation update、transition、resampling、interval の計算入力には使わない。
- caveat: exp209 enriched cache と exp072 v2 full artifact の exact parity は未証明である。したがって v2 は同一 comparison cache 内の T=1 reference に対する audit として記録し、source/reconstruction/alignment/SHA を Kaggle summary に残す。
- v2 実行対象は変更なし: `temp_t2` / `temp_t4` の 2 variants、LightGBM config / fold / booster は `0 / 0 / 0`、CPU-only、parent/control の再学習なし。
- 次アクション: Jupytext / static / experiment validation と package preflight を通し、canonical kernel `kentookumura/exp232-adaptive-robust-likelihood-pf-train` の v2 として push する。

## 2026-07-11 Kaggle train v2 起動

- static validation: `py_compile`、`ruff --select F821`、Jupytext `--to ipynb --test`、`make validate-exp EXP=exp232_adaptive_robust_likelihood_pf` は PASS。`make update-summary` も実行した。
- package preflight: canonical id/title slug 一致、private、CPU (`enable_gpu=false`)、internet false、competition source、exp072 / exp115 / exp209 kernel source、bootstrap 内の復元 helper/config/train notebook を確認した。
- push: `make push-kaggle-train EXP=exp232_adaptive_robust_likelihood_pf`。Kaggle kernel version `2` を成功として受理した。
- canonical metadata pull: 同じ kernel id から v2 metadata を取得した。
- status: `KernelWorkerStatus.RUNNING`。
- 実行対象は v1 から不変: `temp_t2` / `temp_t4` の 2 variants、LightGBM config / fold / booster `0 / 0 / 0`、CPU-only、parent/control の再学習なし。runtime 見積りも約9〜11時間で変更なし。
- 実行中の `kaggle kernels logs` が空でも既知挙動として扱い、同一 kernel id の完了または ERROR を根拠に後続記録を行う。

## 2026-07-11 Kaggle train v3 起動

- canonical kernel id を pull で確認後、同じ id に corrected package を push し、Kaggle kernel version `3` を受け取った。v3 を current run とする。
- current status: `KernelWorkerStatus.RUNNING`。`temp_t2` / `temp_t4` の2 variants、LightGBM config / fold / booster `0 / 0 / 0`、CPU-only、control / parent 再学習なし。
- v2 は同じ canonical history として残し、v3では slug を増やしていない。完了判定と結果は v3 の logs / notebook output を根拠にする。

## 2026-07-11 Kaggle train v3 中断（timeout 報告）

- ユーザーから timeout の報告を受け、canonical kernel
  `kentookumura/exp232-adaptive-robust-likelihood-pf-train` の最終 status を確認した。
  結果は `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。
- 最終 `kaggle kernels logs` は debugger warning と `Prepared 8 Kaggle support files from zip bootstrap.`
  までで、notebook cell の出力、PF progress、traceback はない。したがって利用可能な証拠は
  「実行が中断された」ことまでであり、Kaggle の時間上限か手動 cancel かは識別できない。
- PF metrics、row artifact、SHA は得られていない。control、gate、particle / seed、temperature
  variants は変更していない。
- 再実行は runtime 対策として variant 別または deterministic target-well shard 別に分ける必要が
  ある。どちらも比較意味を変えないが、実行構成の変更なのでユーザー判断後に実装する。

## 2026-07-11 split run 実装（並列化前の記録）

- ユーザー承認: timeout 対策として、同じ exp232 のまま実装・再実行する。
- 実行構成: 初期案は `temp_t2` の後に `temp_t4` を実行するものだった。この案は後段のユーザー承認により、同じ科学設定の並列2 kernel に置き換えた。T=1 control、gate、500 particles、128 seeds、transition、resampling、score rows は変更しない。
- cache reuse: notebook の input/control contract cell が読んだ exp072 / exp209 frame と provenance metadata を PF orchestration に渡し、同一 Kaggle run 内で2回目の cache load / SHA scan を行わない。
- checkpoint は Kaggle cancellation 後の `/kaggle/working` に永続化されないため撤去した。16 well ごとの completed well 数と elapsed seconds の flush のみ残し、最終 metrics/artifact は active variant ごとの `artifacts/runs/<active_variant>/` に保存する。
- 実行対象: active variant 1、LightGBM config / fold / booster `0 / 0 / 0`、CPU-only、GPU/internet disabled、parent/control の再学習なし。既存1 variantの実測基準から、cache 読込と診断を含めても12時間上限内を狙う。

## 2026-07-11 parallel variant split への更新

- ユーザー承認: `temp_t2` / `temp_t4` を順番ではなく並列で再実行する。
- package: `train_variant0` は `temp_t2` のみを `kentookumura/exp232-adaptive-robust-pf-t2`（title: `exp232 adaptive robust pf t2`）へ、`train_variant1` は `temp_t4` のみを `kentookumura/exp232-adaptive-robust-pf-t4`（title: `exp232 adaptive robust pf t4`）へ push する。
- selection: どちらの Jupytext notebook も `config.yaml` を deep-copy し、その notebook に宣言された predeclared variant だけをメモリ上で `execution.active_temperature_variants` に設定する。共有 config に default active variant は持たせない。
- semantics: T=1 exp209 reconstructed control、raw GR/typewell GR、gate、500 particles、128 stable seeds、transition、resampling、eligible well 面、interval audit は完全に同一である。変更点は kernel あたり1 variantという実行分割のみ。
- progress: checkpoint/resume は使わない。16 well ごとに progress を flush し、各 kernel が成功完了したときだけ `artifacts/runs/<variant>/` に最終 output を保存する。
- 実行対象: 2 parallel CPU kernels、各1 PF variant、LightGBM config / fold / booster は各 `0 / 0 / 0`。parent/control の再学習・再生成はない。

## 2026-07-11 parallel split kernels 起動

- `temp_t2`: `train_variant0` を `kentookumura/exp232-adaptive-robust-pf-t2` / title `exp232 adaptive robust pf t2` に push。Kaggle kernel version `1`、status `KernelWorkerStatus.RUNNING`。
- `temp_t4`: `train_variant1` を `kentookumura/exp232-adaptive-robust-pf-t4` / title `exp232 adaptive robust pf t4` に push。Kaggle kernel version `2`、status `KernelWorkerStatus.RUNNING`。
- 両 package preflight: canonical id/title slug 一致、private、CPU、internet disabled、competition source、exp072 / exp115 / exp209 kernel sources、variant 固定 assertion、cache一回読込を確認した。
- 実行中 package は16 wellごとの progress flush だけを行う。checkpoint/resume は Kaggle cancellation をまたいで永続化できないため使わず、完走時の最終 artifacts を正とする。

## 2026-07-12 split kernel 状態

- `kentookumura/exp232-adaptive-robust-pf-t2` v1 は `KernelWorkerStatus.COMPLETE`。
  `temp_t4` v2 は `KernelWorkerStatus.RUNNING`。t2 output は後続の同一 ID comparison と
  metrics/SHA 記録が必要になる時点で取得する。

## 2026-07-12 T=2 v2 output-recovery run

- v1 は 773 個の checkpoint row を Kaggle output に残し、output archive の保存ファイル数上限で
  final metrics / row candidate が取得できなかった。一方、Kaggle log と保存済み result / metrics の
  full-surface score は存在し、T=2/T=4 を不採用とする判断は既に確定している。
- checkpoint を撤去した同一 `temp_t2` notebook を canonical kernel v2 として再実行した。
  この run は row-level output の回収確認だけが目的で、実験の再判定、temperature 再探索、inference、
  submit を再開するものではない。

## 2026-07-12 split kernels 完了・train-side 不採用

- direct Kaggle status: `kentookumura/exp232-adaptive-robust-pf-t2` v1 と
  `kentookumura/exp232-adaptive-robust-pf-t4` v2 はともに `KernelWorkerStatus.COMPLETE`。
  全 773 wells / 3,783,989 score rows が `ok` で、traceback / OOM はない。
- T=1 comparison control は両 run とも exp209 enriched cache の
  `likpf_mean = hmm_mean_tvt - hmm_minus_likpf_mean`。control RMSE は 11.594897672、
  MAE は 7.067632584、within10 は 0.772807479 だった。
- `temp_t2`: runtime 29,602.282 sec、RMSE 13.529887109（control比 +1.934989437）、
  MAE 9.104082608、within10 0.671243230、`1000_plus` RMSE 14.775088743
  （control 12.704015215）、最大 well regression +45.905685193。全 seed gate は 685 rows、
  any-seed gate は 4,673 rows、sampled interval coverage は 0.219654231。
- `temp_t4`: runtime 34,207.477 sec、RMSE 13.532730350（control比 +1.937832678）、
  MAE 9.100793812、within10 0.671092331、`1000_plus` RMSE 14.778863970、
  最大 well regression +45.706170985。全 seed gate は 715 rows、any-seed gate は 4,764 rows、
  sampled interval coverage は 0.219930644。
- 両 variant は overall RMSE、`1000_plus`、exp115 hidden-like、worst-well guard を満たさず、
  coverage は control interval が未出力のため改善を主張できない。T=2 が T=4 より僅かに小さい
  overall regression だが、候補として採用しない。raw-test regeneration、inference、submit は行わない。
- provenance caveat: T=2 は exp072 input cache を direct exp072 Kaggle source、T=4 は exp209 copy
  から解決した。T=1 control 指標は一致するが、hidden-like 等の細かな subgroup 行数は一致しないため、
  variant 間の小差を exact parity evidence として扱わない。いずれも control比約 +1.93 RMSE のため、
  不採用判断はこの caveat に依存しない。
- 次候補は temperature の再 grid ではない。gate 発火後に resampling/seed aggregation で生じる
  累積 path divergence を event 単位で読む containment audit とし、将来の robust likelihood は
  長期回帰がないことを先に証明する。

## 2026-07-13 temp_t2 v2 output-recovery complete

- `kentookumura/exp232-adaptive-robust-pf-t2` v2 は `KernelWorkerStatus.COMPLETE`。checkpoint を
  出力しない recovery package により、773 wells / 3,783,989 rows の final row candidate、metrics、
  by-well、gate、interval、SHA を取得できた。runtime は 39,327.236s。
- `temp_t2` の RMSE 13.529887109、`1000_plus` 14.775088743、worst-well regression
  +45.905685193 は v1 に記録済みの結論と一致する。従って experiment decision は変わらず
  train-side 不採用である。
- v2 の exp072 input、exp209 reconstructed control、schema content SHA は exp233
  `mix_eps_0p05` v4 と一致した。ID-aligned 比較で mix e05 の RMSE 13.550173069 は T=2 より
  0.020285960 悪く、mixture の採否記録の根拠として使用した。
