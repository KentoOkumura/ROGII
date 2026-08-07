# exp235_fixed_lag_particle_smoother_pf セッションノート

## 2026-07-12 実装開始

- `fixed_lag_particle_smoother_pf` backlog を exp235 として実験化した。
- Route は `pf_beam`。親は exp072 likelihood-PF。既存 baseline/control の再学習・再生成はしない。
- active variants は `lag64` / `lag128` / `lag256` の3本。LightGBM config / fold / booster は `0 / 0 / 0`。
- forward transition、Gaussian likelihood、resampling、500 particles、128 seeds は固定。
- 各 row の pre-resampling particle position と ancestor map を `lag+1` ring buffer に保存し、`t+lag` で `t` の weighted smoothed mean を計算する。末尾 lag 行は forward fallback。
- future raw GR / typewell GR のみを利用し、future TVT、target/error、oracle、well横断情報は利用しない。
- Kaggle train-side audit 未実行。inference / submission は無効。

## 2026-07-12 Kaggle train 起動

- `lag64`: `kentookumura/exp235-fixed-lag-pf-lag64` version 1 を push、`KernelWorkerStatus.RUNNING`。
- `lag128`: `kentookumura/exp235-fixed-lag-pf-lag128` version 1 を push、`KernelWorkerStatus.RUNNING`。
- `lag256`: account-level `Maximum batch CPU session count of 5 reached` により push 未作成。専用 kernel は 404 で、空き CPU slot 後に同じ canonical ID/title で再試行する。
- 3 variant とも CPU-only、GPU/internet disabled、LightGBM config / fold / booster `0 / 0 / 0`。親 control の再学習・再生成はない。
- Kaggle CLI は version 2.2.0（2.2.2 upgrade warning のみ）。

## 2026-07-12 Kaggle train v1 ERROR の修正

- `lag64` / `lag128` の version 1 は bootstrap 後に notebook `In [5]` で ERROR。
- 原因は split notebook が `config_for_single_outlier_mixture_variant()` で active variant を1本に絞った後も、全3本の `lag64/128/256` が同時に見えることを要求していた guard の矛盾。
- PF generation、transition、likelihood、particle / seed / resampling、入力契約には到達しておらず、いずれも未実行。
- 修正: `train_variant0/1/2` はそれぞれ `lag64` / `lag128` / `lag256` の単一 active variant を厳密に要求するように変更。科学設定は不変。
- Jupytext `--test`、`py_compile`、`ruff --select F821`、strict experiment validation は再度 PASS。
- canonical kernel ID/title は維持し、lag64 / lag128 を同じ kernel へ version 2 として再 push する。lag256 は CPU slot が確保でき次第 version 1 として push する。

## 2026-07-12 Kaggle train v2 再起動

- `lag64`: canonical kernel `kentookumura/exp235-fixed-lag-pf-lag64` version 2 の push に成功し、`KernelWorkerStatus.RUNNING`。
- `lag128`: canonical kernel `kentookumura/exp235-fixed-lag-pf-lag128` version 2 の push に成功し、`KernelWorkerStatus.RUNNING`。
- v1 error は guard 修正前の package に限られ、v2 は同じ data source / PF settings / full eligible-well surface を維持する。

## 2026-07-12 lag256 起動試行

- lag64 / lag128 が `RUNNING` の状態で、`kentookumura/exp235-fixed-lag-pf-lag256` に train_variant2 package を push した。
- Kaggle API は `Kernel push error: Notebook not found` を返した。package の canonical ID/title は `exp235-fixed-lag-pf-lag256` / `exp235 fixed lag pf lag256` で一致している。
- 同じ canonical ID で1回だけ再 push したが、同じ `Notebook not found` が再現した。slug は変更しない。
- lag256 は Kaggle 側 notebook creation の回復、または既存 lag64 / lag128 の完了後に既存 canonical kernel を同一 exp の variant2 package へ切り替えて実行する必要がある。実行中 kernel は停止していない。

## 2026-07-12 lag64 / lag128 v2 停止と runtime 診断

- `lag64` / `lag128` の最終 status はともに `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。最終 `kaggle kernels logs` は bootstrap の `Prepared 11 Kaggle support files from zip bootstrap.` までで、PF progress / metrics / row artifact は残らなかった。
- Kaggle の12時間上限による自動終了か、GUIからの cancel かは status と logs だけでは区別できない。いずれにせよ採否に使える実行結果はない。
- 現実装は各 seed・各 row で全 particle の祖先を最大 lag 行だけ逐次 trace する。lag64 でも概算 `128 seeds × 500 particles × 3,783,989 rows × 64` の ancestor lookup が追加され、forward PF を大幅に超えるため、現形式の full surface は CPU 12時間枠に収まらない。
- 次 run には、(a) exact binary-lifting ancestry を導入して O(lag) trace を O(log lag) にする、または (b) smoothing seed / trajectory 数を事前固定で削減する、という実行意味の異なる判断が必要。ユーザー判断前に変更・再pushしない。

## 2026-07-12 exact lag64 four-shard 実行開始

- ユーザー選択: per-well PF semantics を変えない exact 4 deterministic well shard を採用。`well_shard_index = SHA256(experiment, well) mod 4`、shard単体のmetricsは採否に使わず、4 outputs をID/well/row_idxでstrict mergeする。
- 実装: target well selection、full/shard manifest、progress flush、shard-only status、`merge_fixed_lag_shards.py` を追加。transition / likelihood / 500 particles / 128 seeds / resampling は不変。
- static validation: Jupytext `--test`、`py_compile`、`ruff --select F821`、strict experiment validation は PASS。
- `shard1`: `kentookumura/exp235-fixed-lag-pf-lag64-shard1` version 1 をpush、`RUNNING`。
- `shard3`: `kentookumura/exp235-fixed-lag-pf-lag64-shard3` version 1 をpush、`RUNNING`。
- `shard0` / `shard2`: account-level `Maximum batch CPU session count of 5 reached` のため未作成。既存実行を停止せず、CPU slot が空いた時点で同じcanonical ID/titleへpushする。

## 2026-07-13 lag64 shard1 / shard3 v1 完走と logical shard0 / shard2 再利用

- `kentookumura/exp235-fixed-lag-pf-lag64-shard1` v1 は logical shard1 として COMPLETE。189 wells / 914,147 rows、runtime 12,139.836 sec。output は `kaggle/output/lag64_shard1_v1/` に保存した。
- `kentookumura/exp235-fixed-lag-pf-lag64-shard3` v1 は logical shard3 として COMPLETE。177 wells / 899,663 rows、runtime 13,266.105 sec。output は `kaggle/output/lag64_shard3_v1/` に保存した。
- shard-only metric は採否に使用しない。残り2 shardとの strict merge 前の参考値として、logical shard1 は `pf_lag64_mean` RMSE 12.764335 vs exp072 control 11.222902、logical shard3 は 13.796830 vs 11.548294。
- logical shard0 / shard2 専用 IDs は CPU slot 解放後も `Notebook not found` のまま。v1 outputs を取得・保持済みの shard1 / shard3 existing kernels を器として再利用した。
  - shard1 kernel v2 は source `train_variant0`（logical shard0）を実行中。
  - shard3 kernel v2 は source `train_variant2`（logical shard2）を実行中。
- kernel title と logical shard index が v2 では異なるため、採用時は notebook titleでなく `summary.execution.well_shard_index`、target manifest、row candidate IDs を正とする。

## 再現性

- stable seed は experiment / well / lag / seed index から SHA256 で生成する。
- stochastic components は particle propagation / systematic resampling。
- CPU-only、GPU/internet disabled、Numba single worker。
- row candidates と gzip 生成物は decompressed content SHA を記録する。

## 検証

- 実行前に `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp` を通す。
- Kaggle CPU train では overall、step-delta、particle coverage、first-loss、1000+、hidden-like、worst-well、runtime/memory を比較する。

## 2026-07-13 lag64 4 shard 完走・strict merge・不採用

- user の完了通知後に、logical shard0 を実行した
  `kentookumura/exp235-fixed-lag-pf-lag64-shard1` v2 と、logical shard2 を実行した
  `kentookumura/exp235-fixed-lag-pf-lag64-shard3` v2 がともに
  `KernelWorkerStatus.COMPLETE` であることを確認した。Kaggle logs はそれぞれ
  215 / 192 wells の progress 最終行と標準 artifact 出力を示す。
- v2 reused kernels の logical shard は title ではなく outputs の
  `summary.execution.well_shard_index` と manifests で確認した。
  - logical 0: 215 wells / 1,038,647 rows / 15,374.035 sec、shard1 kernel v2、source `train_variant0`
  - logical 1: 189 wells / 914,147 rows / 12,139.836 sec、shard1 kernel v1、source `train_variant1`
  - logical 2: 192 wells / 931,532 rows / 12,782.885 sec、shard3 kernel v2、source `train_variant2`
  - logical 3: 177 wells / 899,663 rows / 13,266.105 sec、shard3 kernel v1、source `train_variant3`
- `kaggle kernels output` で logical shard0 / shard2 outputs を取得した。これらは
  4 shard の候補 CSV、target-well manifest、decompressed SHA を strict merge するために必要な
  実ファイル確認であり、CV のみを理由に取得したものではない。
- 通常 command wrapper は merge v1/v2 の完了前に応答を返したため、当初は途中出力として見えた。
  後から v1（従来の一括 DataFrame）と v2（streaming merge）がともに完走していたことを確認した。
  v1/v2/v3 はすべて 3,783,989 rows / 773 wells と同一 decompressed SHA256 を出力した。
- 同じ検証意味を保ちつつ、10,000 rows chunk、on-disk SQLite の row-ID unique index、streaming gzip
  出力、online RMSE/MAE/within10 aggregation を `merge_fixed_lag_shards.py` に追加した。持続 PTY
  で監視した `lag64_merged_v3` を採用生成物とし、v1/v2 は同一性を確認する再現結果として保全する。
- strict merge は 4 sources の full target-well manifest 一致、selected well の重複なし・全被覆、
  全 row ID の重複なしを通過した。結果は 773 wells / 3,783,989 rows、merged decompressed SHA256
  `6376acff1762b438c0bf173da3fc8c3fc6feebad692d1e3b4eb2628b0c0ae0e5`。
- full-surface row-aligned score:
  - exp072 `likpf_mean`: RMSE 11.594897884、MAE 7.067632615、within10 0.772807479
  - `pf_lag64_mean`: RMSE 13.495447533（+1.900549649）、MAE 9.067266907、within10 0.673754866（-0.099052614）
- 結論: `lag64` は overall RMSE と within10 guard に不合格。raw-test inference / submission は行わない。
  `lag128` / `lag256` は未実行であり、同じ4-shard protocol を続ける CPU cost と、lag64 の明確な
  全体悪化を踏まえ、ユーザー判断待ちとする。

## 2026-07-13 高lag中止と悪化要因の追記

- ユーザー判断: `lag128` / `lag256` は実行しない。exp235 fixed-lag PF 枝は train-side
  不採用として閉鎖し、seed-paired 再監査、raw-test inference、submission も行わない。
- merged row candidates の distance readout を追加で確認した。lag64は全 bucket で悪化し、
  `000_050` +1.474550、`050_100` +1.095399、`100_250` +0.895861、`250_500` +0.929080、
  `500_1000` +1.599661、`1000_plus` +2.035222 RMSE。long-tailだけ、または最後64行のforward
  fallbackだけが主因ではない。
- ただし因果比較上の制約を確認した。exp235 は
  `stable_seed(EXPERIMENT_NAME, holdout.well, mixture.name, "public_likpf")`、frozen exp072
  cache は `stable_seed("likpf", "train", wid)` を使うため、forward PF乱数列が一致しない。
  各wellの最後64行（fixed-lag処理をせずexp235 forward estimateを残す49,472 rows）でも、
  exp235はRMSE 18.449056、frozen exp072は16.682135、差+1.766921だった。このsubsetは全体の
  約1.3%なので全体+1.900550を説明しないが、smoothing単独の効果を分離できていない。
- 実装candidateの採否は維持する。将来もし再検証するなら、同一 run に exp072 seed policyの
  forward candidateを保存し、同じ particle realizationで forward vs fixed-lagを比較する必要がある。
  今回はユーザー判断により実行しない。
- HMM comparison: exp205 / exp209 / exp221 の `run_hmm2` は既に alpha-beta の full
  forward-backward posteriorで、unknown suffixの全 future raw GR を使う exact smoother。
  HMMへfixed lagを加えるのは情報追加でなくbackward messageを切る近似/正則化であり、
  exp236でposterior meanがMAP / dominant-mode decoderより最良だった既存根拠もあるため、
  fixed-lag HMMを新規backlogには追加しない。
