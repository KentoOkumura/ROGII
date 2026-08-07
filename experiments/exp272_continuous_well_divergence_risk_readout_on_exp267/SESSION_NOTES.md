# exp272_continuous_well_divergence_risk_readout_on_exp267 セッションノート

## 目的

exp267 の保存済み 18 次元 target-free well 署名を連続量としてだけ再監査し、exp264 OOF
candidate score の actual MAE / calibration bias と fold-stable な単調関係が残るか判定する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU train v1完了・primary guard FAIL・branch closed
- CV / LB: 新規予測を作らない diagnostic のため対象外
- inference / submission: disabled

## 実行量契約

- active variant: 0
- LightGBM config: 0
- trained fold: 0
- total booster: 0
- parent/control retraining: なし
- runtime: Kaggle CPU、GPU/TPU/internet off
- 入力: exp267 signature 1件、exp264 Stage B v2 candidate score 1件
- score集約対象: 6 primitive × 3,783,989 rows、45M-row Parquet は batch streaming

## 設計判断

- primary は score 非依存で事前固定した 12 range/gap 特徴の outer-train robust-scaled 等重み平均。
- PCA1 は全 18 特徴の sensitivity。符号は outer-train primary axis だけで固定し report-only。
- candidate-bank outcome は well 内で 6 candidates を等重み平均する。
- primary guard は actual MAE 正方向 5/5、calibration bias 負方向 5/5 と、fold-stratified
  well-bootstrap 95% 区間の actual lower `>=0.05` / calibration upper `<=-0.05`。
- PCA1 / candidate 別結果は primary guard を救済しない。

## 変更点

- exp267のKMeans / semantic cluster / soft membershipを除外し、連続軸だけをOOF生成する。
- exp264 scoreは6 primitiveをstreamingでwell×candidateへ集約する。
- new model / prediction / submissionは追加しない。

## 再現性

- `docs/06_reproducibility.md` 確認済み。
- stochastic component は bootstrap resampling のみ。
- bootstrap は scope 名から SHA256 stable seed を作り、local `default_rng` を single-thread で使う。
- PCA は `svd_solver=full`。axis fit・符号決定に outer-valid score を使わない。
- exp267 signature byte/logical SHA と exp264 score byte SHA を fail-closed 照合する。
- OOF axes / by-well logical SHA、preprocessor JSON SHA、全生成物 byte SHA を保存する。
- model / prediction / submission は生成しないため各 SHA は対象外。
- Kaggle rerun 前は deterministic anchor と呼ばない。

## コマンドログ

### 2026-07-17 実装

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py/inference.py>
.venv/bin/ruff check <exp272 sources and tests>
.venv/bin/pytest -q tests/test_continuous_well_divergence_risk.py
.venv/bin/pytest -q
.venv/bin/python scripts/validate_experiment.py --experiment exp272_continuous_well_divergence_risk_readout_on_exp267
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp272_continuous_well_divergence_risk_readout_on_exp267 --notebook train --kernel-id kentookumura/exp272-continuous-well-divergence-risk-readout-train --title 'exp272 continuous well divergence risk readout train' --strict
```

- 実装時のtargeted tests 4件、repository全107件: PASS。完了記録後の最終repository testsは
  全110件PASS。
- Ruff、py_compile、Jupytext train/inference round-trip、notebook JSON、strict experiment validation、
  template validation: PASS。
- 親exp267 trainは351行/7章、exp272 trainは378行/7章。入力・軸・stream集約・readout・
  生成物をnotebook上で追える構成を維持した。
- local notebook full実行とKaggle pushは行っていない。最初のfull readoutはKaggle CPUとする。

## Kaggle package監査

- canonical id/title:
  - `kentookumura/exp272-continuous-well-divergence-risk-readout-train`
  - `exp272 continuous well divergence risk readout train`
- private CPU、GPU/TPU/internet off、competition source 1、exp267 kernel source 1、
  exp264 immutable dataset source 1、`run_on_push=false`。
- canonical / loose package / bootstrap ZIPのSHA一致:
  - config: `03860917...ad4c`
  - settings: `c314583e...931b`
  - `src/continuous_well_divergence_risk.py`: `ce78f17b...4274`
- bootstrap manifest / ZIPは14 entriesで完全一致。
- canonical notebook SHAは`ab0f33b2...e980`、bootstrap付きpackage notebook SHAは
  `c205590a...21f`。
- package configも`run_approved=false`、0 booster、inference/submission disabledを維持する。

## 実行前アクション（完了）

1. 0-booster 契約を再確認した。
2. canonical Kaggle CPU train package を run-on-push on で再生成・監査した。
3. full readout を Kaggle で一度実行し、guard と生成物 SHA を記録した。

## Kaggle CPU readout実行承認

- 2026-07-17、ユーザーの「実行してください」によりcanonical Kaggle CPU readoutの
  実行承認を得た。
- 実行量を再確認した: 0 variant / 0 LightGBM config / 0 trained fold / 0 booster、
  親/control再学習0、GPU/TPU/internet off。
- 入力は保存済みexp267 version 2 signatureとimmutable exp264 Stage B v2 scoreだけで、
  PF/Beam再生成、selector学習、inference、submissionは行わない。
- credential checkerはOAuth credentialとlegacy credentialを利用可能と確認した。token実値は
  記録していない。
- canonical kernelの事前pullは403でresource未作成。別slugは作らず、canonical id/titleの
  初回versionとしてpushする。

## Kaggle CPU実行承認後package監査

- targeted tests 4件、Ruff、Jupytext round-trip、strict experiment validationはPASS。
- canonical id/titleは実装時と同じ。private CPU、GPU/TPU/internet off、competition source 1、
  exp267 kernel source 1、exp264 immutable dataset source 1、`run_on_push=true`。
- package configは`run_approved=true`、0 variant / 0 config / 0 trained fold / 0 booster、
  inference/submission disabled。
- canonical / loose package / bootstrap ZIPのSHA一致:
  - config: `779d70c6...aeee`
  - settings: `c314583e...931b`
  - `src/continuous_well_divergence_risk.py`: `ce78f17b...4274`
- bootstrap manifest / ZIPは14 entriesで完全一致。
- canonical notebook SHAは`ab0f33b2...e980`、bootstrap付きpackage notebook SHAは
  `6a155d38...07e1`。

## Kaggle push v1事前SaveKernel 400とslug修復

- `make push-kaggle-train EXP=exp272_continuous_well_divergence_risk_readout_on_exp267`
  はKaggle `SaveKernel 400 Bad Request`で停止し、notebook実行には到達しなかった。
- 失敗後も旧slug `exp272-continuous-well-divergence-risk-readout-train` のpullは403で、
  Kaggle resourceが作成されていないことを確認した。
- 旧id/titleは同じslugへ解決していたが52文字だった。Kaggle server-side validationを避けるため、
  意味を保った47文字のcanonical slug
  `exp272-continuous-divergence-risk-readout-train`へ短縮する。
- 新titleは`exp272 continuous divergence risk readout train`で、新idと同じslugへ解決する。
- 仮説、入力、primary/PCA軸、bootstrap、guard、0-booster契約は変更しない。同じexp272で
  packageを再生成し、別実験や複数slugを増やさない。

## Kaggle CPU train v1 完了

- canonical kernel:
  `kentookumura/exp272-continuous-divergence-risk-readout-train`
- kernel version 1 / id_no `127594096` / private CPU / GPU・TPU・internet off。
- 実行ログは約90秒で完了し、readout本体は約81秒。10 required artifactsをすべて生成した。
- 入力は773 wells / 5 folds / 18 signature features。exp264 scoreは91 batches / 190 row
  groupsでstreamし、6 candidates各3,783,989 rows、well×candidate 4,638 rowsへ集約した。
- 実行量は0 variant / 0 LightGBM config / 0 trained fold / 0 booster、親/control再学習0、
  inference/submissionなしで契約どおり。
- outputは`kaggle/output/train_v1/`へ取得し、manifest記載の全artifact byte SHAとlocal fileを
  照合して一致した。

## Readout と guard

- primary actual MAE Spearmanはfold 0--4で
  `0.710570 / 0.773823 / 0.814784 / 0.801434 / 0.826475`、pooled `0.785818`。
  fold方向は正5/5、5,000回bootstrap 95% intervalは`[0.749473, 0.817829]`でguard PASS。
- primary calibration bias Spearmanは
  `0.123895 / 0.099191 / 0.054332 / 0.045484 / -0.112217`、pooled `0.040968`。
  負方向は1/5、bootstrap 95% intervalは`[-0.032918, 0.115824]`でguard FAIL。
- primary quantileのactual MAEはq0 `3.386843` ftからq9 `15.940152` ftまで上昇し、
  divergenceがactual-error risk軸であることは強く再現した。一方、事前固定した
  calibration低下方向は再現しなかった。
- report-only PCA1もactual MAE `0.781531 [0.744880, 0.814806]`、calibration bias
  `0.050812 [-0.024266, 0.126782]`で、primary calibration guardを救済しない。
- 総合primary continuous-risk guardはFAIL。`separate_add_only_candidate_supported=false`とし、
  exp267 K=3 branchをclosedのまま維持する。clip/subset/candidate別の事後grid、学習、
  raw-test inference、submissionは行わない。

## 再現性記録

- exp267 signature byte/logical SHA:
  `7c452a927a106dc837e81fa0cff629b04ad0c91cfa95120db1fb38008179d9b2` /
  `bb193a6a540675a469c52c2cb8abb572a2e5827a2bec0ee6262b866ca593decf`
- exp264 candidate score SHA:
  `e51bb6747c71ed38a550b72a119558241220bdddab3fa9bdc2a21e0587445a5a`
- axes / by-well logical SHA:
  `04629ae0f4118ff965305be81753c90fd023d30182cbb05bea10cbc298672682` /
  `e00c071d52feb705f38c5eb305d249536a8d20424e5f9886a0f332374b4198a0`
- preprocessor SHA:
  `a62efc0262e913ab7d61caabbdb49336e2ac0efb632b6422f852fd6d86fc0ef5`
- artifact SHAは`kaggle/output/train_v1/artifacts/reproducibility_manifest.json`を正とする。
  model / prediction / submission SHAは対象外、rerun未実施のためdeterministic anchorはfalse。
- 完了後、local `config.yaml`の`execution.run_approved`をfalseへ戻した。

## 完了後package lock

- accidental repushを防ぐため、同じcanonical id/titleでpackageを`run_on_push=false`へ再生成した。
- canonical / package configは`run_approved=false`、statusは
  `kaggle_cpu_readout_complete_guard_failed_branch_closed`で一致し、SHAは
  `5574534d250eeabf3fed94e4d079d27e04b6465360493f53343116372abcba4f`。
- canonical notebook SHAは`ab0f33b2...e980`、locked package notebook SHAは
  `f5b29352...11a7`。実際に実行したrun-on-push packageは取得済み
  `kaggle/output/train_v1/`と上記の実行前監査記録を証跡とする。
