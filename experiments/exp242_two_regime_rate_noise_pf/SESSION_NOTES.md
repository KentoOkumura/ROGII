# exp242_two_regime_rate_noise_pf セッションノート

## 目的

exp072-compatible likelihood-PFのparticle stateを`(position, rate, regime)`に拡張し、
stickyな`smooth / turn`の2状態だけでtransition model mismatchを吸収できるか検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle train v2完了・不採用
- 親: `exp072_exp063_full_replay_feature_cache`
- 実装参照: `exp232_adaptive_robust_likelihood_pf`
- inference / submission: guard不通過のため実施しない

## 実行設定

- active variant: `two_regime_k4` 1件
- PF: 500 particles x 128 seeds x 全eligible wells
- LightGBM config / fold / booster: `0 / 0 / 0`
- control: exp209 enriched cacheから復元する保存済みexp072 `likpf_mean`
- control再学習・PF再生成: なし
- GPU: なし、Kaggle CPU、internet disabled、single worker
- Kaggle push: ユーザー承認後に実行済み

## 実装内容

- regime 0=`smooth`、1=`turn`の`int8` stateをparticleごとに保持する。
- 各rowのpropagation前に固定Markov transitionをsampleする。
- matrixは`[[0.9998, 0.0002], [0.02, 0.98]]`。
- 各seedの初期500粒子を`495 smooth / 5 turn`に固定し、local stable RNGでshuffleする。
- smoothは`velocity_noise=0.002`、turnだけ`0.008`を使う。
- resamplingではancestorのposition/rate/regimeを一緒にcopyする。
- Gaussian GR likelihood、momentum、position noise、ESS threshold、resampling jitterは固定する。
- row-levelのturn particle fraction、posterior mass、entry/exit/switch fractionを保存する。

## 禁止事項

- continuous acceleration
- position-noise変更
- true TVT/error/oracle/GR gateによるregime切替
- adaptive likelihoodとの同時変更
- particle/seed数増加
- raw-test inference、submission

## 再現性

- seedは`experiment + well + variant + public_likpf + seed_index`から安定生成する。
- Numba single workerでwell間parallel RNGを使わない。
- stochastic成分は初期particle、regime transition、rate/position propagation、conditional resampling。
- gzip row predictionsはdecompressed content SHAを主証拠にする。
- 保存済みexp072 controlとはseed-paired replayではないため、差はcandidate utilityの比較であり
  seed-paired因果量ではない。

## 静的検証

- Jupytext `.py -> .ipynb`変換: PASS
- Jupytext `--to ipynb --test`: train / inferenceともPASS
- `py_compile`: helper / train / inference / settingsともPASS
- `ruff --select F821`: PASS
- `make validate-exp EXP=exp242_two_regime_rate_noise_pf`: strict PASS
- Numba未導入のローカル環境ではJIT compileを実行できなかった。`njit`をidentity化した
  2 seeds x 10 particles x 3 rowsのpure-Python kernel smokeはshape、有限値、regime fraction範囲をPASS。
- authoritativeな初回Numba実行はKaggle Notebook上で行う。
- canonical train packageを`kentookumura/exp242-two-regime-rate-noise-pf-train`、title
  `exp242 two regime rate noise pf train`として生成した。private、CPU、internet disabled、
  competition source、exp072/exp115/exp209 kernel sourcesを確認した。
- bootstrap manifestにconfig、train/inference script、settings、`two_regime_rate_noise_pf.py`が
  含まれ、config内のtransition、multiplier、variant/config/fold/booster数が正しいことを確認した。
- Kaggleへのpushは実施していない。

## 次のアクション

1. 静的検証を完了する。
2. Kaggle CPU push前にvariant/config/fold/booster数とcontrol再生成なしを再確認する。
3. ユーザー承認がある場合だけfull train-side auditを実行する。

## 2026-07-13 Kaggle train実行承認とpreflight

- ユーザーからKaggle実行の明示依頼を受けた。
- 実行対象: `two_regime_k4` 1 variant、500 particles x 128 seeds、全eligible wells。
- LightGBM config / fold / booster: `0 / 0 / 0`。
- PF control replay / treatment replay: `0 / 1`。保存済みexp072 controlは比較用に読むだけで再生成しない。
- runtime: Kaggle CPU、GPU false、internet false、single worker。
- transition、初期regime、turn multiplierをconfig/packageの双方で再確認した。
- canonical kernel: `kentookumura/exp242-two-regime-rate-noise-pf-train`。
- title: `exp242 two regime rate noise pf train`。
- OAuth credential、legacy credential、strict experiment validation、metadata slug、run-on-pushを確認した。

## 2026-07-13 Kaggle train v1起動

- push: `kaggle kernels push -p experiments/exp242_two_regime_rate_noise_pf/kaggle/train`
- result: kernel version `1` successfully pushed。
- URL: `https://www.kaggle.com/code/kentookumura/exp242-two-regime-rate-noise-pf-train`
- Kaggle kernel id_no: `126894884`。
- pullしたmetadataでcanonical id/title、private、CPU、GPU/TPU false、internet false、
  competition source、exp072/exp115/exp209 kernel sourcesを確認した。
- initial status: `KernelWorkerStatus.RUNNING`。

## 2026-07-13 Kaggle train v2完了

- final status: `KernelWorkerStatus.COMPLETE`。
- runtime: 23,665.002秒、3,783,989 rows / 773 wells、coverage 1.0。
- control `exp072_likpf_mean`: RMSE 11.594897672、MAE 7.067632584、within10 0.772807479。
- candidate `pf_two_regime_k4_mean`: RMSE 13.254455162、MAE 8.912313991、
  within10 0.678840240、control差+1.659557490。
- 全distance bucketで悪化。`1000_plus`は+1.753466596。
- hidden-like spatialは+0.864214500、typewell-purgedは+0.906063709。
- well単位は275改善 / 498悪化、median delta +1.236398714、最大回帰+41.956967959。
- turn particle fraction 0.018088150、posterior mass 0.017897240、switch fraction 0.000561504。
  posterior massがparticle fractionを上回らず、turn regimeが観測尤度に支持された証拠はない。
- output archiveを数値表確認のため取得した。取得中はrow candidatesが一時0 byteに見えたが、
  download完了後は160,223,160 bytesとなりsummaryを含む全10生成物が揃った。
- row candidates decompressed SHA256:
  `13ca093b6fccf197e8c4265f4ede3ddc26e8f5e80e5ef64ad6b8d39dfb448635`。
- exp072 validation source decompressed SHA256:
  `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`。
- exp209 reconstructed control decompressed SHA256:
  `ee3b548b0d38f78966742542e86fa31b7e64698d4762b924c924a5895d2ee3f4`。
- overall、1000+、hidden-like、worst-wellの全guardに失敗したため不採用。
  transition / initial ratio / multiplierの追加grid、raw-test inference、submissionは行わない。
- push後5分超で再確認しても`KernelWorkerStatus.RUNNING`。v1が例外終了した約262秒地点を越え、
  同じcall-signature errorが解消されたことを確認した。実行中ログはCLI上ではまだ空。
- 初期`kaggle kernels logs`は空。実行中にCLI logsが空になる既知挙動として扱い、
  同じkernel IDのまま監視する。空ログを失敗や再pushの根拠にしない。

## 2026-07-13 Kaggle train v1失敗と修正

- final status: `KernelWorkerStatus.ERROR`。
- 最初の有効な例外: `TypeError: not enough arguments: expected 23, got 22`。
- failure class: code / call-signature mismatch。データpath、Kaggle環境、memory、runtime起因ではない。
- 原因: `_two_regime_likpf_allseeds`の署名に追加済みの`turn_multiplier`を、
  `run_pf_for_holdout`の呼び出しで渡していなかった。PF kernel本体の実行前に停止した。
- 修正: `velocity_noise`の直後にconfig由来の`turn_multiplier`を渡す1引数だけを追加した。
- 科学設定、transition matrix、初期regime比、粒子/seed数、variant数、control利用は変更しない。
- 同じcanonical kernel IDでv2を再実行する。

## 2026-07-13 Kaggle train v2起動

- strict validation、Jupytext test、`py_compile`、`ruff --select F821`を再実行してPASS。
- preflightは1 variant、LightGBM 0 config / 0 fold / 0 booster、PF treatment 1、
  control再生成なしでv1から変更なし。
- canonical package内に`turn_multiplier`引数と固定configが含まれることを確認した。
- 同じkernel IDへversion `2`をpushした。
- URL: `https://www.kaggle.com/code/kentookumura/exp242-two-regime-rate-noise-pf-train`。
- initial status: `KernelWorkerStatus.RUNNING`。
