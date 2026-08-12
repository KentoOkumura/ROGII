# exp281_exp226_residual_offset_exact_hmm_transition_probe セッションノート

## 目的

exp280でfold-stableな識別力が確認されたraw GR likelihoodを、exp226`tvt_geop`座標系の
slow residual offset exact HMMへ時系列統合する。absolute TVTを自由探索せず、1 fixed grammarで
exp263 fixed baselineを超えられるか反証可能に評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU version 1完了 / train-side guard FAIL / negative close
- CV / LB: 9.827420 / 未提出
- active HMM variant / well-runs: `1 / 773`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control再学習・再生成: 0
- GPU/TPU/internet: off / off / off
- inference / submission: disabled / disabled
- Kaggle push approval: true（2026-07-19 10:41 JST、ユーザー明示承認）

## 固定scientific contract

- `TVT_t = exp226_tvt_geop_t + delta_t`
- `delta_t = delta_(t-1) + offset_rate_t * dMD_t + position_noise`
- offset grid nominal `[-80,80] ft` / step `0.35 ft`
- offset-rate 41 states / span `+-0.10`
- `sig_r=0.002`、`sig_p=0.02`、`start_delta=0`、`start_sig=0.75`
- `initial_offset_rate=0`、`r0_sig=0.01`、`mom=0.998`、likelihood weight 1.0
- exp209 Gaussian raw-GR emission、known-prefix sigma clip 10～60、missing-GR補間を固定
- exp226 final`tvt_pred` / `gr_delta` / truth / errorはdecoderへ渡さない

## コマンドログ

```bash
make new-steering EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe
make new-exp EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe
.venv/bin/python -m py_compile <exp281 train.py> <exp281 inference.py> <exp281 test.py>
.venv/bin/ruff check <exp281 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q experiments/exp281_exp226_residual_offset_exact_hmm_transition_probe/tests/test_exp281_exp226_residual_offset_exact_hmm_transition_probe.py
# 6 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 <source.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <source.py>
make validate-exp EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe
make validate-template
make prepare-kaggle-notebooks EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp281-exp226-residual-offset-exact-hmm-train --title 'exp281 exp226 residual offset exact hmm train' --run-on-push --strict"
make push-kaggle-train EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe
kaggle kernels pull kentookumura/exp281-exp226-residual-offset-exact-hmm-train \
  -p /tmp/exp281-kaggle-pull.Vs1OAF -m
kaggle kernels logs kentookumura/exp281-exp226-residual-offset-exact-hmm-train
kaggle kernels status kentookumura/exp281-exp226-residual-offset-exact-hmm-train
kaggle kernels output kentookumura/exp281-exp226-residual-offset-exact-hmm-train/1 \
  -p /tmp/exp281-output-v1.7TXaQK --file-pattern '<metrics/manifests>'
kaggle kernels output kentookumura/exp281-exp226-residual-offset-exact-hmm-train/1 \
  -p /tmp/exp281-output-v1.7TXaQK --file-pattern '.*oof_predictions.csv.gz$'
```

承認済みpackageを2026-07-19 10:42 JSTにcanonical kernelへpushし、private CPU version 1を開始した。
full local notebookは実行していない。開始直後のCLI logsは空だったが、この環境では実行中logsが
空になる既知挙動のため、停止・失敗とは判定しない。

## Kaggle version 1

- kernel: `kentookumura/exp281-exp226-residual-offset-exact-hmm-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp281-exp226-residual-offset-exact-hmm-train`
- id_no: `127831519`
- private / CPU / TPU off / internet off
- Kaggle metadata `machine_shape`: `None`（CPU）
- docker image: `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`
- run contract: 1 variant / 773 well-runs / 0 config / 0 trained fold / 0 booster
- initial logs: empty（実行中の既知挙動）
- 2026-07-19 10:46 JST status: `KernelWorkerStatus.RUNNING`
- 2026-07-19 15:18 JST status: `KernelWorkerStatus.COMPLETE`
- completed_at from summary: 2026-07-19 14:54:38 JST
- elapsed: 15,042.787秒（4時間10分42.8秒）
- rows / wells: 3,783,989 / 773、well status `ok` 773/773
- final status: `completed_train_side_guard_failed`
- output取得時に`.../versions/1`形式はCLIのkernel format errorとなったため、正しい`.../1`へ直した。
  実験runへの影響はない。

## Version 1結果

- residual-offset HMM RMSE: 9.827420
- exp263 fixed RMSE: 8.238332、delta `+1.589088 ft`
- exp226 prediction / exp209 exact HMM: 9.427110 / 11.938287
- 改善fold: 0/5。fold deltaは`+0.754790 / +1.402064 / +1.464329 / +3.041602 / +1.081377 ft`。
- near / 1000+ / hidden-like spatial / typewell-purged delta:
  `+0.280916 / +1.792419 / +1.808499 / +1.610008 ft`。
- improved / worsened wells: 408 / 365。median delta `-0.221848 ft`、p95 `+10.982960 ft`、
  worst `8a3da6d1` `+30.961675 ft`。
- persistent episodes: 530 vs exp263 551。256 / 512 recovery delta:
  `+0.000863 / +0.031897`。
- PASS: exp263 parity、delta-grid / finite coverage、episode count、256/512 recovery。
- FAIL: overall gain、fold、scope、worst-well。総合guard FAIL。

## Output・SHA監査

- 必要なmetrics / fold / distance / hidden-like / by-well / recovery / input / well / decoder
  manifestとOOFだけを`/tmp/exp281-output-v1.7TXaQK`へ取得した。
- OOF gzipは206,727,257 bytes、gzip整合性PASS、3,783,989 rows。
- raw gzip SHA `57d18866ce285aca98e62f536cc8f6bada00f6f869e43c7fde74c6b8413872f1`。
- decompressed SHA `3a99b1d9604da27952ef029f56a9eec42c477f3d78d1fdf3763301df03867386`。
- logical content SHAは`float_precision=round_trip`で再計算し、
  `d7f902b856a78bb040a2fa0cfe3ed94daad5caf26449459b40d0b150eb65e440`と一致。
- decoder file / scientific mapping SHAは
  `a5fa3c157badd8d9a2c3ba89118d06a2643de54b68c11c72653dc92b041fede8` / 
  `876a6d57715fd046eaafe60d870e245ecb8704ff18aae1efe09ae79c53b48069`。
- candidate / fold / distance / hidden / by-well / input / recovery / well manifestのbyte SHAも
  Kaggle summaryと全一致。input decompressed SHA hard guardも全PASS。
- stderrはdebugger / mistune / nbconvert warningだけで、traceback / OOM / runtime errorなし。

## 実装内容

- compact self-contained train sourceをexp279 exact kernelから必要部分だけreparameterizeした。
- exp226 safe列reader、raw well/typewell reader、row-dependent `GR(tvt_geop + delta)` emissionを実装した。
- exact forward-backward position transitionから`-dZ`を除き、delta-rateだけをtransitionとした。
- posterior mean delta / TVT / std / loglik、fold / distance / hidden-like / by-well / recoveryを保存する。
- exp263 fixedを保存済みexp226 / exp209 / exp072だけから再構成する。
- all-well candidate生成完了後にだけtruth/controlを結合する。
- overall / fold / scope / worst-well / episode数 / 256・512 recoveryの全guardを実装した。
- inferenceは設定確認後もcandidate/submissionを作れないfail-closed contractとした。

## Notebook構造比較

- 参照exp279 train source: 1,401行 / 10章。
- exp281 train source: compact self-contained 10章。absolute unaryを除き、delta coordinate、
  row-dependent emission、recovery guardを追加した。
- 同一exp helper importなし、notebook sourceに`__file__`依存なし。

## 再現性メモ

- RNG: なし。well文字列昇順、保存済みexp226 folds、outer worker 1、Numba threads 4。
- input SHA hard guard: exp226 `709eb726...e4c609`、exp209 `ee3b548b...2ee3f4`、
  exp072 `99a3c70a...0e1350`、hidden-like `5f9ac9fa...ca6597`。
- gzip outputはraw SHAとdecompressed SHAを分け、logical prediction content SHAを主証拠にする。
- fitted modelなしのためdecoder scientific manifest SHAをmodel SHAの代替とする。
- inference/submission無効のためsubmission SHAは対象外。
- 初回Kaggle成功だけではdeterministic anchorと呼ばない。
- 承認済みpackaged config SHAはloose configと一致
  (`2d7cf9b0ca0eecdcf58f36a30fb4592eddef63b1dfd703a04ac31f7082014984`)。
- packaged train / inference source SHAはloose sourceとそれぞれ一致
  (`f3682657...c577c7` / `9c36994d...ea282`)。
- `kernel-metadata.json`はCPU / internet off、4 kernel sources、private、run-on-push true。

## 次のアクション

1. 本branchをnegative closeし、parameter救済、PF/blend/selector、inference、submitへ進まない。
2. 独立仮説の`exp226_prefix_masked_offset_predictability_readout`と既存future-evidence回復監査を優先する。
