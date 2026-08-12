# exp280_exp226_shift_likelihood_separability_readout セッションノート

## 目的

exp226の局所形状を固定した縦shift bankについて、raw GR/typewell likelihoodがtrue TVTに
近いoffsetをtarget-freeに順位付けできるかを512行blockで監査する。exp279の失敗源を、
候補coverageとlikelihood separabilityへ分け、residual-offset HMMの先行条件を確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train-side readout version 1完了、固定guard PASS
- diagnostic: top1/top3/MRR/signがstable shuffledを各5/5 foldsで上回った
- LB: まだなし

## 実行コスト契約

- active audit variant: 1 (`fixed_shift_bank_raw_gr_gaussian`)
- shift candidates: 13
- validation strata: 保存済み5 folds（trained foldは0）
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- HMM / PF well-runs: `0 / 0`
- parent/control再学習・再生成: 0
- runtime: Kaggle CPU、GPU/TPU/internet off
- inference / submission: disabled / disabled

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実装・検証済み

```bash
make new-steering EXP=exp280_exp226_shift_likelihood_separability_readout
make new-exp EXP=exp280_exp226_shift_likelihood_separability_readout
.venv/bin/python -m py_compile <exp280 train.py> <exp280 inference.py> <exp280 test.py>
.venv/bin/ruff check <exp280 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q experiments/exp280_exp226_shift_likelihood_separability_readout/tests/test_exp280_exp226_shift_likelihood_separability_readout.py
# 6 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 <source.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <source.py>
.venv/bin/python scripts/validate_experiment.py --experiment exp280_exp226_shift_likelihood_separability_readout
# strict validation passed
```

### Kaggle CPU実行前の予定

```bash
task prepare-kaggle-notebooks EXP=exp280_exp226_shift_likelihood_separability_readout \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp280-exp226-shift-likelihood-readout-train --title 'exp280 exp226 shift likelihood readout train' --run-on-push --strict"
task push-kaggle-train EXP=exp280_exp226_shift_likelihood_separability_readout
```

push前に`execution.kaggle_push_approved=true`とする変更、canonical package、実行量を別途確認する。
今回の実装依頼ではpush/full local notebookを実行していない。

### Repository / package validation

```bash
make validate-exp EXP=exp280_exp226_shift_likelihood_separability_readout
# PASS
make validate-template
# PASS
.venv/bin/pytest -q
# 166 passed, 1 unrelated existing exp264 status assertion failed
make prepare-kaggle-notebooks EXP=exp280_exp226_shift_likelihood_separability_readout \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp280-exp226-shift-likelihood-readout-train --title 'exp280 exp226 shift likelihood readout train' --run-on-push --strict"
# PASS, no push
```

- full testの唯一のFAILは`experiments/exp264_exp263_candidate_confidence_dual_selector/tests/test_exp264_candidate_selector_pipeline.py`が
  `inference.status=user_authorized_2026_07_19`を期待する一方、既存exp264 configが外部進行により
  `corrected_inference_v4_running`へ更新済みなことによる。exp280とは無関係なので変更しない。
- canonical metadataはid/title slug一致、private、CPU、GPU/TPU/internet off、run-on-push true、
  competition source 1、kernel source 2。
- loose config SHAとsource config SHAはbyte一致:
  `9faba65a862e93fc3a6e4946dfbeecf54075e21a0274b2069e5d2b95cccaaadc`。
- loose train source SHAとsource train SHAはbyte一致:
  `2f6666e1602e99270f725c080546288fb2b8615b3a392a4fd1bf1a2970bd1db3`。
- prepared notebook SHA:
  `db180c4af39a9b4af25e0680eeea070cf13c06b1fc01406bc045dad016e08db4`。
- embedded bootstrap manifestにも同じconfig/train source SHAが記録されている。

## 変更点

- `docs/legacy/steering/20260719-exp280-exp226-shift-likelihood-separability-readout/`を作成。
- compact self-contained Jupytext train source（9章、1,165行）とfail-closed inferenceを実装。
- 親参照exp279 train source（10章、1,401行）からHMM kernelを除き、代わりにshift scoring、
  score freeze、truth-only label、real-vs-shuffled metricsを展開した。
- exp248の12 nonzero shiftへ0を加えたapproved 13候補を固定した。
- exp252と同じ先頭からの非重複block、末尾short block保持を固定した。
- exp209/279 Gaussian raw-GR emissionとexp279 persistent-offset閾値を固定した。
- target-free score、block readout、fold/scope/shift/by-well、episode、manifest、summaryの保存を実装。
- score APIは`tvt_pred` / `gr_delta` / `tvt_true` / `error` / `abs_error`とraw horizontal `TVT`を拒否する。
- stable shuffled controlだけにlocal SHA256 seed RNGを使い、real scoreはRNGなしとした。
- inference notebookは設定確認後に必ず停止し、submissionを生成しない。

## 再現性メモ

- seed policy: real scoreはRNGなし。shuffledだけ
  `SHA256(experiment, seed, well, block)`由来local RNG。
- stochastic components: stable shuffled-score negative controlのみ。
- CPU/GPU runtime: Kaggle CPU予定、GPU/TPU/internet off。
- input SHA: exp226 decompressed `709eb726...e4c609`、hidden-like raw
  `5f9ac9fa...ca6597`をhard guard。
- target-free score: 全score凍結後にcontent SHAを確定してからtruthを再読込する。
- model manifest / prediction / submission: 生成しない。scientific contract SHAとscore SHAを記録する。
- gzip: mtime=0、raw/decompressed/content SHAを分離する。
- Kaggle kernel id / version / rerun: 未実行。
- deterministic anchor: prediction/submission anchorではなく固定入力へのdiagnosticのみ。

## 次のアクション

1. Kaggle CPU pushが必要なら、実行量を再提示してユーザー承認を得る。
2. guard PASS時だけ別実験のresidual-offset HMMを検討し、FAIL時は優先度を下げる。

## 2026-07-19 Kaggle CPU実行承認

- ユーザーの「実行してください。」を、このcanonical train-side readout 1回のpush承認として記録する。
- canonical kernel ID: `kentookumura/exp280-exp226-shift-likelihood-readout-train`
- active audit variant: 1 (`fixed_shift_bank_raw_gr_gaussian`)
- shift候補: 13 (`[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80]` ft)
- saved fold strata: 5、trained fold: 0
- LightGBM config: 0、total booster: 0、HMM well run: 0
- 親/controlの再生成・再学習: 0
- Kaggle CPUのみ。GPU/TPU/internetはoff。
- inference、submissionは実行しない。
- 実行時点の`execution.kaggle_push_approved=true`をpackageへ固定し、push成功後は意図しない再pushを防ぐためlocal configをfalseへ戻す。

### Kaggle train v1 push

- `2026-07-19`にcanonical kernel version 1をpushし、run-on-pushで実行開始。
- URL: `https://www.kaggle.com/code/kentookumura/exp280-exp226-shift-likelihood-readout-train`
- pushed config SHA256: `4a95c5143f8decd163be14913d5bf4717f76513f724e2655e2eaa8523ff70ed1`
- pushed train source SHA256: `2f6666e1602e99270f725c080546288fb2b8615b3a392a4fd1bf1a2970bd1db3`
- pushed notebook SHA256: `cbe713b145ce0aa28e684bdc896d9ffeba054e9a903deac4b3dbac76b2efc991`
- push成功後、再push防止のためloose configを`kaggle_push_approved=false`へ戻した。v1 packageは承認済みconfigを保持する。
- remote metadata: kernel id_no `127828902`、private、CPU、GPU/TPU/internet off、指定した2 kernel sourcesで一致。
- 最終確認時の状態は`KernelWorkerStatus.RUNNING`。ユーザー指示により定期監視を停止し、Kaggle上のversion 1実行は継続する。

## 2026-07-19 Kaggle train v1完了確認

### コマンドと実行状態

```bash
kaggle kernels status kentookumura/exp280-exp226-shift-likelihood-readout-train
# KernelWorkerStatus.COMPLETE
kaggle kernels logs kentookumura/exp280-exp226-shift-likelihood-readout-train
kaggle kernels output kentookumura/exp280-exp226-shift-likelihood-readout-train \
  -p /tmp/exp280-output-v1
```

- kernel version: 1
- kernel id_no: `127828902`
- generated at: `2026-07-19T01:06:55.358377+00:00`
- runtime: `456.972453`秒
- rows / wells / blocks: `3,783,989 / 773 / 7,787`
- active audit variant / trained fold / booster / HMM well-run: `1 / 0 / 0 / 0`
- inference / submission: 未実行 / 未実行

### 固定guard

| 指標 | real | shuffled | lift | real > shuffled folds |
| --- | ---: | ---: | ---: | ---: |
| top1 | 0.189547 | 0.075767 | +0.113779 | 5/5 |
| top3 | 0.452421 | 0.234493 | +0.217927 | 5/5 |
| MRR | 0.389626 | 0.245536 | +0.144090 | 5/5 |
| sign | 0.498523 | 0.418518 | +0.080005 | 5/5 |

- expected folds / row identity coverage / score finite coverage: PASS / 1.0 / 1.0
- bank range / quantization coverage: 1.0 / 1.0
- mean rank: 4.653140 / 13 candidates
- top1 regret RMSE: mean 13.955240 ft、p90 38.615667 ft
- decision: `separability_supported_consider_separate_residual_offset_hmm`

fold別lift範囲はtop1 `+0.101440～+0.125935`、top3 `+0.186518～+0.246523`、
MRR `+0.128698～+0.159668`、sign `+0.064136～+0.099751`。4指標とも全foldで正だった。

1000+、hidden-like spatial / typewell-purged、persistent-offset scopeでも4指標のliftは全て正。
persistent-offsetはtop1 0.150524、top3 0.394130、MRR 0.347918、sign 0.530818。
nearは1 block / 1 wellだけなので性能根拠に使わない。

### 再現性・生成物確認

- scientific contract SHA256:
  `60d32ba96e0f71fc1f02f53d9e274e97d96516549ee27c87ea40bbb666af7978`
- target-free score content SHA256:
  `4a546cfe5f9291168bdb4dcb912182b079e0343af845f76005f6a7100ac3aa46`
- score contractの`truth_attached=false`、summaryのtruth attachment stageは
  `after_all_target_free_scores_frozen`で、上記content SHAが一致した。
- exp226 input decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- hidden-like assignment raw SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- target-free score gzip raw / decompressed SHA:
  `fd698c81a6ffcb0e63921fa34f55d9ded199071d8e9c7d27536ad91ae4fe2bad` /
  `c6e9e39a5fd3944f7516e68bdd6b9d27430a47f0bfbe3c50d39f5369791d99c3`
- block readout gzip raw / decompressed SHA:
  `fe2b6527cac719993c6f6acfd5fff90305aafb4caec9072731aaa2297c8d48b8` /
  `c1cd8fb1ab19b5ac8a45fd0fd98141f31d7e2336ba35b0f5c546b81e87c16ee3`
- fold/scope/shift/by-well/episode/well-manifest/input-manifestの7 file SHAをsummary記載値と照合し、全一致。
- output側config / train source SHAはpushed記録
  `4a95c514...0ed1 / 2f6666e1...d1db3`と一致した。

### 解釈と次アクション

- shift likelihood separabilityの先行条件は支持された。
- top1とsignの絶対値はhard shift correctionに不十分なので、direct correctionは不採用のまま。
- `exp226_residual_offset_exact_hmm_transition_probe`を「高・先行readout通過済み」へ上げる。
- 後続はexp226座標系のslow offsetだけを状態にする別実験の1 fixed grammarとし、
  exp280同一OOFでのshift/grid/calibration救済、inference、submissionは行わない。
