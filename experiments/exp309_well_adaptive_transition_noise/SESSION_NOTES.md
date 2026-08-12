# exp309_well_adaptive_transition_noise セッションノート

## 目的

exp308観測モデルを固定し、known-prefix U-rate innovationからHMM `sig_r`だけをwell適応する。

## 現在の状態

- Route: `pf_beam`
- 状態: upstream exp307 chain FAILにより未実行のまま閉鎖
- variants / HMM runs / boosters: `1 / 773 / 0`
- parent再実行: 0

## 2026-07-21 設計

```bash
make new-steering EXP=exp309_well_adaptive_transition_noise
make new-exp EXP=exp309_well_adaptive_transition_noise
```

- q/a式、MAD、support shrinkage、clip、dependency、gateを固定した。
- `sig_p`はposition floorのため固定し、rate diffusionだけに限定した。
- 実装、Kaggle package/push/run、inference、submissionは行っていない。

## 再現性メモ

- RNGなし。prefix統計と固定式で決定的。
- exp308 PASS後にdependency SHAを固定する。
- 実行時はrate audit、sig_r、prediction、metrics content SHAを保存する。

## 2026-07-21 実装

- ユーザーの「exp309を実装してください」「実装だけ先に進める」を実装承認として記録した。Kaggle package/push/run承認は含めていない。
- `exp309_well_adaptive_transition_noise_compact_selfcontained_train.py`をJupytext percent形式で実装し、compact/正規train Notebookを生成した。
- exp307 finite-only MAD `σ_GR`と、exp308の固定missing confidence `observed=1 / missing=max(0.25,2^(-distance/8))`を候補decoderへ組み込んだ。
- known-prefix `U=TVT_input+Z`の連続rate差から`1.4826*MAD`を計算し、20 innovation未満fallback、`n/(n+100)` log shrink、clip `[0.001,0.004]`を実装した。
- exp209 exact forward-backward kernelをself-containedに維持し、`sig_r`だけをwell別値へ置換した。`sig_p=0.02`、position floor 0.1225、41 rate states、momentum 0.998、prior、posterior meanは固定した。
- exp308 saved predictionをcontrolとして読むdependency preflightを実装した。parent promotion gate、prediction SHA、metrics不一致、またはpending値のままではHMM開始前に停止する。
- transition auditとcandidate predictionをgzip content SHA付きでfreezeした後にだけtruth/folds/hidden-like/LikPFを読むlate joinにした。
- overall/fold/1000+/hidden-like/by-wellに加え、sig-r quintile、support、turning、distance readout、fixed LikPF 50:50 guardを実装した。
- `exp309_well_adaptive_transition_noise_compact_selfcontained_inference.py`と正規inference Notebookはraw-test prediction/submissionを明示的にfail-closedにした。
- 親exp308にはcompact実装がない。実行可能な科学祖先exp307との比較は、exp307 `1,676行/10章`、exp309 `1,991行/10章`で、exp309はhelper importだけの薄いNotebookではない。

### 実行量ガード

- active variants: 1 (`robust_prefix_rate_diffusion`)
- HMM well-runs: `1 x 773 = 773`
- model / LightGBM configs / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle GPU: 0、CPU予定、internet off

### 検証コマンド

```bash
.venv/bin/python -m py_compile experiments/exp309_well_adaptive_transition_noise/exp309_well_adaptive_transition_noise_compact_selfcontained_train.py experiments/exp309_well_adaptive_transition_noise/exp309_well_adaptive_transition_noise_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp309_well_adaptive_transition_noise/exp309_well_adaptive_transition_noise_compact_selfcontained_train.py experiments/exp309_well_adaptive_transition_noise/exp309_well_adaptive_transition_noise_compact_selfcontained_inference.py --select F821,F811,F601
.venv/bin/pytest -q experiments/exp309_well_adaptive_transition_noise/tests/test_exp309_well_adaptive_transition_noise.py
```

- exp309 contract tests: `9 passed`
- 構文チェックと未定義/重複定義チェック: PASS
- Jupytext `--test`: train/inferenceともPASS
- exp307 `_hmm2_fb`とのAST同一性: `True`
- strict experiment validation: PASS
- template validation: PASS
- 全repo test: `521 passed, 2 skipped, 2 failed`。2 failureは既存exp296が`completed_train_side_guard_failed_closed` / `run_variant=false`へ更新済みなのに、testがKaggle実行承認中status/flagを期待している既知の不整合で、exp309 testは全PASS。
- Notebook実行、Kaggle package/push/run、output取得、inference、submissionは未実施。

## 次のアクション

exp308 PASS後にparent status、prediction SHA、parent/direct/blend metricsをconfigへ固定する。Kaggle package/push/runは別途承認後にだけ行う。

## 2026-07-22 dependency close

exp307 v2のpromotion gate FAILによりexp308は未実行のまま閉鎖された。exp309の固定parent dependencyは成立しないため、package/push/run、inference、submissionなしで閉じる。
