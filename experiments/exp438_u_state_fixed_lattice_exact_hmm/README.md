# exp438_u_state_fixed_lattice_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: `stage0_fail_closed_v1`
- Kaggle: private CPU version 1 COMPLETE（id_no `129056676`）
- CV / Public LB / Private LB: なし
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Stage 1 / inference / submission: 無効

## 仮説

`U=TVT+Z`は連続状態ではexp209の単なる座標変換になる。しかしexp209は
0.35 ftのTVT格子を全rowで固定するため、`r_U*ΔMD-ΔZ`を離散位置kernelへ
投影するとZ由来の位相が入る。

本実験は最後の既知点でabsolute U格子を一度だけ固定し、
`ΔU=r_U*ΔMD`で遷移する。各rowのemissionは`TVT=U-Z`で評価し、
出力を`E[U]-Z`へ戻す。rate process、noise、GR emission、prior、
forward-backwardはexp209から固定した。

## 実装

- compact self-contained Jupytext trainとcanonical train notebook
- fixed `(U, r_U)` joint exact HMM
- coordinate/emission/readout identity
- constant-Z parent parity、brute-force small-path reference
- truth-free quantization ledgerとtruth-late mechanism gate
- 保存済みexp209 controlのload-only比較
- fail-closed inference guard
- 専用test 12件、py_compile、Ruff F821、Jupytext round-trip、
  strict experiment/template validation PASS

## 検証方針

Stage 0はfixed32の1 candidateだけを保存済みexp209 controlと比較する。
technical gateは座標identity、数値reference、normalization、truth-late、
artifact SHA、runtime、RSSをAND判定する。mechanism gateはquantization bias、
forward-cause / persistent SSE、persistent well / fold、matched-control
pooled / p95をAND判定する。全gate PASS時だけStage 1を検討し、
1項目でもFAILならrescueなしで閉じる。

## Stage 0結果

Kaggle kernel:
`kentookumura/exp438-u-state-fixed-lattice-exact-hmm-train` version 1。
1 variant、32 HMM well-runs、5 reporting foldsをCPU 1 threadで実行した。
parent HMM rerun、ML、booster、PF、Beam、GPU、submissionはすべて0。

- elapsed: `1452.118 sec`
- Stage 1 projection: `33907.307 sec` > `30600 sec`（FAIL）
- technical: 20 PASS / 1 FAIL
- mechanism: 0 PASS / 7 FAIL
- quantization bias reduction: `-43.580%`
- persistent episode SSE reduction: `-824.234%`
- persistent improved wells / folds: `2/16`, `0/5`
- matched-control pooled RMSE delta: `+43.320 ft`
- control by-well delta p95: `+72.481 ft`

連続座標identity、constant-Z parity、brute-force、normalization、
truth-late、artifact readback SHAはPASSした。科学差分だけが強く悪化したため、
実装不備ではなくfixed absolute-U lattice仮説のnegative evidenceと判断する。

## 実行入口

- `exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py`
- `exp438_u_state_fixed_lattice_exact_hmm_train.ipynb`
- `exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_inference.py`
- `exp438_u_state_fixed_lattice_exact_hmm_inference.ipynb`

run後は`execution.run_hmm=false`、`runtime.run_approved=false`へ戻している。
inferenceはStage 0不合格のためfail-closedのままである。

## 所見

fixed-U化は連続座標contractを保ったまま実行できたが、離散格子の
quantization bias、persistent episode、matched controlを同時に悪化させた。
したがって問題は座標変換の実装誤差ではなく、absolute-U側へ固定した
0.35 ft格子のsupport alignmentにあると判断する。固定TVT格子のZ依存位相は
exp209でregularizationとして働いていた可能性がある。

## 結論

固定absolute-U latticeはquantization biasを減らさず、persistent/controlを
大きく悪化させた。事前登録どおりStage 1、inference、submissionへ進まず、
grid phase/anchor/step/band、noise、rate、emission、blend/selectorによる
same-fixed32 rescueも行わない。
