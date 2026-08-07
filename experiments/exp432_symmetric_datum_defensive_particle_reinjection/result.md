# exp432_symmetric_datum_defensive_particle_reinjection 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了し、`stage0_fail_closed`。
fixed32 mechanism preflightであり、CV、LB、提出はない。

- kernel: `kentookumura/exp432-symmetric-defensive-reinjection-train`
- version / id_no: `1 / 128974856`
- full eligible: false

## 仮説

方向なしの±datum defensive proposalを最初のrate-gap eventで一度だけ使うと、元PF targetを保ちながらtruth近傍particle supportを回復できる。

## 固定設定

- PF親: exp404
- trigger: exp412 persistent beta-filter rate-gapの最初のfalse→true event
- direction: 不使用
- mixture: base/minus/plus=`0.80/0.10/0.10`
- datum: `max(filtered HMM position std, 0.35 ft)`
- correction: full mixtureに対する`p0/q`、clipなし
- PF/readout: 500 particles、128 seeds、x1.0、Gaussian evidence T=5

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 | `stage0_fail_closed` |
| Triggered / no-event wells | `21 / 11` |
| Support外率 absolute reduction | `-0.004278274` |
| Trigger後512-row SSE reduction | `0.120871403` |
| Nonworse folds | `3 / 5` |
| Control pooled RMSE delta | `-1.005694415 ft` |
| Control worst-well RMSE delta | `+0.583072976 ft` |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 実装検証

- exp432専用contract test: `12 passed`
- exp209 first-pass HMMの独立kernel parity: PASS
- no-event exp404 seed prediction / log-likelihood / resampling / ESS /
  clip counter bitwise parity: PASS
- event前baseline/treatment common-random prediction parity: PASS
- `0.80/0.10/0.10` component stream、対称shift、finite log importance、
  `log(p0/q) <= log(1.25)`、importance quadrature moment: PASS
- Jupytext round-trip、`py_compile`、Ruff F821: PASS

これらは実装contractの検証であり、Stage 0機構結果やCVではない。

## 生成物

Kaggle outputにtarget-free prediction、trigger schedule、well metrics、
truth-late rows、SHA ledgerを生成した。主要なdecompressed SHAは次のとおり。

- prediction:
  `c25de60c841a2fed6f27981aa65fb44759f41235e44a56feaa3bc0d28ce5ad4b`
- trigger schedule:
  `75cd32276c0a772635705b18d165dcd00b674f2ccb9b496f6b688a2b8fb15b79`
- well metrics:
  `adf5383d47ac5517e02cd74484a206209b44f28549b5e2630da97d2cb18aae42`
- truth-late rows:
  `62e62a096bd016a6bf7d1e2058bb606d3628d192236802dd29ff6a284e2d646f`
- SHA ledger:
  `c6cbf48f52491f6e269697083380f65e1a626c9bd38c9bdf346b47277d1ea7f6`

## 再現性

- deterministic anchor: false（stochastic cross-rerun parity未確認）
- scientific contract SHA:
  `46358f83b27cc5481f14b7dd76267046cf25d5805e690689bb602ce574f0efc8`
- executed config SHA:
  `c059b0895f35eb850246a41374dfcc9505c63fc27519a44dec502fc5d4cb04af`
- package / Kaggle pull 21-cell source SHA:
  `07dfda43af46b43e4bfa344e9365bfa427320a966a6028ce9ea525d792bafcf3`
- elapsed / peak RSS: `3,798.063205秒 / 1.244778 GB`

## 解釈

importance correction、保存parent parity、truth-late、no-event bitwise parityは
通った。SSEは12.09%改善したが、主目的のsupport外率は改善せず0.428 points悪化した。
fold安定性は3/5、worst controlも固定上限を0.333 ft超えたため、局所的なSSE gainを
support回復の証拠とは扱えない。

実行Notebookのruntime gateはStage 0全elapsedを単一kernelへ線形投影し、
`91,746.964秒`でFAILした。一方、設計は4 PF shardsであり単純4分割参考値は
`22,936.741秒`。この実装上の注意点は残るが、mechanism gateが独立に3項目FAIL
しているためfull不適格の結論は変わらない。

## 次

現行branchをfail-closedで閉じる。fixed32上の救済、full、inference、submissionは
行わず、後続案のpositive evidenceにも使わない。
