# exp486_exp226_geometry_residual_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: Stage 1完了 / 両variant科学ゲートFAILでterminal close
- Train-side CV: absolute `9.726938029` / residual `11.139812021`
- LB / Submit: なし
- 親: exp417、実装参照・保存control: exp404

## 仮説

group-safe exp226 `tvt_geop`をabsolute unaryまたはslow residual-offset stateとして
使うと、PFのpersistent mode slipを抑えられる。

## 変更点

- A: exp279の`sigma20/lambda0.50` geometry unaryを元PFへ加える。
- B: exp281由来の`TVT=tvt_geop+offset` stateでslow offsetを追跡する。

両者は独立報告し、同じOOFからwinnerを選ばない。

## 検証方針

2×fixed32で列allowlist、geometry coverage、state/likelihood、seed、truth-late、
SHAを検証した。ユーザーのruntime例外承認後、2×773 wellsを保存exp404と
独立比較するStage 1へ進む。

## 所見

### 良い点

exp279はexp209を`1.902300 ft`、exp281はexp279を`0.208567 ft`改善した。

### リスク

両HMMともexp226 directに届かずtailが悪化した。exp419のrelative-rate proposalも
supportとwell tailをFAILした。

## 成果物

- Jupytext percent形式のcompact self-contained train候補。
- absolute unary / residual-offsetのNumba PF kernel。
- exp226 allowlist、common stable seed、truth-late freeze、SHA、実行量guard。
- fail-closed inference guardと専用contract test。

compact候補を正規train Notebookへ採用した。正規inference Notebookは
placeholderのまま維持している。Kaggle outputのprediction、mechanism ledger、
gate、runtime、SHAを一時取得して検証済み。

## Stage 0結果

- kernel: `kentookumura/exp486-exp226-geometry-residual-likpf-train` v1
- 64 PF well-runs / 8,192 seed-well / 4,096,000 particle starts
- technical: FAIL（full runtime投影 `180,871.020 sec > 30,600 sec`）
- mechanism: FAIL（support最大`1 + 1.1e-15`のstrict bound）
- absolute unary fixed32記述RMSE: `9.183489453`
- residual-offset fixed32記述RMSE: `10.399506240`
- saved exp404 fixed32記述RMSE: `9.616740808`

fixed32はCVではない。absolute unaryの記述改善からwinnerを選ばず、元の
runtime/support FAILも履歴として保持する。

## Stage 1

ユーザーが実行時間を許容してStage 1進行を明示承認した。strict support
FAILは最大約`1.1e-15`の正規化丸め誤差だったため、Stage 1のtechnical
readbackに限って`1e-12` toleranceを適用する。元のStage 0判定は変更しない。

- 2 variants ×773 wells = 1,546 PF well-runs
- 197,888 seed-well trajectories
- 98,944,000 particle starts
- saved control / HMM / Beam / model / booster / GPU rerun: 0
- truth、保存control、fold、hidden-like roleは両variant freeze後にattach
- 二variantは独立判定し、same-OOF winnerを選ばない

raw-test inferenceとsubmissionは引き続き未承認。

### version 2停止とresume

version 2は両variantの3,783,989行をfreezeした後、exp209保存HMMの
期待SHAが62文字になっていたmanifest typoでERRORになった。科学設定や予測には
変更を加えず、prediction / absolute ledger / residual ledger / well auditを
SHA検証してprivate Dataset
`kentookumura/exp486-v2-stage1-frozen-targetfree`へ保存した。

version 3は展開済みledgerのfloat再serialize SHA差で停止した。version 4では
元CSV payload SHAとschema/coverage/finiteを分離検証し、current kernel
PF well-runs 0で同じtruth-late readoutを完了した。

### 最終結果

| variant | RMSE | saved exp404比 gain | 科学gate |
| --- | ---: | ---: | --- |
| absolute geometry unary | 9.726938029 | +1.187584044 ft | FAIL |
| slow residual-offset state | 11.139812021 | -0.225289948 ft | FAIL |
| saved exp404 control | 10.914522073 | - | - |

technical gateはPASSした。absoluteは4/5 foldsと全事前scopeで改善したが、
by-well p95 / worst regressionが`+10.069 / +44.022 ft`でFAILした。
residualは2/5 folds、pooled `-0.225 ft`でFAILした。eligible variantは0。

同じOOFでのwinner選択、parameter/gate/blend rescue、raw-test inference、
submissionは行わずbranchを閉じた。
