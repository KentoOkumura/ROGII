# exp486_exp226_geometry_residual_likelihood_pf 結果

## 仮説

exp226 geometryのabsolute potentialまたはslow residual stateがPF mode slipを減らす。

## 設定

- Variant A: geometry unary sigma `20 ft`、lambda `0.50`
- Variant B: `TVT=tvt_geop+offset`、slow offset-rate
- 共通: 500 particles、128 seeds、temperature 5
- Stage 0: 2 variants ×32 wells = 64 PF well-runs
- Stage 1: 2 variants ×773 wells = 1,546 PF well-runs
- control PF / HMM / Beam / model / booster / GPU rerun: 0

## 結果

Kaggle private CPU version 1（id_no `129170320`）は`COMPLETE`。
32 wells / 156,088 rows、8,192 seed-well、4,096,000 particle startsを完走した。

Stage 0はtechnical / mechanismともFAILし、最終statusは
`stage0_fail_closed`となった。

- 事前固定runtime投影:
  `180,871.020133 sec > 30,600 sec`
- fixed32 target-free wall time:
  `1,029.996205 sec`
- peak RSS gate measurement:
  `1.239162 GiB`
- residual support:
  min `0.9999999999999988`、max `1.0000000000000011`
- `support > 1`:
  54,924 / 156,088 rows

support gateは正規化weightの浮動小数誤差により最大約`1.1e-15`だけ1を
超えたものだが、runtime gateも独立に大幅FAILした。このoriginal判定は
Stage 1実行後も変更しない。

fixed32の記述RMSEは次のとおり。これはCVでもpromotion判定でもない。

| variant | RMSE | saved exp404比 gain |
| --- | ---: | ---: |
| absolute geometry unary | 9.183489453 | +0.433251355 ft |
| slow residual-offset state | 10.399506240 | -0.782765432 ft |
| saved exp404 control | 9.616740808 | - |
| exp226 `tvt_geop` reference | 9.267204778 | - |

## 再現性

- common seed: `likpf::train::<well_id>`、variant名を含めない
- scientific contract SHA:
  `62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e`
- prediction logical / decompressed SHA:
  `10451d62e5921fd5624d93b5c0025ac2e575fdbc7e42c8d9bd24b7ba6f736821` /
  `1bf351e9e57e08e84b4d1a9d719d2f1dfbef2a99d7e95b585b8c97039804a058`
- absolute-ledger decompressed SHA:
  `7bf5d5a11db73045a3967ddf27b2d983afa904242355dae59d98b2a1f8c62310`
- residual-ledger decompressed SHA:
  `177cf13a28633f5b6ca0672f5e8759c571edb7e7499d5adb692289b05e05e38d`
- truth/control/role/fold read before両variant freeze:
  `0 / 0 / 0`
- frozen variant-wells:
  `64 / 64`

Kaggle outputを一時領域へ取得し、summary、gate、runtime、prediction、
mechanism ledger、truth-late rowsのraw SHAがログ記載値と一致することを確認した。

## 解釈

列allowlist、geometry coverage、variant式/state、finite prediction、common seed、
execution count、truth-late freeze、RSS、geometry factor active、ESS、
residual state non-degenerateはPASSした。

absolute geometry unaryはfixed32記述値では保存exp404より良かったが、
fixed32はCVではなく、事前technical gateを通過していない。二variant間の
winner selectionも禁止しているため、この値からabsolute variantを昇格させない。
slow residual-offset stateはfixed32記述値でも保存controlより悪かった。

## Stage 1結果

ユーザーが実行時間を許容し、全well Stage 1を明示承認した。元の
`runtime_projection=false`を保持したままruntime例外を適用する。supportは
物理的な逸脱ではなく丸め誤差だったため、Stage 1のtechnical readbackだけ
`[-1e-12, 1+1e-12]`を許容する。

実装・push前契約:

- 2 variants ×773 wells = 1,546 candidate PF well-runs
- 197,888 seed-well trajectories / 98,944,000 particle starts
- saved control PF / HMM / Beam / LightGBM / booster / GPU rerun: 0
- 5 foldsはtruth-late reportingのみで、学習foldは0
- absolute / residualを保存exp404へ独立判定し、同じOOFからwinnerを選ばない
- raw-test inference / submissionは未承認

canonical Kaggle private CPU kernel version 4でtruth-late評価まで完了した。
技術ゲートは全項目PASSしたが、科学ゲートは両variantともFAILし、
`stage1_all_variants_gate_failed_terminal_close`で閉じた。

| variant | RMSE | saved exp404比 gain | 改善fold | by-well p95悪化 | worst悪化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| absolute geometry unary | 9.726938029 | +1.187584044 ft | 4/5 | +10.069321492 ft | +44.021977054 ft |
| slow residual-offset state | 11.139812021 | -0.225289948 ft | 2/5 | +4.795182565 ft | +32.921501347 ft |
| saved exp404 control | 10.914522073 | - | - | - | - |

absolute variantはpooled、raw observed/missing、high-missing、1000+、
hidden-like 2面をすべて改善し、固定exp209 HMMとの50:50も
`10.084909849 → 8.871021642`へ改善した。しかしwell単位の悪化尾部が
事前上限を大幅に超えたため昇格しない。residual variantはpooledと多くの
scopeで悪化した。50:50だけの改善を使ったblend rescueも行わない。

実行量は契約どおり1,546 PF well-runs、197,888 seed-well、
98,944,000 particle startsで、control PF / HMM / Beam / model /
booster / GPU再実行は0。version 2とresume version 4の合算時間は
`13,769.492 sec`、peak RSSは`4.506 GiB`だった。

Stage 1 prediction logical SHA:
`70a5ac662c9c58fe54d050f1350ed08e912ecb4edc6362e98e3c3663cd704ea8`。
raw-test inferenceとsubmissionは実行していない。
