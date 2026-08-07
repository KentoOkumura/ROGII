# SESSION_NOTES

## 2026-07-30 設計・実装・実行承認

- ユーザー判断: exp458の差がわずかであれば次へ進む。
- 解釈: exp458のexact-parity FAILをPASSへ変更せず、Stage 0Bに限る
  `user_approved_small_numerical_deviation` waiverとして後継exp489に記録。
- Route: `pf_beam`
- 親: runtime/implementation=`exp458`、scientific=`exp444`、root=`exp209`
- scientific variant: 1
- LightGBM config: 0
- fold: 0（fixed32 manifestのreporting foldは評価集計だけに使用）
- booster / fitted model: 0 / 0
- HMM well-run: 新規28、exp458 v2再利用4、評価対象32
- exp458からの累計physical unique-stage runs: 36（exp458 8 repeat runsのうち
  fixed4 repeat2だけを再利用するため、Stage 0B契約上の親+後継表記は4+28）
- PF / Beam / GPU: 0 / 0 / 0
- CPU: Kaggle private CPU、outer 4 process、各worker Numba/BLAS 1 thread
- Stage 1 / inference / submission: false

## 凍結waiver

- exp458 v2 kernel: `kentookumura/exp458-accel-state-exact-runtime-eng-audit-train`
  version 2、id_no 129168013
- parent prediction mean max abs error: `0.00010413506242912263 ft`
- parent prediction std max abs error: `0.0000635657412058066 ft`
- parent acceleration posterior max abs error: `0.00000897726402104837`
- parent rate diagnostic max abs error: `0.0000010862973659972464`
- scientific contract SHA:
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`
- runtime engine contract SHA:
  `cb14a4f1dfc1d5de03a6e9329402fef476918b7ffb235552e9cd9d98d6c71451`

## Stage 0B 凍結gate

- technical: 32 wells、156,088 rows、finite 1.0、nonzero acceleration mass
  0.01–0.80、pre-freeze forbidden read 0、peak RSS 25 GiB以下。
- mechanism: direction agreement 0.60以上かつ4 folds、forward-cause episode
  SSE改善10%以上、persistent episode SSE改善5%以上、persistent改善10 wells /
  4 folds、matched exp209 control pooled delta 0.02 ft以下、by-well p95
  0.25 ft以下。
- 失敗時はparameter、engine、閾値を救済せずcloseする。

## 2026-07-30 Kaggle Stage 0B結果

- kernel: `kentookumura/exp489-accel-state-fixed32-mechanism-audit-train`
- version / id_no / status: `1` / `129171668` / `COMPLETE`
- 28 well decode wall / notebook elapsed: `794.097712` / `851.420677 sec`
- process-tree peak RSS: `13.971542 GiB`
- 32 wells / 156,088 rows / finite 1.0 / outer 4 workers / inner 1 thread
- technical gate: `10/10 PASS`
- posterior acceleration nonzero mass: `0.664838701`（PASS）
- direction agreement: `0.500308694 < 0.60`、fold
  `0.498489 / 0.501190 / 0.500353 / 0.502353 / 0.499597`、positive
  `0/5`（FAIL）
- forward-cause episode SSE reduction: `0.435458% < 10%`（FAIL）
- persistent episode SSE reduction: `-3.666734% < 5%`（FAIL）
- persistent improved wells / folds: `8/16 < 10/16`、`2/5 < 4/5`（FAIL）
- matched exp209 control pooled RMSE:
  `3.265587 - 3.428436 = -0.162849 ft`（PASS）
- matched control by-well delta p95: `+0.077808 ft`（PASS）
- mechanism gate: `2/8 PASS`、all-AND FAIL
- status: `stage0b_fail_closed`
- Stage 1 / inference / submission: false
- executed bootstrap config SHA:
  `6c98d6c1eb45b1f8e9efbf1a770170e20918c72d90b29d8feff144a2a0007633`

## 解釈

nonzero acceleration massとruntimeは十分で、失敗原因はstate collapseや計算資源
ではない。加速度方向は約50%でランダム同等、persistent episodeも悪化した。
exp459のlikelihood-PF acceleration branchと同じnegative patternがexact-HMM側でも
再現されたため、固定3状態persistent acceleration仮説はrouteをまたいで閉じる。
control safetyだけを根拠にparameter/span/transition/gateを救済しない。
