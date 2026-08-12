# 要件

## 依頼

HMM の持続位置状態を TVT から `U=TVT+Z` へ変える実験を行いたい。
`exp438_u_state_fixed_lattice_exact_hmm` として、まず設計を確定する。

今回は steering、実験 scaffold、backlog、静的検証までとする。
実行可能な HMM、Kaggle package / push / run、Stage 1、推論、提出は
それぞれ別のユーザー承認まで行わない。

2026-07-29の追加指示`exp438を実装してください`により、実行可能な
compact self-contained Stage 0候補と専用testの実装を承認範囲へ追加した。
正規notebookの置換、Kaggle package / push / run、Stage 1、推論、提出は
引き続き別承認とする。

2026-07-29の追加指示`実行してください`により、compact候補の
正規notebook採用とKaggle private CPU fixed32 Stage 0のpackage / push / runを
承認範囲へ追加した。Stage 1、推論、提出、same-fixed32 rescueは対象外のままとする。

## 仮説

exp209 は持続位置状態を固定 TVT 格子に置き、

```text
delta_TVT = r_U * delta_MD - delta_Z
```

で遷移する。連続状態では `U=TVT+Z` への変換は厳密に同値だが、
0.35 ft の固定離散格子では `-delta_Z` を含む position kernel の丸め位相が
毎行変化する。

最後の既知点で固定した絶対 U 格子へ状態を変え、

```text
delta_U = r_U * delta_MD
TVT_t = U_t - Z_t
```

とすれば、rate process や連続運動学を変えずに Z 由来の格子位相だけを除ける。
これにより position-kernel の離散化誤差と累積 TVT drift が減る可能性がある。

## 制約

- Route は `pf_beam`。
- 親/control は
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 持続 joint state は `(U, r_U)` とし、rate historyを維持する。
- U 格子は、exp209 の親 TVT 格子へ最後の既知 `Z` を一度だけ足して固定する。
  行ごとの再格子化、adaptive grid、interpolation transportは行わない。
- emission は各行・各 U state を `TVT_state=U_state-Z_t` へ厳密変換して、
  exp209 と同じ Type Well GR Gaussian likelihoodを評価する。
- 出力は smoothed `E[U_t]-Z_t` とし、提出 target は TVT のまま。
- exp209 の rate grid/kernel、momentum、position noise、5-cell support、
  GR calibration、initial prior、forward-backward、posterior meanを固定する。
- position meanは exp209 と同じ到着 rate を使う。
  source/destination rateの台形 joint edgeは併用しない。
- exp435 の memoryless rate、exp364 の curvature state、reset/re-anchor、
  formation/XY prior、ML、PF sampling、selector、blendを追加しない。
- 保存済み exp209 control をSHA固定で使い、再実行しない。
- prediction / diagnostic / SHA freeze前にfold、episode outcome、suffix truthを読まない。
- 同じ fixed32 / OOF で U grid anchor、step、band、noise、rate、emissionを調整しない。

## 受け入れ基準

- steering 3文書、実験 scaffold、backlogが
  Stage 0実装前は`design_frozen_unimplemented`、実装後は
  `stage0_implemented_unrun`で一致する。
- 連続座標変換が親と同値であり、科学的差分は固定離散格子の選択だけだと明記する。
- constant-Z parent parity、U/TVT readout identity、transition/emission coordinate
  identity、brute-force small referenceを実装時の数値contractに含める。
- Stage 0 fixed32 mechanism、Stage 1 full OOFの順序とAND gateを固定する。
- scientific variant 1、Stage 0 HMM well-runs 32、Stage 1最大773、
  parent rerun / ML model / booster / PF / Beam / GPU各0を記録する。
- 初回runをdeterministic anchorとせず、同一設定rerunのprediction /
  transition SHA一致を必要とする。
- 実装承認を記録し、Kaggle実行は別にユーザー承認を得る。
