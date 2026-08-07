# exp438_u_state_fixed_lattice_exact_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1完了。technical gateはruntime projectionだけFAIL、
mechanism gateは7項目すべてFAILしたため、`stage0_fail_closed`で終了した。
Stage 1、inference、submission、same-fixed32 rescueへは進まない。

## 仮説

exp209の固定TVT格子を最後の既知点でanchorした固定U格子へ変えると、
rate processや連続運動学を変えずにZ由来のposition-kernel離散化位相を除ける。

## 実行

- kernel:
  `kentookumura/exp438-u-state-fixed-lattice-exact-hmm-train`
- version / id_no: `1 / 129056676`
- runtime: private CPU、internet無効、Numba thread 1
- variant / HMM well-runs / folds: `1 / 32 / 5`
- parent HMM rerun / ML / booster / PF / Beam / GPU: すべて0
- fixed32はmechanism preflightであり、CVやpromotion evidenceではない

## 結果

| 項目 | 値 | gate |
| --- | ---: | --- |
| Stage 0 elapsed | 1,452.118秒 | report |
| Candidate HMM | 1,403.666秒 | report |
| Stage 1投影 | 33,907.307秒（上限30,600秒） | FAIL |
| Peak RSS | 1.133 GiB | PASS |
| Constant-Z parity max abs | 5.457e-12 ft | PASS |
| Brute-force max abs | 2.946e-08 | PASS |
| Transition row-sum max error | 3.331e-16 | PASS |
| Posterior normalization max error | 1.554e-15 | PASS |
| Quantization bias reduction | -43.580%（43.580%悪化） | FAIL |
| Forward-cause episode SSE reduction | -214.796% | FAIL |
| Persistent episode SSE reduction | -824.234% | FAIL |
| Persistent improved wells | 2/16（必要10/16） | FAIL |
| Persistent improving folds | 0/5（必要4/5） | FAIL |
| Control pooled RMSE delta | +43.320 ft | FAIL |
| Control by-well delta p95 | +72.481 ft | FAIL |

Technical 21項目のうち20項目はPASSした。coordinate/emission/readout identity、
constant-Z parent parity、brute-force reference、normalization、finite coverage、
truth-late ledger、readback SHAは成立している。一方、Stage 1 runtime投影は上限を
3,307.307秒超えた。

Mechanismは7項目すべてFAILした。posterior-weighted quantization bias sumは
親`3658.937 ft`からcandidate`5253.499 ft`へ増えた。persistent episode SSEは
`13,363,710.665`から`123,511,976.229`へ増え、全5 foldsで悪化した。
matched controlでも親`3.428436 ft`に対してcandidate`46.748911 ft`となった。

## 解釈

連続座標変換と数値実装は成立しているため、FAILは実装contract違反ではなく、
固定格子をTVTからabsolute Uへ移す科学差分に由来するnegative evidenceである。
0.35 ft離散化をU側へ固定するとquantization bias自体が増え、persistent/controlの
双方で大きく不安定化した。固定TVT格子のZ依存位相は単純な除去対象ではなく、
exp209では有効なregularization/support alignmentとして働いていた可能性が高い。

exp435、exp437、exp436とは変更軸が異なるため、それらの結論は再分類しない。
本結果を使ってgrid phase/anchor/step/band、noise、rate、emission、blend/selectorを
同じfixed32で調整することもしない。

## 再現性

- 初回runはdeterministic anchorではない。
- prediction logical SHA:
  `6a2096746b4b98bf192be5180b8f3763e528262b91bcedd8301ece927993753d`
- transition ledger logical SHA:
  `24ee57208298466bb215eb763f4073b2ab2f62e198d40be21a1105c2d3d68806`
- Kaggle metrics SHA:
  `7118b009a518aec5bf0bc2916bc38a273243202cc701ccf6912d61dc4f9f2322`
- gate report SHA:
  `ed4a3745acafe29e97e031442054d92c67137441ac181def8809f39468233ff2`

FAIL判定はmechanism劣化が極めて大きく、全foldへ再現しているため、
同一設定rerunによるanchor確立は行わない。

## 結論

固定absolute-U lattice仮説はStage 0で棄却する。Stage 1 full OOF、
inference、submissionへ進めない。再開には、現行格子のparameter rescueではない
独立したtarget-free仮説と、新しい実験・ユーザー承認が必要である。
