# タスクリスト

## TODO

- なし。Stage 0 FAILにより実験を閉鎖した。

## ブロック中

- Stage 1 full OOF、inference、submission。
- grid anchor/step/band/phase、position noise、rate/emissionの追加variant。
- source/destination rate台形transitionとの同時変更、blend、selector。

## 完了

- 2026-07-29: `exp438_u_state_fixed_lattice_exact_hmm`として採番した。
- 2026-07-29: 連続座標変換は親と同値で、固定格子の離散化だけが差分だと明確化した。
- 2026-07-29: last-known Zで固定するU格子、arrival-rate transition、
  row-wise TVT emission/readoutを固定した。
- 2026-07-29: Stage 0 fixed32、Stage 1 full OOF、no-rescue、
  truth-late、数値contract、実行量を確定した。
- 2026-07-29: compact self-contained Stage 0 train候補とfail-closed
  inference候補をJupytext percent形式で実装した。
- 2026-07-29: joint fixed-U HMM、coordinate/emission/readout identity、
  constant-Z parent parity、独立exp209回帰、exhaustive small-path reference、
  transition/posterior normalizationを実装した。
- 2026-07-29: truth-free quantization ledger、全fixed32 freeze後の
  role/fold/truth/episode join、technical/mechanism AND gateを実装した。
- 2026-07-29: `py_compile`、Ruff F821、専用test 12件、
  Jupytext round-trip、strict experiment/template validationをPASSした。
- 2026-07-29: ユーザー承認後、compact候補を正規notebookへ採用し、
  strict no-src packageをprivate CPU canonical kernelへpushした。
- 2026-07-29: Kaggle version 1（id_no `129056676`）がCOMPLETE。
  technicalはruntime projectionだけFAIL、mechanismは7項目すべてFAILした。
- 2026-07-29: `stage0_fail_closed`とし、Stage 1、inference、submission、
  same-fixed32 rescueへ進まないことを確定した。
