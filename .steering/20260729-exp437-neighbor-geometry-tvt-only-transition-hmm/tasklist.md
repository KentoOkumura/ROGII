# タスクリスト

## 目的

exp435のTVT-only HMMを維持し、transition centerだけをfold-safe exp226
`tvt_geop`隣接差へ置換するStage 0を、truth-lateかつfail-closedで実装する。

## 未着手

- なし。

## 閉鎖

- full 773-well Stage 1、raw-test regeneration、inference、submission。
- rate state、rate mixture、branch HMM、transition scale/noise grid、GR/emission変更。
- exp226 final unary/fallback、exp263 blend、selector、same-OOF rescue。

## 完了

- 2026-07-29: `exp437_neighbor_geometry_tvt_only_transition_hmm`として採番した。
- 2026-07-29: 親をexp435、geometry evidence parentをexp226へ固定した。
- 2026-07-29: `tvt_geop`隣接差だけをTVT-only transition centerへ入れる単一介入に固定した。
- 2026-07-29: exp355 joint-rate prior、exp394 soft branch、exp436 sparse fieldとの差を固定した。
- 2026-07-29: Stage 0 fixed32 mechanism gateとStage 1 full OOF promotion gateを固定した。
- 2026-07-29: route、実行量、禁止事項、truth-late、SHA、determinism契約を固定した。
- 2026-07-29: backlog、実験メタデータ、steeringを作成した。
- 2026-07-29: 明示実装指示を受け、Jupytext percent形式のcompact
  self-contained Stage 0 trainと正規train notebookを実装した。
- 2026-07-29: exp226 OOFのread-time 5列allowlist/decompressed SHA、
  exp435保存control logical SHA、fixed32 manifest SHAを検証するresolverを実装した。
- 2026-07-29: geometry first-difference schedule、5-cell direct transition、
  TVT-only forward-backward、truth-late readout、technical/mechanism AND gateを実装した。
- 2026-07-29: fail-closed inference guardとexp437専用contract testを追加し、
  構文、Ruff F821/E9、8 tests、Jupytext round-tripをPASSした。
- 2026-07-29: ユーザーの明示実行指示を受け、1 candidate ×32 wells、
  parent/control再実行0の契約を記録してKaggle private CPUへpushした。
- 2026-07-29: canonical kernel version 1（id_no `129056603`）を完了した。
  technical gateは全PASS、mechanism gateはmatched-control 2項目だけPASSし、
  残り5項目をFAILした。
- 2026-07-29: candidateはfixed32 allでexp226 geometryより
  `+3.751804309 ft`、persistent 16で`+6.823650264 ft`悪化し、
  改善fold `2/5`、by-well p95 / worstは
  `+21.699228790 / +24.452435654 ft`だった。
- 2026-07-29: `stage0_fail_closed_without_same_oof_rescue`として、
  Stage 1、再実行、救済、inference、submissionなしでterminal closeし、
  canonical configを実行不可へ再ロックした。

## 次アクション

なし。exp438 / exp439はexp437の救済ではなく独立仮説として扱う。
