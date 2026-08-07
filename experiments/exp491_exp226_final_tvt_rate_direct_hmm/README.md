# exp491 exp226最終rate直接入力HMM

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU Stage 0 version 2完了・`stage0_fail_closed`
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-30
- 親実験: `exp437_neighbor_geometry_tvt_only_transition_hmm`
- 予測源: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp209 HMMでは、初期rateとstickyなrate状態の追従遅れが長距離のTVT offsetを
作っていた。fold-safeなexp226最終`TVT`予測の一行差分を毎行の遷移中心へ直接使えば、
persistent rate状態を持たずに局所rateを与えられる。exp226に残るabsolute offsetは、
固定したTVT位置posteriorとtypewell GR emissionで補正できる可能性がある。

## 変更点

- exp437のTVT-only HMMを構造参照とする。
- transition centerだけを`tvt_geop`隣接差からexp226最終`tvt_pred`隣接差へ変える。
- 最初のunknown rowは`tvt_pred[0] - last_known_TVT_input`、以降は
  `tvt_pred[t] - tvt_pred[t-1]`をそのまま使う。
- rateの平滑化、clip、scale、K16集約、momentum、rate stateを使わない。
- GR emission、TVT grid、position noise、start prior、forward-backwardはexp437から固定する。

## 検証方針

- Stage 0: 既存fixed32の16 persistent + 16 matched control、候補1本、
  32 HMM well-runs。機構確認でありCVではない。
- Stage 1: Stage 0全gate PASSと別承認後だけ、773 wells /
  3,783,989 rowsのgroup-safe OOFを候補1本で評価する。
- primary比較: 保存済みexp226 final OOF、RMSE `9.427109596582213`。
- control再実行、LightGBM、PF、Beam、GPUはすべて0。
- candidateのschedule・prediction・診断をfreezeした後だけsuffix truth、
  persistent/control role、fold別結果を結合する。

## 主要リスク

- exp226最終`tvt_pred`は既にsuffix GRを使っており、HMMも同じGRを再利用する。
  target leakageではないが、同じ観測を二重に使う一般化リスクがある。
- exp226の局所rateが正しくてもabsolute offsetを継承する可能性がある。
- exp437のgeometry-only direct transitionはpersistent 16で悪化しており、
  final predictionへ変えるだけで安全になる保証はない。

## 所見

version 1は32/32 wellsのHMM計算後、truth-late readout前のgzip artifact
readbackで技術失敗した。close順を修正した同一契約のversion 2は完了し、
technical gateを全件PASSした。一方、all32はexp226 final
`7.976057 ft`に対してcandidate `12.290251 ft`（`+4.314194 ft`悪化）、
persistentは`8.757067 → 16.169236 ft`だった。mechanism gateは
matched-control safetyだけPASSし、残り6件はFAILした。

## 生成物

compact self-contained Stage 0 source / notebook と契約テストを実装した。
Kaggle version 2で予測、well / episode metrics、summaryを生成し、SHAをログで
記録した。modelとsubmissionは対象外である。

## 実行入口

- Stage 0実装:
  `exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.ipynb`
- 正規train notebook scaffold:
  `exp491_exp226_final_tvt_rate_direct_hmm_train.ipynb`
- inference notebook:
  `exp491_exp226_final_tvt_rate_direct_hmm_inference.ipynb`
- 2026-07-31の実行指示によりcompact実装を正規trainへ採用し、
  `run_hmm: true`、private CPU、internet/GPU offでpackageを生成した。
  inference scaffoldは上書きしていない。

## 次

exp491はStage 0で完了・fail-closedとする。Stage 1、PF救済、inference、
submissionへ進まない。
