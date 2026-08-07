# exp437_neighbor_geometry_tvt_only_transition_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical PASS / mechanism FAIL・terminal close
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 親: `exp435_tvt_memoryless_u_rate_dzonly_hmm`
- geometry evidence parent: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 実装: compact self-contained Stage 0 train、正規train notebook、
  専用contract test、fail-closed inference guardまで完了
- Kaggle Stage 0: private CPU version 1完了
- Stage 1 / inference / submission: 未実施・不適格
- notebook: trainは9 code cells / 11 markdown cells。inferenceは意図的に
  fail-closed

## 仮説

exp435のTVT-only状態への縮約自体ではなく、transition centerを毎行
`-ΔZ`へ置いたことが大幅悪化の主因かを検証する。exp226のgroup-safeな
geometry-only path `tvt_geop`の隣接差をtransition centerへ直接入れることで、
rate stateやbranch stateを復活させずに、周辺井戸から学習した符号付きの
`Δ(TVT+Z)`を状態遷移へ持ち込める可能性がある。

## 単一の変更点

exp435からTVT grid、start prior、position process noise、five-cell kernel、
typewell GR emission、forward-backwardを固定し、遷移中心だけを変更する。

```text
control:   mu(t) = -delta_Z(t)
candidate: mu(0) = tvt_geop(0) - last_known_TVT_input
           mu(t) = tvt_geop(t) - tvt_geop(t-1), t >= 1
```

candidateは`neighbor_geometry_direct_transition`の1本だけとする。
rate state、rate mixture、rate-to-rate transition、geometry branch、exp226の
GR correction / U projection / final prediction、blend、selectorは追加しない。

## 既存実験との差

- exp355はjoint `(TVT, U-rate)` HMMのrate-prior meanを変更した。
- exp394はgeometry branchとHMM branchをsoft-stickyに周辺化した。
- exp436は新しいglobal sparse potentialを解くdirect predictorである。
- exp437は保存済みexp226 `tvt_geop`の隣接差だけを、exp435の単一TVT chainへ入れる。

## 検証方針

- Fold / Group: exp226と同じouter 5-fold / `well_id`。
- Score: `TVT_input`欠損部の773 wells / 3,783,989 rows。
- Leakage: candidate freeze前は保存exp226 OOFの
  `well_id,row_idx,suffix_offset,tvt_geop,fold`だけをread時allowlistで読む。
- Control: exp226 geometry、exp226 final、exp435 fixed32 predictionを保存値で使い、
  parent/control HMMは再実行しない。
- Stage 0: fixed32の1 candidate × 32 HMM well-runs。機構確認専用で、CVやpromotion
  evidenceとは呼ばない。
- Stage 1: Stage 0の全AND gate PASSと別承認後だけ、1 candidate × 773 HMM
  well-runsのfull group-safe OOFを行う。
- RNG: なし。fold / well / row / variant / reduction順を固定する。

### Stage 0の主要gate

- exp226 geometry fixed32 allより`0.10 ft`以上改善。
- matched controlでexp226 geometry比`+0.02 ft`以内。
- persistent 16でexp226 geometryより`0.10 ft`以上改善。
- matched controlでexp435 dz-onlyより`1.0 ft`以上改善。
- exp226 geometry比で4/5 folds改善。
- by-well delta p95 `<=+0.25 ft`、worst `<=+2.0 ft`。
- 技術、leakage、SHA、runtime、memory gateを含む全条件のAND。

### Stage 1の主要gate

- pooled RMSE `<=9.377109596582213`。
- exp226 final比`0.05 ft`以上、exp226 geometry比`0.20 ft`以上改善。
- exp226 final比で4/5 folds改善。
- 1000+、hidden-like 2面、near、by-well tailの全guardを通過。

## 実行入口

Jupytextの正は
`exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py`
であり、正規train notebookへ同じセル構造を採用済みである。Stage 0完了後は
`execution.run_hmm=false`、`create_prediction=false`、
`design.kaggle_stage_0_authorized=false`へ再ロックした。

実装済み範囲:

- exp226 OOFをread時5列allowlistとdecompressed SHAで読む。
- exp435 fixed32保存予測をlogical SHAで固定し、dz-only controlだけを比較に使う。
- `tvt_geop`隣接差を直接中心とする5-cell TVT-only forward-backwardを1 well 1回実行する。
- 32 wellsすべてのschedule/prediction/diagnostic SHAをfreeze後にだけ
  role、fold、suffix truthを結合する。
- exp226 geometryとexp435 dz-onlyに対するtechnical/mechanism AND gateを出力する。

Stage 0でもparent/control HMMは再実行していない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| fixed32 candidate RMSE | 13.019009088 |
| fixed32 exp226 geometry RMSE | 9.267204778 |
| 差 | +3.751804309 |
| 改善fold | 2/5 |
| by-well delta p95 / worst | +21.699229 / +24.452436 ft |

technical gateは全PASSしたがmechanism gateはFAILした。matched controlでは
candidate `7.771562`がexp226 geometry `8.719886`を改善した一方、
persistent 16では`16.592455 vs 9.768805`と大幅悪化した。

## 所見

### 設計上の利点

- 「TVT-only縮約」と「`-ΔZ`中心」を分離し、後者だけを検証できる。
- exp226 final predictionの後処理利用ではなく、周辺井戸情報をHMM内部の一行遷移へ入れる。
- 保存controlを再実行しないため、新candidate以外の計算と比較条件を増やさない。

### リスク / 注意

- exp226 geometryの低周波biasをHMMがそのまま追従する可能性がある。
- 強いGR emissionがgeometry driftを修正するとは限らず、別のmode slipを作り得る。
- fixed32は機構偏重の選択標本であり、全OOF性能を推定できない。
- Stage 0 FAIL時にscale、clip、noise、emission、grid、subset、gateを同じOOFで救済しない。

## 次

Stage 1、same-OOF救済、raw-test geometry再生成、inference、submissionへ進まない。
exp438 / exp439は独立仮説として扱う。

## 表記

用語は`KAGGLE_DIRECTION.md`と`docs/glossary.md`に合わせ、日本語優先で記録する。
