# exp394_soft_sticky_exp226_k16_branch_hmm

## 状態

- ルート: `pf_beam`
- 状態: fixed16 technical preflight runtime gate FAIL・closed
- CV / LB / Submit: なし
- 作成日: 2026-07-25
- 親実験: `exp355_exp226_dip_rate_prior_on_exp209`
- 主比較: exp263 fixed physical candidate OOF `8.238331546`

## 仮説

exp226 geometry path と free exact HMM は異なる区間で誤差を持つ。exp226 pathを
1 state branch、exp226/K16 rate priorを持つ全TVT-grid HMMをもう1つの branch とし、
GR 観測と持続性 priorで同時に周辺化すれば、固定blendより安全に誤mode吸着を抑えられる。

## 確定したモデル

- `E branch`: group-safe exp226 `tvt_geop`。GR correction / U projection 前。
- `H branch`: exp209 の全 absolute-TVT grid × 41 residual-rate states。
- H transition mean: exp355 の K16 relative geometry rateをknown-prefix rateへ再anchor。
- GR emission: 両branchともexp209 Gaussian。Eでは`v=tvt_geop`として評価。
- regime: 初期50/50、base switching length 1,000 MD-ftのsoft-sticky。
- H→E: H kernelの次TVTがexp226 pathから6 ft以内であるほど接続しやすく、
  dockingによりH側の滞在長は1,000 MD-ft以上になり得る。
- 出力: regimeとHMM stateを全て周辺化したjoint posterior mean。

低ランク3D地層場、finite HMM mode bank、MAP/Viterbi/top-K、Huber/Student-t、
selector、後段blendは含めない。

## 検証方針

- 16-well preflightはfinite、normalization、full-grid coverage、runtime、RSSだけを確認する。
  小標本RMSEで773-well候補を止めない。
- full OOFは別承認後に1 variant / 5 folds / 773 switching-HMM runs。
- LightGBM config / trained fold / booster / parent control rerunは全て0。
- 主promotion条件はexp263比`0.25 ft`以上、4/5 folds、stressとwell-tailの安全性。
- sticky length、docking幅、emission、gridの同一OOF tuningは禁止する。

## 過去結果からの見込み

exp355はK16 relative rateでexp209を`0.646311 ft`、5/5 folds改善しており、
相対rate signalはある。一方でhidden-likeとworst wellが悪化した。exp226単独は
OOF `9.427110`、exp263固定物理blendは`8.238332` / Public LB `7.800`である。
したがって統合を試す根拠はあるが、LB 6.5を実現できるという直接証拠はまだない。

## 実行入口

`exp394_soft_sticky_exp226_k16_branch_hmm_compact_selfcontained_train.py`を
Jupytextの正として正規train Notebookへ採用した。設定では承認済みfixed16
technical preflightだけを有効化し、full OOF / inference / submissionは無効にする。

## 生成物契約

- preflight: 固定16-well ledger、rate schedule、prediction/branch posterior、
  by-well runtime、technical gate、summary。
- full OOF: prediction、branch posterior、schedule、scope/by-well metrics、
  persistent-offset episode/recovery、promotion gate、summaryと各logical SHA。
- suffix truth、exp263 baseline、hidden-like roleはprediction/branch SHA freeze後だけ読む。

## 結果

実装と専用testは完了。Kaggle private CPU version 1（id_no `128536142`）は
fixed16を`3703.079064 sec`で完了した。数値、coverage、leakage、RSSはPASSしたが、
full runtime projection `112,736.889439 sec`が上限`30,600 sec`を超えてFAILした。
RMSE、full OOF、CV、LBは未実行。

## 所見

- exp355の平均signalとexp226の独立pathを統合する仮説は検証価値がある。
- 既存の最良物理baseline exp263との差は大きく、tail安全性も未確認である。
- 実装完了を実験結果として扱わず、LB 6.5を達成可能と断定しない。

## 次

fixed runtime gateに従い、773-well full OOFへ進めず閉じる。再訪時も
sticky/model/grid/gateの救済はせず、同じfixed16出力を維持する計算最適化を
独立technical auditとして先に検証する。
