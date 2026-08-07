# exp394 要件

## 依頼

exp226 と exact HMM の誤差要因が異なる可能性を利用し、次の 2 branch を
GR 観測で soft-sticky に周辺化する物理モデルの設計を確定する。

1. exp226 の fold-safe geometry-only TVT path を保持する branch
2. exp226/K16 の相対 rate schedule を遷移平均に使い、全 absolute-TVT grid の
   多峰性を保持する exact-HMM branch

このターンでは backlog、steering、実験 scaffold だけを作り、実装、Kaggle package、
train、inference、submission は行わない。

## 2026-07-25 実装承認追記

後続のユーザー指示「exp394を実装してください」により、上記の design-only 制限のうち
source、compact self-contained Notebook候補、unit testの実装が承認された。
Kaggle package、push、16-well preflight実行、full OOF、inference、submissionは
引き続き未承認とする。

## 2026-07-25 fixed16 technical preflight実行承認追記

後続のユーザー指示「実行してください」により、正規train Notebook採用、
Kaggle package/push、および固定16-well technical preflightだけが承認された。
実行量はtechnical candidate 1、switching-HMM well runs 16、
LightGBM config / trained fold / booster / parent-control rerun / GPUは全て0。
固定well選択は5 reporting foldsを覆うが、preflightではRMSEを計算しない。
full 1 variant / 5 reporting folds / 773 HMM well runs、inference、submissionは
引き続き未承認とし、technical PASS summary SHA凍結後に別承認を必要とする。

## 2026-07-25 fixed16実行結果

Kaggle private CPU canonical version 1（id_no `128536142`）は
`3703.079064 sec`で16 wells / 140,721 rowsを完了した。finite prediction、
H full-grid、identity、leakage、posterior normalization、transition row sum、
RSSはPASSしたが、full runtime projectionは`112,736.889439 sec`となり、
固定上限`30,600 sec`の`3.684212x`でFAILした。RMSEは計算していない。
よってこれはscientific negativeではなくtechnical blockerであり、full OOF、
inference、submissionは実行せずexp394を閉じる。

## 目的

- 第一目的は、保存済み exp263 OOF `8.238331546 ft` を安全に上回る物理候補を作ること。
- Public LB `6.5` は到達目標であり、この実験単独の既存証拠から保証できる成功条件ではない。
- exp226 branch と HMM branch の区間別の信頼度を、target GR と typewell GR の
  尤度および持続性 prior だけで推定する。

## 必須設計

- Route は `pf_beam` とする。LightGBM、selector、PF、Beam は使わない。
- regime は `E=exp226 geometry path` と `H=free exact HMM` の 2 状態だけとする。
- `E` の候補 TVT は group-safe exp226 OOF の `tvt_geop` とする。
  exp226 の `gr_delta`、最終 `tvt_pred`、U projection は使用しない。
- `H` は exp209 と同じ per-well absolute-TVT grid と 41 residual-rate states を
  全て保持する。手作業の `HMM mode 1, 2, ...` や top-K path bank は作らない。
- `H` の遷移平均は exp355 と同じ exp226/K16 geometry-only relative rate schedule
  とし、既知 prefix の rate に再 anchor する。この値は GR 補正前である。
- `E` と `H` の emission は同じ exp209 Gaussian raw/typewell-GR 観測モデルを使う。
- regime switch は MD-aware soft-sticky Markov prior で forward-backward に含める。
- 最終 TVT は regime と HMM state を全て周辺化した posterior mean とする。

## 固定値

- 初期 branch prior: `P(E)=0.5`, `P(H)=0.5`
- regime の base switching length: `1000 MD-ft`
- HMM から exp226 branch へ接続するときの docking 幅: `6.0 ft`
- HMM grid / transition / Gaussian emission: exp209 固定
- K16 segmentation / relative rate schedule: exp355 固定
- 同じ OOF を見ながら sticky 長、docking 幅、emission、grid、rate scale を探索しない。

## 検証契約

- 16 wells の preflight は数値安定性、state coverage、推定 runtime、peak RSS だけを
  判定する。RMSE、fold 改善、hidden-like score は full run への gate にしない。
- 科学評価は統合 candidate 1 本、773 wells、5 reporting folds の full OOF 後だけ行う。
- 親 control は再実行せず、保存済み exp226、exp209、exp263、exp355 を比較に使う。
- full OOF の主比較は exp263 fixed physical candidate `8.238331546 ft` とする。
- prediction と branch posterior を freeze して SHA を記録した後にだけ suffix truth、
  error、hidden-like role を結合する。

## 実験数と計算量

- scientific variant: 1
- reporting folds: 5
- switching exact-HMM well runs: 773
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control rerun: 0
- GPU: 0、Kaggle CPU

## 対象外

- 低ランク 3D 地層場、formation surface、RGT graph
- exp226 最終予測または GR correction の再利用
- Student-t、Huber、GR sigma、transition noise の同時変更
- finite mode bank、MAP、Viterbi、top-K path の選択
- hard router、rowwise selector、oracle branch 選択、post-hoc blend
- parameter grid、同一 OOF rescue、inference、submission

## 実装完了の受け入れ基準

- `.steering` の requirements / design / tasklist が上記 contract を矛盾なく固定している。
- compact self-contained train sourceと別名Notebook候補があり、正規placeholderは
  明示採用まで上書きしない。2026-07-25の実行承認後は正規trainへ採用する。
- dense全列挙parity、transition row-sum、posterior normalization、missing GR、
  K16 schedule、leakage、16-well選択のtestがある。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` に exp394 が記録されている。
- implementationは有効で、承認済みfixed16 preflightだけを実行可能にし、
  full OOF / inference / submissionは無効で、結果値を捏造していない。
