# exp302_exp226_multiscale_k_segment_candidate_audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 2完了・direct FAIL・candidate novelty PASS
- CV: K24 `9.413244`、K12 `9.551938`、保存済みK16 `9.427110`
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-20
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp226の幾何segment数をK=16からK=12またはK=24へ1変数だけ動かすと、direct OOFまたは
exp293 fixed deployable12へ追加したときのcandidate noveltyが改善する可能性がある。

## 変更点

- scientific variantは`K=12`と`K=24`だけ。
- 保存済み`K=16` OOFをcontrolにし、再生成しない。
- exp226のその他のパラメータ、5 fold、score rowsを固定する。
- direct qualityとadd-one candidate noveltyを独立guardで判定する。
- HMMの`step`/`n_rates`、blend、selector、inference、submissionは変更しない。

## 検証方針

- Fold: exp226保存済み5 fold identity
- Group: `well_id`完全分離
- Score rows: train unknown suffix
- Direct primary: pooled RMSE、4/5 folds、1000+、hidden-like 2面、by-well p95/worst
- Novelty primary: exp293 fixed12に各variantをadd-oneしたH512/whole-well oracle
- Leakage check: candidate predictionとblock manifestをtruth-freeでSHA freezeし、別loaderでtruthを後結合

## 固定PASS条件

- Direct PASS: pooled `<=9.3771096741`、4/5 folds改善、1000+/hidden-like各`<=+0.02 ft`、
  p95/worst各`<=+0.25 ft`。
- Candidate novelty PASS: H512 oracle `>=0.03 ft`改善、whole-well `>=0.02 ft`改善、
  H512 strict unique-best `>=2%`、4/5 folds改善。
- exp303へ進めるのはcandidate novelty PASS時だけ。

## 実行契約

- 2 variants × 5 folds = 10 variant-fold runs
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- control再生成: 0
- CPU、推論なし、提出なし

## 実行入口

- 実装元: `exp302_exp226_multiscale_k_segment_candidate_audit_compact_selfcontained_train.py`
- train: compact self-contained版を正規`_train.ipynb`へ採用済み
- inference候補: raw testを読まず停止する`_compact_selfcontained_inference.py` / `.ipynb`
- 正規train Notebookは2026-07-20の実行承認で採用した。正規inference Notebookはplaceholderのままで、実行対象外。

## 結果

- technical guard: PASS
- direct guard: K12/K24ともFAIL。K24はK16比`-0.013865 ft`だが3/5 folds改善、
  pooled閾値`9.377110`未達。
- candidate novelty guard: K12/K24ともPASS。H512 oracle改善は`+0.066095 / +0.083901 ft`、
  strict unique-bestは`10.6973% / 10.8899%`、両方5/5 folds改善。
- exp303 dependency: exp302側だけ充足。exp276の完了+promotion guard FAIL待ち。

## 所見

- K16のdirect不採用とcandidateとしての有用性を分けて評価する設計にした。
- K gridを2点に限定し、改善がなければ救済探索へ進まない。
- compact self-contained train sourceにはexp226数値核、exp293 fixed12再構成、freeze、late truth join、
  direct/novelty判定、SHA保存を実装した。
- Kaggle output manifest 16/16件のfile SHAと、K12/K24/blockのdecompressed SHAを照合した。

## 次

direct候補への昇格は行わない。exp276がpromotion guard FAILで完了した場合だけ、既に設計済みの
exp303を別途検討する。inferenceとsubmissionは引き続き対象外。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせる。
