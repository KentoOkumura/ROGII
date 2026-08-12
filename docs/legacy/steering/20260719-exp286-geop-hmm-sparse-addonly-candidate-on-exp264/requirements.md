# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `geop_hmm_sparse_addonly_candidate_on_exp264` を、
`exp286_geop_hmm_sparse_addonly_candidate_on_exp264` として実装する。

Stage 0完了後、ユーザーから「固定gateの有効性ではなく、RMSEが有望な`geop_hmm`を
他候補と同じ情報付きでselectorへ追加し、改善するかを再学習で確認する」と明示された。
この追加依頼をexp286の同一仮説の続きとしてStage Bへ実装する。

## 制約

- Route: `ensemble`。
- 親は raw-test-safe 修正版 `exp264_exp263_candidate_confidence_dual_selector`、追加候補は
  保存済み `exp279_exp226_geop_centered_exact_hmm_redecode` の `geop_hmm` とする。
- Stage 0は保存済みOOFだけを読む1 audit variant、LightGBM config / trained fold / booster
  `0 / 0 / 0`として完了済み。固定gateはfull whole-well gain保持率27.71%で失敗したが、
  これはsparse gate分岐だけを閉じる結果とする。
- Stage Bはexp264の既存12候補を保持し、`geop_hmm`を全well・全OOF行でavailability=1の
  13番目primitiveとしてselectorへ追加する。gate、NaN化、既存候補による補完は行わない。
- `geop_hmm`も他候補と同じcandidate-long契約に載せる。string candidate ID、one-hot ID、
  kind、family、availability、finite flag、anchor/local-shape/bank proxyに加え、native confidence
  `sigma_tvt`、`source_loglik`、`loglik_per_row`、`candidate_finite_source`、
  `confidence_valid`を保持する。
- gate は exp226 geometry/GR、exact HMM confidence、既存候補 disagreement、MD/tail/known-prefix だけから、
  truth join 前に固定する。true TVT/error/oracle、well ID、`geop_hmm-exact_hmm` 差は禁止する。
- Stage Bが元selector比で改善したため、ユーザーは2026-07-19にStage CからStage Dまでの実行を
  明示承認した。Stage Cは1 variant x 2 objectives x 5 outer x 4 inner = 40 CPU boosters、
  Stage Dは新規full13 compact add-onlyだけ3 configs x 5 folds = 15 GPU boostersとする。
- 親/controlは再学習せず、保存済みexp264 Stage D controlおよび12候補compact add-onlyを比較基準にする。
  HMM/PF再生成、inference、submissionは行わない。
- 再現性は `docs/06_reproducibility.md` に従い、入力 SHA、truth-free gate schema/content SHA、
  readout SHA、Kaggle kernel version を記録する。

## 受け入れ基準

- compact self-contained Jupytext train source と disabled inference sourceを作り、正規 `.ipynb` に変換する。
- Stage 0 が row / 512-block / whole-well oracle、unique-best、5 folds、1000+、hidden-like、
  by-well worst regression、top-25% gate retention を生成できる。
- gate feature名、単調方向、等重み rank、cutoff、truth attachment順序を機械可読 contract に保存する。
- 13候補unionが3粒度すべてpooled改善し、各粒度4/5 folds以上改善する guard を固定する。
- gate が full `geop_hmm` whole-well oracle SSE gain の50%以上を保持し、selected-well gainが
  pooledかつ4/5 foldsで正となる guard を固定する。
- 200-well paired shadow runtime manifest が無い場合は runtime guard を PASS にせず、
  gate coverage 25%/50 wells、geop p95 45分、total p95 7.5時間を固定判定する。
- deterministic anchor としては扱わず、prediction/model/submission SHA は対象外理由を記録する。
- 13候補contractのID順、2 legal domain、confidence mappingを機械可読に保存する。
- Stage A/Bで`id__candidate__geop_hmm`が採用schemaに残り、`geop_hmm`のavailabilityと
  4 native confidence fieldのcoverageが全foldで1.0であることを検証する。
- 保存済みexp264 Stage B v5を再学習せず12候補baselineとし、13候補版のhard selector RMSE、
  fold RMSE、rank regret、top-3 oracle coverage、選択率、shared-12 candidate scoreを比較する。
- selector追加による改善guardは13候補hard RMSEが12候補hard RMSEを下回り、3/5 folds以上改善し、
  `geop_hmm`が少なくとも1行選択され、13候補score guardがPASSすることとする。
- Stage Bはmodel manifest 10件、feature/candidate-score/compact/model SHAを保存する。
- 構文、F821/F401/F841/E722、専用 tests、Jupytext test、strict experiment validation が通る。

## 完了結果と次

- Stage Bはparent12 `8.587004 -> 8.477740`、3/5 foldsでselector-addition guardをPASSした。
- Stage Cはparent12 `8.652532 -> 8.448682`、4/5 foldsでscore/leakage guardをPASSした。
- Stage Dはparent12 add-only `8.460811 -> 8.403784`へpooled改善したが、2/5 folds、
  worst-well `+5.862833 ft`で総合guardをFAILした。
- exp286は完了とし、inference/submissionへ進めない。必要なら保存済みOOFだけを用いるtarget-free
  tail-risk attributionを別承認の0-booster readoutとして検討する。
