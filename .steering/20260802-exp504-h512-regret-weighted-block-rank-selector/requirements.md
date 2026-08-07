# 要件

## 依頼

現状の行ごとのselectorとは異なる枠組みとして、固定長区間をqueryとするrank学習を
設計する。バックログ、実験ディレクトリ、steeringを作成して設計を確定するが、
実装、学習、Kaggle実行、推論、提出はまだ行わない。

## 仮説

exp264系の行別absolute-error分類・回帰は、長区間で一貫して誤った候補を高信頼に
選ぶことがある。exp293で大きなoracle headroomが確認されたH512区間をqueryとし、
候補間の誤差差を重みにしたpairwise rank学習へ目的を変えることで、行ごとの局所的な
順位揺れを減らし、固定anchorより低いOOF RMSEを得られる可能性がある。

## スコープ

- 実験: `exp504_h512_regret_weighted_block_rank_selector`
- Route: `ensemble`
- 主親: `exp293_physics_only_candidate_bank_headroom_contract`
- feature / selector契約参照: `exp264_exp263_candidate_confidence_dual_selector`
- 原因診断参照: `exp300_exp264_candidate_rank_misranking_readout`
- 負の比較参照: `exp348_h512_neural_path_bank`
- 対象: 保存済みfixed12候補、3,783,989行、773 wells、外側5 folds、H512のみ
- 実行単位: 1 scientific variant、1 rank config、5 outer-fold CPU models
- 再学習・再生成: 親control 0、PF/HMM/Beam 0、GPU 0

## 固定要件

- exp293のcandidate順、H512 block assignment、outer foldを変更しない。
- 候補は12個のままとし、追加・削除・再計算・blendを行わない。
- corrected exp264のraw-test-safe 88列schemaだけを行特徴の入口とする。
- `MD/X/Y/Z/GR`以外のtrain-only horizontal列およびTVT/error/oracle情報をfeatureにしない。
- rank labelとweightはouter-train truthだけから作る。outer-valid truthは候補選択予測と
  prediction SHAを凍結した後の評価でのみ読む。
- H128、H256、whole-well、可変長区間、overlap、horizon gridを同一実験で試さない。
- rank loss、pair weight、モデル設定、anchor guardの閾値をsame-OOFで探索しない。
- 選んだ候補をH512 block内の全行に適用し、評価は元の行単位RMSEで行う。

## 受け入れ基準

設計完了の受け入れ基準は次のとおり。

- steering 3文書、`config.yaml`、README、SESSION_NOTES、result、metricsを作成する。
- candidate順、入力SHA、block/query、feature集約、pair label/weight、モデル、rank集約、
  tie/anchor guard、CV、成功条件、禁止事項を曖昧さなく固定する。
- 将来の実行量を `1 variant × 1 config × 5 folds = 5 CPU models` と記録する。
- train / inference notebookはmarkdown-only placeholderとし、実装・実行できない状態にする。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へdesign-onlyとして登録する。

科学的PASSは将来実装・実行した場合にのみ判定し、全ANDとする。

- fixed anchor `exp226_w500_50_50` RMSE `8.238331546`から0.05 ft以上改善する。
- 5 fold中4 fold以上でanchorに非劣化とする。
- `0--250`、`250--1000`、`1000+`、hidden-like spatial、hidden-like typewell-purgedの
  各scopeでanchor比 `+0.02 ft`以内とする。
- by-well RMSE deltaのp95とworstをそれぞれ `+0.25 ft`以内とする。
- technical gateをすべて通す。

PASSでも推論実装やdownstream feature化は自動承認しない。FAIL時は本実験を閉じ、
H128/H256、loss、weight、thresholdの救済gridを行わない。

## 2026-08-02 追加依頼: 実装

ユーザーの「exp504を実装してください」を、凍結済み科学契約を変更しない実装承認として
扱う。別名のJupytext percent形式compact self-contained train source、候補Notebook、
contract test、静的検証、実験記録までを対象とする。

- 既存の正規train/inference placeholder notebookは上書きしない。
- compact train候補はfixed12再構成、corrected exp264 88列生成、H512固定9統計、
  ordered-pair両方向、regret weight、5 outer-fold CPU LightGBM、反対称化Borda、
  anchor guard、truth-late評価、必須readout、SHA生成物を実装する。
- `ctx__` 22列はshared block context、残り66列はcandidate-specific block vectorとし、
  pair幅を`3 × (66 × 9) + (22 × 9) + 6 = 1,986`列に固定する。
- NDCG@1の実装詳細はcandidate MSEのquery内rankをlinear relevance
  `12-rank`へ変換し、selected relevance / best relevanceとする。promotion判定には使わない。
- 正規Notebook採用、Kaggle package、push/run、inference、submissionは追加承認まで行わない。

## 2026-08-02 追加依頼: Kaggle train実行

ユーザーの「実行してください」を、凍結済みtrain候補の正規Notebook採用、Kaggle CPU package、
push/run、完了監視、train-side OOF記録の承認として扱う。実行量は1 scientific variant、
1 LightGBM config、5 outer folds、合計5 CPU boostersで、親/control再学習、候補再生成、
PF/HMM/Beam、GPUは0とする。inference、submission、downstream昇格は承認範囲外とする。
