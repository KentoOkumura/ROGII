# 要件

> 2026-07-23更新: 当初のdesign-only境界後、ユーザーがexp360実装とKaggle実行を
> 順に明示依頼した。Stage 0はprivate CPU version 2で完了し、固定technical /
> scientific gateをFAILしたためbranchを閉じた。推論、提出は未承認・未実施。

## 依頼

- 全 well に対して、通常の Type Well GR matching（`δ=0`）を残したまま、複数の Type Well TVT reference shift を同時に評価する。
- shift の有無を未知としたまま、shift score surface が exp264 の誤差を予告する confidence readout になり得るかを OOF で検証する。
- exp340 の negative result を閾値調整で救済せず、exp280 の raw absolute Gaussian score を raw-finite ZNCC に置き換える独立仮説として扱う。
- この段階では予測、候補、selector、学習モデル、推論、提出を変更しない。
- 当初依頼では設計のみを確定し、後続の明示依頼で実装・実行へ進む。

## 制約

- Route: `ensemble`
- 親実験: `exp340_exp226_depth_alias_block_confidence_readout_on_exp264`
- 比較対象: `exp280_exp226_rawgr_likelihood_confidence_readout_on_exp264`
- 予測評価対象: `exp264_selector_compact_addonly_on_exp226`
- scientific path / fold / block / shift bank は exp226 / exp280 / exp340 の既存契約を固定する。
- shift は horizontal GR の row/MD を動かす操作ではなく、`GR_typewell(TVT_geop + δ)` と Type Well 側の参照 TVT を動かす操作とする。
- 対象は 773 OOF wells の unknown suffix 全体とし、元の相関が悪い well だけを事後選択しない。
- `δ=0` は常に候補に含め、shift 適用を既定値にしない。
- 固定 shift bank は `[-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80] ft` とし、000d7d20 等に合わせた dense grid を追加しない。
- 水平井 GR は raw finite 値だけを使い、補間や欠損埋めを行わない。
- 予測対象 TVT、actual TVT、exp264 error、bad10 label は score / feature / control / quantile / manifest の freeze 完了前に参照しない。
- 000d7d20、8cc21f01、e5ff9fd2、2ddad940 は説明用 sentinel とし、閾値、shift bank、合否判定の調整に使わない。
- exp340 の既存 family、閾値、blend、selector を再調整しない。
- 実装、Kaggle package、push、実行、output 取得、提出は別途承認があるまで行わない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

### 設計完了

- `.steering/20260723-exp360-typewell-reference-shift-zncc-confidence-readout/` に仮説、単一変更、固定条件、leakage barrier、control、合否判定、停止条件を記録している。
- design freeze時点では`experiments/exp360_typewell_reference_shift_zncc_confidence_readout/`
  をscaffoldに留め、その後の明示実装依頼でStage 0を追加する。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` に設計を記録し、後続実装・実行結果で
  completed gate FAIL / closedへ更新している。

### 将来の Stage 0 実装・実行

- 期待する 773 wells、5 folds、7,787 blocks と exp226 / exp280 / exp340 の lineage・SHA が一致する。
- core-supported block coverage が 0.98 以上で、773 wells 全てに少なくとも1 supported block がある。
- freeze 前の truth / target / error / label 参照回数が 0 である。
- fold 別 Q1/Q4 境界、score bank、valid mask、stable control、feature table、manifest の content SHA を truth join 前に記録する。
- primary family `best_nonzero_minus_zero_zncc` が、次の preregistered gate を全て通過する。
  - Q4−Q1 の平均 block RMSE 差が `+0.50 ft` 以上で、中央値差も正。
  - 正方向が 5 folds 中 4 folds 以上。
  - pooled row-weighted bad10 AUC が `0.60` 以上で、AUC `>0.50` が 5 folds 中 4 folds 以上。
  - 1000+ wells と hidden-like spatial/typewell-purged scope の双方で正方向。
  - 対応する exp280 raw Gaussian family より pooled AUC が `+0.02` 以上高く、5 folds 中 4 folds 以上で改善。
  - stable shift-label permutation control より pooled AUC が `+0.02` 以上高く、5 folds 中 4 folds 以上で改善。
- primary gate が1つでも未達なら、この branch を閉じる。他 family や sentinel well の結果で救済しない。
- primary gate を全通過しても、本実験内では予測を変更せず、別の add-only ML feature 実験を提案するだけとする。
- 本実験は deterministic anchor にしない。model / prediction / submission は生成しないため、それらの SHA は `not_applicable` と記録する。
- gzip 入力を比較する場合は raw gzip SHA に加え、decompressed content SHA を主証拠として記録する。
