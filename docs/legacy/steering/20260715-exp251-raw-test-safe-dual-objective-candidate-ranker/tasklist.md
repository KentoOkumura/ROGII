# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp251 steering/scaffoldを作成した。
- `docs/06_reproducibility.md`を確認し、stage、seed、SHA、bootstrap方針を設計した。
- raw-test candidate/HMM/multi-observation surface再生成を実装した。
- 297列provenance/fallback/distribution auditとselected schema/sample SHA保存を実装した。
- same-run audit passでのみ10 CPU boosterへ進むguardを実装した。
- fixed Viterbi評価、overall/1000+/hidden-like/worst-well guard、model/prediction SHA保存を実装した。
- inference停止notebookを実装した。
- synthetic feature-audit contract test、py_compile、Ruffを通した。
- steeringを、OOF-only一律除外からcross-fit/full-train再生成contractへ訂正した。
- `raw_test_regenerated_copcf`のfull-train typewell/spatial prior生成を実装した。
- test typewell cluster再割当とraw-test well source exclusion fail-closed guardを実装した。
- raw-test base 41列からcandidate-long 165 `copcf_*`列を再生成するschema contractを実装した。
- prefix denylistを、実生成の有無と再生成provenanceを分ける判定へ変更した。
- 実raw-test partial-source smoke、synthetic 165-column expansion、synthetic provenance test、py_compile、Ruffを通した。
- 既存正規ipynbを上書きせず、更新済みtrain Jupytext `.py`から別名`rawtest_copcf_parity.ipynb`を生成し、Jupytext testとstrict experiment validationを通した。
- 0 variant / 0 config / 0 fold / 0 booster、parent/control再学習なしのKaggle feature audit packageを作成し、既存canonical kernelのpullとmetadata確認を行った。
- canonical kernelへversion 3をpushし、0-booster corrected feature auditを開始した。
- Kaggle CPU feature audit version 3を完走し、297 parent / 295 selected / 165 regenerated `copcf_*`、test-well source overlap 0、hard check全PASSを確認した。
- audit / contract / selected schemaとtrain/raw-test sampleのSHAを必要生成物の選択取得で再検算し、全一致を確認した。
- version 3 audit PASS後のoptional train実行についてユーザー承認を得た。
- canonical kernelへversion 4をpushし、post-push pullでid_no・CPU・sources・bootstrap config SHAを確認した。
- Kaggle CPUで`feature_audit_only` version 1を実行し、297 feature contract、selected 130列、excluded 167列、distribution warning 38列、artifact SHAを確認した。
- stage、1 variant、2 config、5 folds、10 CPU boosters、parent/control再学習なしを提示し、ユーザーの明示承認を得た。
- `train_after_feature_audit` version 2を完走し、CV/fold/guard/model・imputer・OOF SHAを記録した。
- overall / 1000+ / hidden-likeはPASSしたがworst-well guardがFAILしたため、inference実装条件は未達と判定した。
- Kaggle CPU version 4を完走し、same-run 297/295/165 feature auditと10 CPU boosters完走を確認した。
- 295列版fixed Viterbi 8.502212はoverall、1000+、worst-well guardがFAILし、`adoption_supported=false`と判定した。
- OOFからfold RMSEを再構成し、診断生成物11/11、10 models、5 imputers、feature contractのSHAを検算した。
- 必要なversion 4診断生成物とOOFだけを選択保存し、inference / submissionへ進めないことを記録した。
