# タスクリスト

## 進行中

- なし。

## 閉鎖

- 固定判定`FAIL_STOP_NO_STAGE4`によりStage 3、Stage 4、inference、submissionを閉じた。

## 完了

- exp293 support PASSと固定Stage 2 branchを確認した。
- `exp297_prefix_calibrated_latent_registration_gr_evidence`のsteering/scaffoldを作成した。
- candidate、registration、calibration、reliability、shuffle、truth freeze、PASS条件を設計へ転記した。
- `docs/06_reproducibility.md`のstable seed、SHA、bootstrap方針を固定した。
- compact self-contained trainとfail-closed inferenceを実装した。
- synthetic contract testsを10件追加し、全PASSした。
- Jupytext変換/round-trip、構文、undefined-name、実験静的validationを完了した。
- 実装段階でREADME/SESSION_NOTES/result/metricsと横断summary/backlogを「実装完了・未実行」へ更新した。
- compact trainをcanonical trainへ採用し、同一Kaggle private CPU kernel version 2で完走した。
- 3,783,989 rows / 773 wells / 105,818 block-controlとtruth freeze前access 0を確認した。
- target-free/readout 12ファイルのSHAを取得outputで再計算し全一致した。
- H256 recovery `-0.116476`、realがshuffleよりpooled/5 foldsで悪いことを固定条件で判定した。
- `FAIL_STOP_NO_STAGE4`をconfig/metrics/README/SESSION_NOTES/result/横断記録へ反映した。

## 次

exp297 branchでは追加実施しない。同一posteriorの救済gridを行わず、独立したexp298/exp295だけを別契約・別承認で扱う。
