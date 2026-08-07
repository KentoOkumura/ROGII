# exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm 結果

## 状態

Stage 0 Kaggle CPU version 1（id_no `127890033`）を完了し、performance hard gate FAILでbranchを閉じた。Stage 1、inference、submissionは未実行。

## 仮説

known prefixのraw missing self-GR donor cellだけを、observed known GRへ頑健 affine校正したType Well復元GRで補完すれば、観測値やanchor eligibilityを変えずにexp223 self-GR HMMを改善できる可能性がある。

## 設定

- 親: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- Route: `ensemble`
- 検証: Stage 0 擬似欠損信号監査 -> 全PASS時だけStage 1固定HMM比較
- Stage 0実行variant: 1
- Stage 1 HMM variant / well-runs: `0 / 0`実行（設計上の候補 `1 / 773` はStage 0 FAILで閉鎖）
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- シード方針: stable SHA256、global RNGなし

## 実装差分

- 親exp223のHMM本体は再実装・再実行せず、known-prefix donorのStage 0信号監査だけを別名compact notebookへ実装した。
- `TVT`を読み込まないmask-first / truth-late-join、Type Well範囲内挿、deterministic Huber IRLS、observed/raw-mask parity、全hard gateとSHA manifestを追加した。
- Stage 1 HMM、inference、submissionは実装していない。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 | `FAIL` / branch closed |
| control RMSE | 8.138530741 |
| Type Well gap-fill RMSE | 12.842185844 |
| RMSE delta / relative improvement | `+4.703655102 ft` / `-57.7949%` |
| control / variant MAE | 5.151894538 / 8.963970941 |
| RMSE改善 reporting folds | 0 / 5 |
| by-well RMSE p95 delta | `+15.494310655 ft` |
| well勝敗 | 157改善 / 610悪化 / 6同値 |
| finite coverage | 1.0 |
| Stage 1 | 未実行・閉鎖 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 判定

performance gate 5件をすべてFAILした。Type Well gap-fillはpooled RMSE、全5 folds、by-well p95で線形補間より大幅に悪く、Stage 1へ進む根拠がない。自然欠損run長は全foldでq25/q50/q90 `1/1/3`行となり、minimum length 4のZNCCは未定義でZNCC gateもFAILした。RMSEとp95だけでも棄却は確定するため、block長やZNCC定義を変更する救済は行わない。

technical hard gate 6件はPASSした。773 wells / 2,319 blocks / 3,865 rowsを評価し、observed GRとraw missing maskのexact parity、pseudo-mask fit overlap 0、target-side Type Well fill 0、finite coverage 1.0、5 reporting foldsを確認した。truth-free manifestをprediction前にfreezeし、suffix `TVT`は読んでいない。

## 再現性

- deterministic anchor: false
- kernel: `kentookumura/exp294-typewell-gapfill-selfgr-stage0-train` version 1 / id_no `127890033` / `COMPLETE`
- runtime: 160.32秒、CPU、GPUなし
- input manifest SHA: `62cb5af49b74a352a3a75448166ab5165d520265d229fbb05471eb73e6f0c9bf`
- feature schema SHA: `b48ac632409f8932996abf9c50ee3c72b26c9e470e78f1c79f2548b3fe413b0c`
- pseudo-mask decompressed SHA: `92fe1685ddc2e931f6ad334ebb0965a18b8a8c3984477a4e42603b8257e49e33`
- held-out prediction decompressed SHA: `fe147623888c880ab51e8326099e5848e3069488b0d3ca7a1d3ab9b668e7f451`
- artifact manifest SHA: `b003790cbbce60d9f12554c2b11cd46476387369113ff1b07e3044a740f3bca3`
- 全9 manifest entry: byte数・raw SHA・gzip展開後SHAを取得ファイルで一致確認
- model SHA / manifest SHA: 学習modelなし
- submission SHA: submissionなし
- rerun result: 未実行。再実行は契約上不要

## 静的検証

- Jupytext `--test`: PASS
- Python syntax / Ruff: PASS
- `make validate-exp`: PASS
- `make validate-template`: PASS
- 専用 tests: 11 PASS
- repository tests: 298 PASS

## 次のアクション

このbranchは完了。Stage 1、affine/window/alpha/threshold救済grid、inference、submissionへ進めない。本結果だけを根拠とする新しい救済backlogも追加しない。
