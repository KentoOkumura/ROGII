# SESSION_NOTES

## 2026-07-15 実験開始

- ユーザーがselector top1候補を最終予測へ明示反映する2つ目の修正の実行を依頼した。
- exp238の仮説・最終構造を変えるため、別実験exp255を作成した。
- exp238 add-only `lgb_mean` OOFをbaseに、selector top1方向へのgated/bounded correctionを評価する。
- fixed profilesはconservative/balanced/assertiveの3個。最大移動量は4/7.5/12 ft。
- active audit 1、model config 0、fold training 0、booster 0、parent/control再学習なし。
- Kaggle CPU、internet off。submission生成・competition submitなし。
- outer-valid truthはmetric/guardだけに使い、candidate selection/gate/correctionには使わない。
- 実行前契約を再確認: active audit 1、fixed profile 3、LightGBM config 0、fold training 0、合計booster 0。既存control/親実験の再学習は含まない。
- canonical train notebookのsyntax、ruff、Jupytext round-trip、strict experiment validationが通過した。
- exp238 nested selector score 5 foldのdecompressed SHAをexp255 configにも固定し、summaryとの差異・artifactとの差異をfail-closedにした。
- Kaggle package config SHA256: `7878feb423e3c4bb7795c8c1723dea1352b005cad517e0608e5d4654aa2a54ae`。
- Kaggle metadata: private / CPU / internet off / run_on_push / kernel source 7個。初回pre-pullは新規kernelのため403だった。
- 初回pushは67文字の長いslugをKaggle APIが400で拒否し、計算は開始されなかった。slugだけを`exp255-gated-bounded-selector-readout-train`へ短縮した。
- CPU train v1 / id_no `127317283`をpushし、`RUNNING`を確認した。URL: https://www.kaggle.com/code/kentookumura/exp255-gated-bounded-selector-readout-train
- Kaggle pull-backは16/16 cellのsourceがlocal packageと一致した。cell source SHA256: `7c15de81dd240e3d1e9f3671750b6221842bee73c4431aaed922a1325c74574e`。
- ユーザー指示に従い継続監視は行わない。完了後にoutput/metrics/guardを同期する。

## 2026-07-15 Kaggle train v1完了

- status `COMPLETE`、notebook runtime 691.353秒。model config / fold training / boosterは0 / 0 / 0のまま。
- 3,783,989 rows / 773 wells / 11 candidatesは全finite。exp238 selector role=`valid` scoreは全行を一度ずつcoverし、5 foldのdecompressed SHAは全一致した。
- exp238 add-only base RMSE 7.936690、hard top1 diagnostic 8.512262（+0.575572）。hard replacementは再確認でも不採用。
- conservative 7.938384（+0.001694）、balanced 7.929965（-0.006725）、assertive 7.877990（-0.058700）。
- assertiveはnear -0.038930、1000+ -0.066824、hidden-like spatial -0.189661、typewell-purged -0.191090、3/5 folds改善。
- assertiveは604 wellsを補正し324改善 / 280悪化。106 wellsが+0.25 ft超悪化し、worst `d7ba4f9d` +3.151245のため採用guard fail。
- balancedはworst +2.167282かつ2/5 folds、conservativeはglobal +0.001694かつworst +1.163183。passing profileは0。
- `truth_used_in_gate=false`、direction violation 0、move cap全PASS。失敗点はwell-tail riskであり実装・SHA・coverage失敗ではない。
- Kaggle CLI既定page-size 20ではsummary/well-gateが次ページになったため、page-size 200とfile patternで追加取得した。selected OOF本体は同期せず、decompressed SHA `2ea22a46c46ef62a0411b170a0f9a81e85664355d5f224f27d73ffd0f929296a`をsummaryから記録した。
- metrics/by-well/gate/candidate distribution/well-gate/input manifest/plot/summaryを`artifacts/`へ同期し、全small-output SHAがnotebook summaryと一致した。
- inference、submission生成、competition submitはいずれも未実行。exp255は`completed_no_profile_passed_inference_forbidden`で終了する。
