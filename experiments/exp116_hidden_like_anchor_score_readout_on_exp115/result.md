# exp116_hidden_like_anchor_score_readout_on_exp115 結果

## 仮説

通常 OOF / train-side 全体では強い anchor でも、公式 PPT verification map に寄せた exp115 hidden-like holdout 上では順位や worst-well profile が変わる可能性がある。exp115 split を固定 stress readout として使うことで、次実験の合否ゲートに使える追加診断を作る。

## 設定

- 親: `exp115_hidden_like_spatial_holdout_from_ppt`
- 検証: `verification_like_spatial` / `verification_like_typewell_purged`
- メトリック: TVT RMSE、MAE、bucket RMSE、by-well RMSE、worst-well delta
- シード: 42
- 新規学習: なし

## 結果

| split | best | RMSE | rows | wells |
| --- | --- | ---: | ---: | ---: |
| verification_like_spatial | exp073 lgb2 | 10.765221 | 972463 | 200 |
| verification_like_spatial | exp098 lgb2 | 10.795376 | 972463 | 200 |
| verification_like_spatial | exp073 lgb1 | 10.802345 | 972463 | 200 |
| verification_like_spatial | exp092 lgb1 | 10.832060 | 972463 | 200 |
| verification_like_typewell_purged | exp073 lgb2 | 10.725383 | 976449 | 200 |
| verification_like_typewell_purged | exp098 lgb2 | 10.750165 | 976449 | 200 |
| verification_like_typewell_purged | exp073 lgb1 | 10.756291 | 976449 | 200 |
| verification_like_typewell_purged | exp092 lgb1 | 10.778500 | 976449 | 200 |

`exp073` / `exp098` は row-level prediction から採点した。`exp092` は手元の row-level prediction が空だったため、by-well metrics から weighted RMSE として集計した。したがって exp092 は row-level bucket や path continuity まではこの run では比較できない。

## 再現性

- deterministic anchor: false。診断 readout。
- seed policy: no new RNG。
- kernel version: `kentookumura/exp116-hidden-like-anchor-readout-train` v2 COMPLETE。
- Kaggle output: `experiments/exp116_hidden_like_anchor_score_readout_on_exp115/kaggle/output/train_v2/`
- feature content SHA: `metrics.json` の `readout.artifact_sha256` に保存。
- model SHA / manifest SHA: 新規モデルなし。
- prediction SHA: 新規 prediction なし。upstream input SHA は source inventory に保存。
- submission SHA: なし。
- rerun result: `--max-sources 1` smoke と full local readout が完了。

## 解釈

exp115 hidden-like stress holdout では、row-level で比較できる範囲では exp073 lgb2 が exp098 lgb2 と exp073 lgb1 を僅差で上回った。exp092 lgb1 は by-well fallback でも exp073 / exp098 の上位 row-level 候補より少し悪い。ただし exp092 は by-well 集計入力なので、row-level prediction を正式に取得するまでは exp092 の distance bucket や worst-well row profile を同じ粒度では評価できない。

この結果は exact hidden split や LB 代替ではない。提出判断の主根拠ではなく、後続実験が exp073/exp098/exp092 に対して hidden-like subset で大きく崩れないかを見る補助ゲートとして使う。

## 次

exp116 自体は診断実装として完了。次に exp092 系の feature 追加実験を行う場合は、通常 OOF に加えてこの exp115 readout の `verification_like_spatial` / `verification_like_typewell_purged` で悪化しないことを確認する。
