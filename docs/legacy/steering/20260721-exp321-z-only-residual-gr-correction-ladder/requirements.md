# 要件

## 依頼

`TVT` の変化を、最後の既知 `TVT_input` と `Z` の変化だけから作る固定係数 `-1` の経路で予測し、残差構造の診断、GR shift 尤度の識別力監査、exp226 window GR 補正の順に同一実験内で評価する。初回依頼では backlog、steering、実験 scaffold と設計までを確定した。2026-07-21 の追加依頼「exp321を実装してください」を Stage A/B の実装開始承認として扱い、compact self-contained train/inference Notebookと専用testまで実装する。Kaggle package/push/runとStage C実装、inference、submissionは今回行わない。

## 仮説

各wellの最後の既知行を `s` とすると、

```text
tvt_z(t) = TVT_input(s) - (Z(t) - Z(s)),  t > s
```

は `U=TVT+Z` を一定とした最小物理経路であり、未知suffixの局所形状の大半を説明する。残る低周波offsetがType Well GRとの対応で識別可能なら、exp226と同一のwindow GR補正を加えた
`tvt_z_gr=tvt_z+gr_delta` は `tvt_z` を安全に改善する。

## 制約

- Route: `pf_beam`。LightGBM、CatBoost、XGBoost、PF、Beamは使わない。
- Z係数は厳密に`-1`。known prefixからの傾き、切片、rate、`b`はfitしない。
- anchorはraw行順で最後の有限`TVT_input`行の`TVT_input`と`Z`だけを使う。
- XY、donor、ANCC、formation、exp226 kappa、U projectionは使わない。
- GRはStage B/Cだけで使い、Z-only pathには混ぜない。
- Stage Bはexp280の13 shift、512行block、exp209 Gaussian raw-GR/typewell emission、missing処理、stable shuffleを変更しない。base pathだけを`tvt_geop`から`tvt_z`へ置換する。
- Stage Cはexp226のknown-prefix affine calibration、500行window、125行stride、0.5 ft grid、corr/MSE/level likelihood、forward-backward、posterior-SD shrink、`[-4,+4] ft` capを変更しない。base pathとrelpathだけをZ-onlyへ置換し、U projectionを無効にする。
- Stage A/Bのtarget-free path、block、shift scoreをcontent SHAで凍結するまでsuffix真値を読まない。
- Stage CはStage A/Bの全gate PASS後の別runとし、GR補正predictionを凍結するまでsuffix真値を読まない。
- Stage A/B/Cのthreshold、scope、tie order、fallback、禁止事項は結果を見て変更しない。
- Stage A/B実装開始は2026-07-21に承認済み。Kaggle CPU Run AB、Stage C実装、Stage C runはそれぞれ別途ユーザー確認を要する。

## 受け入れ基準

- Stage A/B/Cの入力、数式、実行順、gate、停止条件が`config.yaml`と一致する。
- Stage Aは3,783,989 suffix rows / 773 wells、exp226固定5 foldsを期待し、row/well/fold identityをhard guardする。
- Stage AはH128/H256/H512 blockでoffset/affine quotientを読み、H256/H512の局所形状、低次元残差、cap 4 ftの補正headroomを事前固定gateで判定する。
- Stage Bはtop1/top3/MRR/signがstable shuffleを5/5 foldsで上回り、pooledでexp280の同指標を4/4上回る場合だけPASSとする。
- Stage Cは`tvt_z`比0.05 ft以上、4/5 folds、near/1000+/hidden-like 2面非悪化、by-well p95/worst悪化0.25 ft以下をすべて満たす場合だけ科学的PASSとする。
- Stage Cを将来のroute候補とするには、上記に加えてexp226保存OOF RMSE 9.427110以下を必須とする。
- 後続案4/5は`reserved_followups.md`とsteering designに同じtrigger・範囲外事項で記録し、exp321には実装しない。
- Stage A/BのNotebook、実験ロジック、testを実装する。実データ生成物はKaggle CPU Run AB承認後まで作成しない。Stage Cは未実装のまま保持する。
