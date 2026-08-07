# exp350_exp345_bidirectional_gr_affine_smoother 結果

## 状態

Kaggle CPU Stage 0 version 1を完了した。technical gateはPASS、scientific gateはFAILし、decisionは`stage_0_failed_close_without_rescue`である。Stage 1、inference、submissionは行わない。

## 仮説

exp345と同じaffine forward stateを井戸全体のraw GRでfixed-interval smoothingすれば、causal候補のpooled gainを保ちながらper-well tailを抑えられる可能性がある。

## 設定

- 親: `exp345_exp209_time_varying_gr_affine_calibration_hmm`
- Route: `pf_beam`
- 単一変更: causal affine scheduleから1回のbidirectional extended RTS scheduleへの置換
- Stage 0: last-640、494,720 rows / 773 wells
- 実行量: forward 773 + smoother 773 + new HMM 773
- control HMM再実行: parent 0 / causal 0
- LightGBM config / trained fold / booster / PF / Beam / GPU: 0 / 0 / 0 / 0 / 0 / 0
- kernel: `kentookumura/exp350-bidirectional-gr-affine-smoother-train` version 1、id_no `128274195`
- runtime: `2749.356610 sec`（45.82分）

## 結果

| 比較 | baseline RMSE | candidate RMSE | 改善 |
| --- | ---: | ---: | ---: |
| masked exp209 parent | 14.501048 | 14.367548 | +0.133499 ft |
| exp345 causal | 14.331543 | 14.367548 | -0.036006 ft |

- parent比は5/5 folds改善し、hidden-like spatial / typewell-purgedも改善した。
- causal比は2/5 folds改善に留まった。fold 0 / 3 / 4は悪化し、特にfold 4は`+0.328832 ft`悪化した。
- parent比で403/773 wells改善、370/773 wells悪化した。
- parent比by-well deltaはmedian `-0.008672 ft`だが、p95 `+1.346427 ft`でFAILした。
- worst well `8995c945`はparent `14.255149`、causal `10.343780`、candidate `35.142523`で、parent比`+20.887374 ft`、causal比`+24.798744 ft`悪化した。
- boundary jump p95は`0.839762 sigma`、scale clip率は0%であり、tail failureは境界clipの単純な数値異常ではない。

## Technical gate

PASSした。

- exp345保存成果物SHA: 全一致
- saved parent / causal metric parity: 差0
- forward schedule parity最大差: `7.070433e-11`（上限`1e-10`）
- prediction: 494,720 rows / 773 wells、全finite
- HMM: 773/773
- posterior正規化最大誤差: `2.886580e-15`
- terminal state / covariance誤差: 0 / 0
- covariance最小固有値: `3.431266e-06`
- contraction最大正固有値: `2.685941e-17`
- runtime: `2749.356610 sec`（上限30,600秒）

## Scientific gate

FAILした。

- parent比改善量、5/5 folds、hidden-like 2面はPASS。
- causal比は`-0.036006 ft`、2/5 foldsでFAIL。
- parent比by-well p95 `+1.346427 ft`とworst `+20.887374 ft`がFAIL。
- GR reconstruction NLL `4.640711`はfuture GRを見たin-sample診断なのでpromotionには使わない。

## 再現性

- prediction decompressed SHA: `5b66ee9cfa76fce320a5806849e49c5f711b6dbafde617567adc211cda4de3f1`
- forward schedule decompressed SHA: `682e75d6cbed11b96e3b687f66a5c851399ae39250d5147ec3544601892a72cb`
- smoothed schedule decompressed SHA: `3c5a4268ee08e94b79b7a8fb971b2a678a45cde9edae6743e67db8cb881ac88b`
- freeze manifest SHA: `57d7a6207225281c8e4b5a517db1c286f3682b3bffda4b05036516826e471df6`
- promotion gate raw SHA: `2f003dfd46cda4efcd25c1e2555bf68a77db78273342f2425809c5a109d94616`
- deterministic anchor: いいえ。別承認のrerun parityは未確認。
- model / submission SHA: 非該当。

## 解釈

実装や数値安定性の失敗ではない。forward parity、covariance、terminal、runtimeはすべて成立したうえで、井戸末尾側のGR evidenceをprefix方向へ戻すことが一部wellのaffine stateを強く誤らせた。平均では親を改善するが、exp345 causalより悪く、tailを抑える目的に反してworstをさらに拡大した。

固定base pathに対するfull-suffix smoothingは、base path自体が局所的に誤るwellで誤calibrationを広い区間へ逆伝播する可能性がある。したがって本方式をStage 1へ進めず、Q / rcond / clip / iteration / blend / row・well gateのpost-hoc救済も行わない。

## 次

branchを閉じる。backlogからexp350を削除し、同じaffine-smoother familyの救済候補は追加しない。既存の独立0-booster候補exp340を優先する。

