# exp387_prefix_gr_rgt_scenario_posterior 結果

## 仮説

exp386 の複数物理 scenario が真の経路を十分に覆うなら、outer-train 由来 GR template に対する target GR 尤度で posterior を更新し、graph-cost prior mean より真の scenario に近づけられる。

## 設定

- 親: exp386
- route: `pf_beam`
- 検証: exp386 と同じ outer 5-fold、`well_id` group
- 尤度: robust GR level + first difference、Student-t(df=4)
- decoder: compatibility-constrained exact forward-backward
- 出力: posterior-weighted scenario TVT
- hard top-1: 禁止

## 結果

exp386 version 1はscenario-bank coverage `0.0`、finite-path coverage `0.0`、
cycle residual p95 `2.363303 > 0.10`でStage 0 FAIL_CLOSEとなった。
親scenario bankが空のため、exp387は未実装・未実行で閉鎖した。

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 合格条件

- Stage 0: parent SHA 一致、posterior 正規化、prefix gain 0.25 ft以上、real-vs-circular MRR gain 0.02以上、leakage 0
- Stage 1: pooled RMSE 7.20 ft以下、exp226 比2.0 ft以上改善、5 fold 中4 fold以上で改善
- 長距離 / hidden-like scope の改善と短距離 non-regression を同時に満たす
- Stage 2 promotion safety は別途承認制

## 再現性

- deterministic anchor: 未確立
- seed policy: RNG なし、不変キーによる安定順序
- parent manifest SHA: 未生成
- artifact / prediction SHA: 実装時に記録予定
- rerun: 初回成功後に content SHA 一致を要求

## 解釈

これはexp386の固定済み複数解を評価する設計であり、候補を作り直す権限を持たない。
親bankが生成されなかったため、GR尤度の識別力を評価する前提そのものが不成立だった。

## 次

実装、Stage 0/1、inference、submissionなしで閉じる。別generatorの独立したPASSを理由に
再検討する場合も、自動再開せず新しい事前設計と承認を必要とする。
