# 設計

## アプローチ

exp287の特徴量やモデル構成を変えず、outer-train内で各wellが学習損失へ同じ総量だけ寄与するようsample weightを付ける。長いsuffixを持つwellがrow数に比例して損失を支配する状態だけを取り除き、exp287のglobal gainを保ったままwell-level tail guardを回復できるかを検証する。

これは「周辺well数」仮説の再検証ではなく、学習目的関数の集約単位をrowからwell-balancedへ変える独立仮説である。

## 実験範囲

- 対象実験: `exp334_equal_well_loss_weighting_on_exp287`
- Route: `ml_model`
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 変更する変数: outer-trainの学習行へ付けるwell均等化sample weight
- 固定する変数:
  - exp287のclean 273 + nested compact 74 + fold-safe formation 74 = 421特徴
  - exp287のouter 5-fold group split、score rows、row identity、target residual
  - exp287のLightGBM configs `[0, 1, 2]`、early stopping `250`、seed `42`
  - fold-local formation reference、selector、候補生成、feature順序、欠損処理
  - valid/OOF/scope/by-wellの非加重RMSE評価

## 比較設計

| 役割 | 成果物 | 用途 |
| --- | --- | --- |
| 主control | exp287保存済みOOF、fold metrics、by-well metrics | global精度とexp287からのtail変化を比較 |
| clean tail control | exp264保存済みOOF、by-well metrics | exp287が失敗したworst-wellと悪化well数guardを再評価 |
| new variant | exp334 well-balanced train weight | 変更する唯一の学習variant |

- exp287 OOF RMSE: `8.136708220359452`
- exp287 OOF SHA256: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA256: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 metrics SHA256: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- exp287 fold metrics SHA256: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- exp287 by-well metrics SHA256: `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- exp264 corrected Stage D OOF SHA256: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`

control SHAは実装時の入力gateで照合し、不一致なら学習前に停止する。control boosterは再学習しない。

## 重み計算

outer fold `f` のouter-train評価対象行について、総行数を `N_f`、well数を `W_f`、well `w` の行数を `n_{f,w}` とする。

`weight_{f,i} = N_f / (W_f * n_{f,w(i)})`

この式により次が成立する。

- `sum_i weight_{f,i} = N_f`
- `mean_i weight_{f,i} = 1`
- 各wellの `sum weight = N_f / W_f`

重みはouter-trainのgroup、fold、score-row identityだけから決める。validation Datasetへはweightを渡さないため、early stoppingはexp287と同じ非加重RMSEを使う。これにより「train lossのwell寄与」以外の評価契約を変えない。

## 実行単位

- variant: 1 (`equal_well_total_train_weight`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計予定: 15 GPU boosters
- exp287/exp264 control再学習: 0 boosters
- 現在の承認状態: 実装1、Kaggle package/push 0、学習0、推論0、提出0

実装は2026-07-21に承認済み。train pushの前に、15 boostersのGPUコストとcontrol再学習0を再提示し、ユーザーの明示承認を得る。

## 実装構成

- 親exp287のformation 74列は再生成せず、SHA固定された10 fold-role cacheを再利用する。
- clean 273列とnested compact 74列はexp287と同じsource/config/Stage Cから再構成し、保存model manifestのfeature group/order/SHAと照合する。
- 全foldのweight契約をfit前にprecomputeし、train時の再計算SHAと一致させる。
- 正規notebook scaffoldは上書きせず、compact self-contained候補を別名で実装する。Kaggle package化前に採用判断を行う。

## Preflight契約

実装時は学習開始前に次をfail-closedで確認する。

- exp287のrow identity、fold assignment、421 feature schema/order、target、3 configが一致する。
- exp287/exp264の保存済みcontrol SHAが一致する。
- 各outer foldのtrain weightが全件finiteかつ正値で、平均1である。
- 各wellの総重みが `N_f / W_f` と許容誤差 `1e-10`以内で一致する。
- 同一wellの全行が同一weightである。
- weight作成経路がtarget、予測、error、outer-valid rowsを参照していない。
- valid Datasetにweightが設定されていない。
- 実行量が1 variant × 3 configs × 5 folds = 15 boosters、control再学習0である。

## Promotion gate

全項目をAND条件とする。

1. exp287比の非加重pooled OOF delta RMSE `<= +0.02 ft`。
2. 5 folds中4 folds以上でexp287以下。
3. near / mid / 1000+ / hidden-likeの各scopeがexp287比 `<= +0.02 ft`。
4. by-well delta RMSE p95がexp287比 `<= 0.00 ft`。
5. exp264比worst-well delta RMSE `<= +0.25 ft`。
6. exp264比 `+1/+3/+5 ft`悪化well数が `135/39/14` を超えない。

exp287のworst-well `+8.228410 ft` と悪化well数 `140/40/19` を単にexp287比で非悪化とするだけでは不十分なので、tailの最終gateはclean controlのexp264に対して判定する。gate不通過ならinference/submit候補にしない。

## 本実験に含めないもの

- inner-CV hard-well再重み付け
- exp287同一OOF errorを使う重み、threshold、grid search
- Huber/custom objective、loss blend、weight clipping/power調整
- feature追加/削除、formation reference変更、selector変更
- LightGBM config、seed、fold、target、early stopping変更
- control再学習、inference、submission、guard緩和

inner-CV hard-well重みは、本実験がtail方向を改善したもののclean guardへ届かなかった場合にだけ、別実験として再検討する。

## 再現性設計

- seed policy: exp287と同じfixed global seed `42`
- stochastic処理の有無: LightGBM GPU学習あり。新しい乱数処理は追加しない。
- PF/Beam / likelihood-PF / seed baggingの有無: 新規利用なし。保存済みcompact meta featureは固定入力としてのみ利用する。
- 並列処理と乱数の関係: exp287のthreadsとGPU設定を固定し、weight計算は決定的なgroup countとrow orderだけを使う。
- CPU/GPU runtimeとdeterministic flags: Nvidia Tesla T4、internet off、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8を維持する。GPU runをbitwise deterministic anchorとはみなさない。
- train cache / test feature regenerationのSHA記録方針: exp287入力、fold assignment、feature schema/order/content、formation fold manifest、raw schema auditのSHAを記録する。test生成は本実験範囲外。
- model manifest / prediction / submission SHA記録方針: 承認後に学習した場合はmodel manifestとOOF prediction SHAを記録する。inference/submissionは未承認なので生成しない。
- Kaggle package bootstrap確認方針: push前にembedded config、notebook、support archive、metadataのSHAと、T4/internet off/run approval flagsを照合する。

## リスク

- リークリスク: weightがtarget/errorに依存すると直接的なOOF overfitになる。group countとfold identity以外を参照禁止にする。
- CV/LB不一致リスク: public testが3 wellsしかなく、well-balanced CV改善がLBへ移る保証はない。まずtrain-side tail guardだけで判定し、推論は別承認にする。
- 最適化リスク: 長いwellの有効サンプルを弱めるためpooled RMSEが悪化し得る。pooled/scope budgetとfold gateを併用する。
- ランタイム/メモリリスク: 予定15 GPU boosters。重みベクトル1本/rowの追加メモリが必要だが、feature matrixはexp287と同じ421列に固定する。
- 再現性リスク: GPU LightGBMは完全なbitwise一致を保証しない。設定・入力・weight統計・SHAを記録し、必要時のみ別承認でrerunする。
