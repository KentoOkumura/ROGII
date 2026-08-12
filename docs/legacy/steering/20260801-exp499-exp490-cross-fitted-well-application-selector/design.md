# 設計

## アプローチ

保存済みexp490 full OOFから、候補同士の差・exp226/geometryとの不一致・posterior状態を
well単位に集約し、exp498でfreeze済みのvisible-prefix物理特徴と結合する。正解を読む前に
32特徴とSHAをfreezeする。その後だけ、保存済みby-well metricsから
`exp357_rmse^2 - exp490_rmse^2`を学習targetとして付与する。

outer 5-foldそれぞれについて、残り4 folds内で次の3 policyを比較する。

1. `always_exp490`: 学習なしの基準。
2. `weighted_ridge`: standardized target-free特徴からsigned MSE benefitを予測。
3. `weighted_hist_gradient_boosting`: 同じ特徴・targetを使う浅い非線形モデル。

学習する2モデルはinner 4-fold OOFでpolicy RMSEを計算し、`always_exp490`を含む最良policyを
選ぶ。同点0.01 ft以内なら`always_exp490`、Ridge、HGBの順に単純なpolicyを優先する。
選ばれたモデルだけをouter-train全体で再fitし、予測benefitが正のheld-out wellへexp490を
適用する。固定thresholdは0であり、outer-validに合わせて動かさない。

## 実験範囲

- 対象実験: `exp499_exp490_cross_fitted_well_application_selector`
- Route: `ensemble`
- 親実験: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 比較対象: 保存済み`exp357_exp226_huber_emission_independent_audit`
- 変更する変数: target-free well-level selectorの有無のみ。
- 固定する変数: exp490/exp357/exp226予測、outer fold、候補生成、全HMM/PF設定。
- 実行量: 1 selector variant、2学習モデル設定、outer 5 × inner 4 = 40 inner fits、
  最大5 outer refits、最大45 CPU model fits。LightGBM/PF/Beam/GPU/control再学習は0。

## 特徴

- exp498 9特徴: rows、visible prefix rows、suffix horizon、K16 span、prefix GR sigma、
  GR information ratio、geometry disagreement、early offset、state uncertainty。
- suffix全体16特徴: exp357/exp490/exp226/geometry間の絶対差とposterior delta/stdの
  mean、std、q90。
- suffix先頭128行4特徴: parent-exp226差、candidate-parent差、delta、posterior stdのmean。
- roughness 3特徴: exp357、exp490、exp226の隣接予測差のmean。
- fold、truth、error、truth由来roleは特徴にしない。

## 再現性設計

- seed policy: 固定seed 42。sklearnモデルは`random_state=42`、single-thread相当で実行する。
- stochastic処理: HGBの内部学習のみ。入力順をwellでstable sortし、固定seedを使う。
- PF/Beam / likelihood-PF / seed bagging: なし。保存済み予測のみ読む。
- 並列処理と乱数: joblib並列なし。global RNGやthread schedulingに依存させない。
- runtime: Kaggle CPU。GPUは使わない。
- SHA: exp490 gzip raw/decompressed SHA、by-well SHA、exp498 feature SHA、
  exp499 feature schema/content SHA、selector OOF score SHA、model manifest SHAを記録する。
- submission: 作らないためsubmission SHAはnot-applicable。
- bootstrap: prepare後、canonical notebook、config、metadata、bootstrap内support filesを確認する。

## リスク

- リークリスク: candidate/parent自体はgroup-safe OOFだが、foldまたは誤差をfeatureへ混ぜると
  selector leakageになる。phase分離と列allowlistで防ぐ。
- 選択バイアス: exploratoryに強かった単一特徴へthresholdを後付けしない。意味ベースの固定32特徴、
  nested model selection、固定0 thresholdを使う。
- CV/LB不一致: 773 wellsとfold0の分布差が大きく、train OOFで安全でもtest 3 wellsへ一般化しない。
  gate通過はsubmission承認ではない。
- tail risk: 平均RMSEが改善しても少数wellの大悪化を隠せる。p95/worstを必須gateにする。
- runtime/メモリ: gzip 3.78M行を必要列だけ読み、well単位集約後に解放する。
- 再現性: upstream notebook outputのversion変化をSHA不一致でfail-closedする。

