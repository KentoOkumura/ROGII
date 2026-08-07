# 設計

## 仮説

exp413の最終TVT予測に残るwell内の高周波揺れだけを、公開実装と同じ固定SG filterで除く。
PF/Beam component、ML feature、selector、modelを変えずにfold横断で改善し、scope・well-tail・
prediction startを壊さない場合だけ、exp413推論の最後へ移植できる後処理と判断する。

## 公開sourceと転用境界

- source: `docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260611/jaemin3404__rogii-sp45-fleongg-jy-dynamic-v1/rogii-sp45-fleongg-jy-dynamic-v1.py`
- source selector: `n_eval=4840`、`z_span=(136.73, 185.5133)`と6 binの固定variant map。
- source postprocess: `tau=85` warmup、model 60% / LikPF 40%、well別SG `61 / 3`。
- exp497はwell-shape selector、warmup、SGを含むstrict public-core全体を再現済みだが、exp413とのblendはpooled gain`0.010315 ft`、3/5 folds、hidden-likeとwell-tail悪化でgateをFAILした。

exp508はこの複合recipeを再実行しない。公開sourceが約`0.01 ft`の独立効果と記述する
SGだけをone-factor selectable primaryにし、warmupを非選択診断へ隔離する。

## 実験範囲

- 対象実験: `exp508_exp413_public_trajectory_postprocess_audit`
- Route: `ml_model`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 参照実験: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`
- 変更する変数: 保存exp413最終TVTへの固定SG `window_length=61 / polyorder=3`だけ
- 固定する変数: exp413 prediction、outer folds、well内row order、metric、scope、SG実装、gate
- 範囲外: well router、public fixed threshold/map、direct LikPF blend、model/PF/HMM/Beam再実行、inference、submission

## 候補契約

### control

```text
p_control = p_exp413
```

保存済みStage D OOFをそのまま使う。control再学習は0。

### selectable primary

```text
p_primary = savgol_by_well(p_exp413, window_length=61, polyorder=3)
```

公開sourceの`make_prediction`と同じ規則に固定する。

1. source OOFのglobal row orderを保持し、`well`の出現順とwell内の保存行順を変更しない。
2. wellのprediction vectorを`float64`で取り出す。
3. `wl = min(61, n_rows)`、偶数なら`wl -= 1`。
4. `wl >= 5`なら`scipy.signal.savgol_filter(v, wl, 3)`をSciPy既定modeで適用する。
5. `wl < 5`なら変更しない。
6. filter後のreanchor、clip、projection、residual補正をしない。

### report-only controls

```text
g_i = 1 - exp(-max(md_since_i, 0) / 85)
p_warmup_i = last_known_tvt_i + g_i * (p_exp413_i - last_known_tvt_i)
p_warmup_sg = savgol_by_well(p_warmup, 61, 3)
```

`tau85_warmup_final_delta`と`tau85_warmup_then_sg61_p3`は、primary predictionとprimary
decisionをfreezeした後だけscoreする。両者は`selectable=false`であり、primary FAIL時の救済、
window/tau選択、推論候補化、backlog自動昇格に使わない。

## 入力契約

### exp413 Stage D

- source kernel: `kentookumura/exp413-scale5-likpf-downstream-train` version 2
- OOF file: `stage_d_oof_predictions.parquet`
- prediction column: `scale5_x1p0_full_replacement__lgb_mean__pred_tvt`
- expected rows / wells / folds: `3,783,989 / 773 / 5`
- expected CV: `7.884802794404715`
- expected OOF SHA256: `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- fold manifest SHA256: `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
- fold metrics SHA256: `82e70b6674f218f2892d6e5f70e327dfcbbdaf0fa5e431c4e07231009e9e2d8f`
- scope metrics SHA256: `c89add97cd4cae628b79774615a717e4cfbffe7b65a4a68c58b2c2e2737948ed`
- hidden-like metrics SHA256: `eafa3546e4ea5c0d180d380f7fe2c39b5cac970ea4c8097b68b077017da1f1b8`
- by-well SHA256: `e82c6908ed2caa9b3e5c1664bc66a3226b3bc6d9284f4863bd4fa941ae32d080`
- hidden-like assignment SHA256: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`

実装時は実ファイルのschemaをfail-fast確認し、`well / row_idx`またはsourceが持つ一意な
logical key、fold、score-row role、`md_since`、`last_known_tvt`をmanifestへ固定する。
missing / extra / duplicate key、NaN / Inf、row-order不一致は0件でなければ停止する。

## phase separation / leakage guard

1. exp413 OOFと固定assignmentを読み、raw file SHA、logical key SHA、global row-order SHA、well内row-order SHA、fold SHAを検証する。
2. truth/error/scopeを参照せず、control、SG primary、warmup、warmup+SGの3 candidate predictionsを生成する。
3. 3 candidate predictionとparameter contractをSHA freezeする。
4. freeze後だけsuffix truthと固定scope assignmentを接続する。
5. controlとprimaryのpooled / fold / scope / by-well / prediction-start metricsを計算し、primary decisionをfreezeする。
6. primary decision後だけreport-only 2本のscoreを表示する。

candidateにfitやfold別parameterはない。truth/errorを使ったSG/tau選択、well除外、row除外、
candidate選択はすべて禁止する。

## 評価と判定

Primary:

- pooled suffix-row unweighted RMSE / gain
- fold 0--4 RMSE / delta
- MD 0--250、250--1000、1000+ RMSE / delta
- hidden-like spatial / typewell-purged RMSE / delta
- by-well delta RMSE median / p90 / p95 / p99 / worst
- +0.25 / +1 / +3 / +5 ft悪化well数
- first score-row `abs(primary - control)` p50 / p90 / p95 / p99 / max
- `abs(primary-control)`とtrajectory second-difference normのpooled / well summary

Primary all-AND gate:

- technical / leakage / SHA all PASS
- pooled gain `>=0.01 ft`
- nonworse folds `>=4/5`
- 固定5 scopesのdelta `<=+0.02 ft`
- by-well p95 / worst delta `<=+0.25 ft`
- first score-row correction p95 `<=0.50 ft`かつmax `<=2.00 ft`

PASS時もinference実装は自動承認しない。FAIL時は
`FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`で終端閉鎖する。

## well-level routingの後続条件

exp508ではrouterを実装・fit・評価しない。条件付き後続を作るのは次の全条件を満たす場合だけとする。

1. exp508 primaryが上記all-AND gateをPASSする。
2. raw exp413とSGのtarget-free disagreementにwell間variationがあり、固定単一SGでは捨てる相補性を説明できる。
3. 公開の`n_eval / z_span`固定thresholdを再利用せず、raw-test生成可能なwell featureだけでouter-train fit / outer-valid applyを設計できる。
4. 独立したsteering、experiment number、実装・実行承認を得る。

既存のexp300 raw deployable well feature AUC `0.495675`とexp499 target-free well selector AUC
`0.521151` / router RMSE `8.514311`対always-exp490 `8.480155`は、hard well routingの
negative evidenceとして引き継ぐ。

## 将来の実装・実行量契約

| 項目 | 実行量 |
| --- | ---: |
| selectable primary | 1 |
| report-only controls | 2 |
| outer folds | 保存5 foldsのreportのみ |
| learned model / booster | 0 / 0 |
| LightGBM config | 0 |
| HMM / PF / Beam run | 0 / 0 / 0 |
| exp413 control retraining | 0 |
| GPU | 0 |

## 再現性設計

- seed policy: no RNG。固定parameter、固定row order、float64処理のみ。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 再実行0。保存exp413 OOFのSHAだけを使用する。
- 並列処理と乱数: RNGなし。well処理順を保存順に固定し、並列reduceを使わない。
- runtime: 将来のStage AはKaggle private CPU / internet off。GPUは使わない。
- environment: Python、NumPy、SciPy、pandas、pyarrow versionをmanifestへ保存する。
- record: source file SHA、logical key / row-order / fold SHA、contract SHA、3 prediction SHA、metrics / gate SHA。
- deterministic anchor: Stage Aの同一環境rerunでprediction content SHAが一致するまではfalse。
- inferenceへ進む場合: hidden-dynamic test inventory、current-test prediction content SHA、submission SHA、kernel versionを追加する。
- Kaggle package: 実装承認後にmetadataとbootstrap ZIP内config/input contractの一致をpush前に検証する。

## リスク

- リークリスク: same-OOFでwindow/polyorder/tauを選ぶこと、truth/errorでwell/rowを除外すること。
- CV/LB不一致: exp413はCV 7.884803 / Public LB 7.201であり、小さいCV gainだけで提出しない。
- boundaryリスク: SGが各wellのprediction startを移動する。first-row continuityとnear scopeを必須gateにする。
- tailリスク: pooled改善が少数well悪化を隠す。by-well p95/worstを自動停止gateにする。
- 二重物理リスク: full public 40% LikPF blendはexp413のscale5置換と重複するため除外する。
- routerリスク: 公開thresholdはpublic test shapeに固定され、既存target-free well selectorも弱い。exp508から分離する。
- 再現性リスク: SciPy version、row order、短well処理、float dtypeで出力が変わり得る。contractとversionとSHAでfail closedする。
