# 設計

## アプローチ

### Stage A: 保存 OOF による根本原因の確定

親 corrected exp264 と exp407 の `candidate_score_oof.parquet` を key 順に
stream join し、candidate × fold 平均 score 差を求める。

- `global_shift_only = parent_score + mean(exp407_score - parent_score)`
- `row_local_only = exp407_score - mean(exp407_score - parent_score)`

両 counterfactual の hard-selection RMSE、fold 非悪化数、switch、親 margin bucket
別 delta SSE を保存する。また exp407 の fold-safe RMSE weight table と、
score 差の標準偏差、MAE、logloss、Brier 差を candidate × fold で結合する。

この分解で、候補定数の calibration shift と、行ごとの interaction / ranking
変化を区別する。exp409 で単一 transition に原因を集約できなかったこととも整合し、
原因を「共有損失の task importance 変更による分散した row-local score surface
drift」として判定する。

### Stage B: fold-safe candidate RMSE additive offset

各 outer fold の exact sampled fit partition だけで候補 `c` の RMSE `b_fc` を求める。
feature と sample weight は親と同一に保ち、次の residual だけを学習する。

```text
r_i = actual_abs_error_i - b_f,c(i)
residual_hat_i = LightGBM_L1(x_i)
pred_abs_error_i = max(0, residual_hat_i + b_f,c(i))
```

元の L1 loss は
`|residual_hat - (actual_abs_error - b)| = |residual_hat + b - actual_abs_error|`
なので、候補ごとの task importance を変えない。RMSE は「候補の大域的な期待誤差」
として先に与え、shared tree は行ローカルな偏差だけを学習する。

RMSE で binary `p_within10` の重要度を変えることは目的不整合のため行わない。
Stage B の評価と hard selection は新しい `pred_abs_error` だけで行う。

## 実験範囲

- 対象実験:
  `exp414_fold_safe_candidate_rmse_offset_selector_on_exp264`
- Route: `ml_model`
- 親実験:
  `exp264_exp263_candidate_confidence_dual_selector` corrected Stage B v5
- 原因比較:
  `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`
- 変更する変数:
  `pred_abs_error` の target / prediction parameterization を additive
  candidate-RMSE offset に変更する。
- 固定する変数:
  candidate values / order / legal domains、outer folds、sample row IDs、
  feature schema 88 列、LightGBM hyperparameters / early stopping、
  unweighted validation / metrics、hard-selection tie-break。
- 学習予算:
  variant 1 × objective config 1 × outer folds 5 = CPU booster 5。
  親/control 0、classifier 0、GPU 0。
- 対象外:
  Stage C、current-test inference、submission、LB、parameter grid、
  exp407 の rescue。

## 再現性設計

- seed policy:
  親の seed 42 と deterministic sample index を継承する。LightGBM の
  random / feature_fraction / bagging / data_random seed も 42 に固定する。
- stochastic 処理の有無:
  LightGBM subsample / colsample だけ。局所 RNG 以外は使わない。
- PF/Beam / likelihood-PF / seed bagging:
  すべて保存済み candidate cache の load-only。生成処理は 0。
- 並列処理と乱数の関係:
  fold を逐次学習し、CPU deterministic / force_col_wise を有効にする。
  candidate RMSE と sample row ID は sort 済み配列から決定論的に計算する。
- CPU/GPU runtime:
  Kaggle private CPU、internet off、GPU false。`deterministic=true`、
  `force_col_wise=true`、`n_jobs` 固定。
- train cache:
  exp263 Stage 0 manifest / catalog SHA、parent feature schema logical SHA、
  candidate contract SHA を起動時に照合する。
- offset:
  fold、candidate、fit row count、fit row-ID content SHA、RMSE を CSV と JSON に
  保存し、logical SHA を記録する。
- model / OOF:
  5 model SHA、model manifest SHA、candidate-score OOF SHA、gate artifact SHA、
  実読込 parent / exp407 OOF SHA を記録する。
- prediction / submission:
  current-test prediction と submission は生成しないため対象外。
- Kaggle package bootstrap:
  push 前に生成 Notebook 内の bootstrap ZIP から config を再展開し、
  experiment name、input sources、CPU / internet、Stage B だけが一致することを
  検証する。

## リスク

- リークリスク:
  global OOF candidate RMSE を offset にすると valid truth が混入する。
  各 model の exact sampled fit rows だけで offset を算出し、fit / valid well
  overlap 0 と row-ID SHA を gate にする。
- 数学的取り違え:
  target を RMSE で除算すると元単位へ戻しても inverse-RMSE weighting と同値になる。
  additive residual 化だけを許可し、sample weight 全 1 を監査する。
- 目的不整合:
  候補 RMSE を binary objective に使うと logloss / Brier を悪化させ得る。
  classifier は学習せず treatment 外にする。
- CV/LB 不一致:
  Stage B OOF だけで確立を判定し、current test への適用可能性や LB 改善は主張しない。
  Stage C / inference は別実験判断とする。
- ランタイム/メモリ:
  candidate-long 88 列を fold ごとに作るため CPU / RAM 負荷が高い。
  親と同じ chunk / deterministic sampling を使い、大規模 OOF は必要な場合のみ取得する。
- 再現性:
  cache source drift、Parquet row order、Kaggle package の埋め込み config drift を
  SHA、strict order check、bootstrap audit で fail closed にする。
- 検定の過学習:
  offset scale、clip、candidate subset を固定し、結果を見た後の rescue を禁止する。
