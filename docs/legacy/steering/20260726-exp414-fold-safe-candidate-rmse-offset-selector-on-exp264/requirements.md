# 要件

## 依頼

exp407 の悪化を説明だけで終えず、真の根本原因を保存済み OOF から定量的に
特定する。その原因を避けながら候補別 RMSE を利用する固定手法を 1 つ実装し、
corrected exp264 Stage B v5 と同条件の OOF で有効性を判定する。

## 制約

- Route: `ml_model`
- 親 selector は
  `exp264_exp263_candidate_confidence_dual_selector` corrected Stage B v5 とする。
- 比較対象は親 OOF と exp407 OOF の保存済み artifact に固定し、両者を再学習しない。
- 12 候補、候補順、11 候補 primary hard-selection domain、88 列 feature schema、
  5 outer folds、deterministic sampled fit row IDs、LightGBM parameter を固定する。
- 候補別 RMSE は、各 outer model が実際に使う sampled outer-train fit rows だけから
  計算する。outer-valid、global OOF、hidden-like、current test の truth を禁止する。
- 候補別 RMSE を sample weight、目的変数の除算、binary objective の重み、
  feature、候補削除には使わない。
- 固定 treatment は、`pred_abs_error` 回帰の additive base offset のみとする。
  `p_within10` は再学習せず、exp264 保存済み score を比較資料として扱う。
- exponent、clip、offset scale、candidate subset、fold 別の救済調整を探索しない。
- 初回 Stage B は variant 1、LightGBM config 1、outer fold 5、
  合計 CPU booster 5、親/control 再学習 0、GPU booster 0 とする。
- Stage C、inference、submission は対象外とする。
- Kaggle Notebook 実行を正とし、ローカル Notebook 実行は行わない。
- 再現性は `docs/06_reproducibility.md` に従い、入力、sample row IDs、
  offset table、feature schema、model、OOF、gate の SHA を保存する。

## 受け入れ基準

### 根本原因

- 親 OOF に exp407 の candidate × fold 平均 score shift だけを適用した
  counterfactual と、exp407 から同平均 shift を除いた row-local
  counterfactual を同一行で比較する。
- mean-shift-only が親を悪化させず、row-local-only が exp407 の悪化を再現する。
- exp407 の最終 inverse-RMSE weight が小さい候補ほど、
  親からの row-local score 変動または calibration 悪化が大きい dose-response を
  candidate × fold 単位で確認する。
- hard selection の悪化が tie 近傍だけでなく、親 margin 0.5--2.0 の
  confident row にも集中することを記録する。

### RMSE offset treatment の技術 gate

- 各 fold の RMSE table が exact sampled fit rows のみで計算され、
  fit / valid well overlap が 0 である。
- training sample weight は全行 1、binary model fit は 0、regressor は 5 本である。
- residual target が
  `actual_abs_error - fit_candidate_rmse`、再構成 score が
  `max(0, residual_prediction + fit_candidate_rmse)` と一致する。
- candidate order、feature count / logical SHA、sample row-ID SHA、
  input OOF SHA、offset table SHA、各 model SHA、OOF SHA が期待値と一致する。
- NaN / inf、行欠落、重複 key、candidate count / order drift がない。

### RMSE offset treatment の科学 gate

- pooled candidate expected-error MAE が親以下で、fold 別 4/5 以上が親以下である。
- hard-primary pooled OOF RMSE が親 `8.587004386703422` 以下で、
  fold 別 3/5 以上が親以下である。
- near 0--250、1000+、hidden-like spatial、hidden-like typewell-purged の
  hard-primary RMSE delta がそれぞれ親比 `+0.02 ft` 以下である。
- 親に対する worst-well RMSE regression が `+0.25 ft` 以下である。
- exp407 で見られた低 RMSE-weight 候補ほど大きい score instability を、
  treatment が pooled に増幅しない。
- 全 gate を満たした場合だけ、候補別 RMSE を additive base offset として使う
  方法を Stage B で確立したと判定する。FAIL 時は同一 OOF 上の rescue や
  parameter 探索をせず、この手法を閉じる。

- submission は生成しないため submission SHA は対象外とする。
- deterministic anchor と呼ぶ場合は、Kaggle kernel version と上記 artifact SHA を
  すべて記録する。
