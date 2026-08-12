# 設計

## アプローチ

exp073 raw deterministic OOF baseline の誤差を、exp077 で保存した fold 平均 LightGBM feature importance と exp072 full replay feature cache の値で読む。上位重要特徴量を選び、各特徴量の quantile bucket ごとに baseline / compare policy の RMSE、MAE、error mean、SSE delta を出す。さらに誤差相関と well summary を保存し、次の confidence feature / sample weight / guard 条件の候補を絞る。

## 実験範囲

- 対象実験: `exp086_oof_feature_importance_error_readout`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- readout parent: `exp077_full_replay_postprocess_guard`
- 変更する変数: readout の集計軸、上位特徴量数、feature quantile 数
- 固定する変数: exp073 OOF baseline、exp077 best fixed policy、exp072 feature surface、LightGBM model family

## 再現性設計

- seed policy: 新規 stochastic 処理なし。correlation sampling は `random_state=42` 固定。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072/exp073/exp077 の既存生成物だけを読む。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime と deterministic flags: 学習しないため該当なし。Kaggle runtime は CPU で十分。
- train cache / test feature regeneration の SHA 記録方針: 入力の policy predictions、metrics、feature importance、feature cache の file SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: 新規 model/prediction/submission は作らない。readout output は summary JSON と CSV/PNG 生成物として保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --notebook train --strict` で source files と config が bootstrap に含まれることを確認する。

## リスク

- リークリスク: train OOF と train feature cache だけを読む診断なので、新しい推論 branch には直結させない。診断結果を使った gate や sample weight は別実験で fold-safe に検証する。
- CV/LB 不一致リスク: この実験は CV/LB を改善したとは見なさない。error map の読み取りだけに限定する。
- ランタイム/メモリリスク: exp077 policy predictions は大きいため chunk read で対象 policy だけに絞る。feature cache は上位特徴量と診断列だけを `usecols` で読む。
- 再現性リスク: 入力生成物の source version が変わると readout も変わるため、summary に path と SHA を保存する。
