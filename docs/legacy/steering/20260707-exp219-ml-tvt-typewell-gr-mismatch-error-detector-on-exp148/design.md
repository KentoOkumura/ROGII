# 設計

## アプローチ

exp148 OOF prediction の `pred_tvt` を、typewell TVT 軸上の provisional alignment として扱う。各 row で `pred_tvt + [-50,-25,-10,0,10,25,50]` ft の候補位置を作り、typewell GR window と horizontal GR window の raw / denoised 類似度を計算する。

類似度は window RMSE、NCC、derivative NCC、center GR 差、missing rate から score 化する。offset=0 の score、best offset、score gap、entropy、shuffled typewell decoy との差、raw-vs-denoised gap を confidence feature として保存する。

最初は no-training readout のみ行う。`abs_error_gt10` AUC と high-mismatch bucket の error lift が出た場合だけ、同じ exp219 内で exp148/exp193 add-only LightGBM を追加実装する。

## 実験範囲

- 対象実験: `exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: ML-predicted TVT 周辺の GR mismatch / confidence feature と readout
- 固定する変数: exp148 OOF prediction、raw train files、validation label、親 baseline、no-training 方針

## 再現性設計

- seed policy: no RNG。shuffled typewell decoy の roll だけ well id から SHA256 で決定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: 初期実装は single process。parallel RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU readout。GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp148 OOF prediction gzip は decompressed SHA、生成 feature cache gzip も decompressed SHA、schema は file SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: 初期実装では新規 model / prediction / submission なし。入力 prediction SHA のみ記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、package 内 `.py` を py_compile する。

## リスク

- リークリスク: `target_tvt`、`abs_error`、`abs_error_gt*` は feature cache に readout label として保存されるため、downstream add-only に進む場合は schema role を確認して feature から除外する。
- CV/LB 不一致リスク: typewell GR mismatch は平坦区間、二峰性、typewell 非代表性、GR横方向変化で偽陽性が多い。global AUC だけで進めない。
- ランタイム/メモリリスク: 3.78M rows x 7 offsets x window 65 を well 単位で処理する。CPU readout だが output feature cache は大きくなる可能性がある。
- 再現性リスク: Kaggle input source に exp145 optional feature cache がない場合は candidate disagreement を 0 fallback する。その有無は summary に記録する。
