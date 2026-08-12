# 設計

## アプローチ

exp148 の `learned_likelihood_confidence_addonly` feature surface を固定し、supervised target だけを row-to-row step delta に差し替える。LightGBM の config family は exp148 と同じだが、ユーザー指定どおり `lgb0` だけを CPU deterministic mode で 5 folds 実行する。

評価時は raw predicted delta の RMSE ではなく、well ごとに row order で並べて `last_known_tvt + cumsum(pred_step_delta)` に戻した `pred_tvt` の RMSE を primary とする。これにより NFL 型 step-delta target が long-tail の形状一貫性を改善するか、逆に cumulative drift を起こすかを直接見る。

## 実験範囲

- 対象実験: `exp222_row_step_delta_target_ablation_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: supervised target と TVT 復元評価
- 固定する変数: exp148 feature surface、GroupKFold by well、LightGBM lgb0 config、5 folds、learned-likelihood feature columns、U-projection feature columns
- 範囲外: inference port、submission、lgb1/lgb2、control 再学習、autoregressive feature、OOF prediction feature

## Target 定義

- `target_anchor_delta = target = TVT_i - last_known_tvt`
- `target_tvt = last_known_tvt + target_anchor_delta`
- row order は `id` の suffix row number を主キーにし、`well` ごとに単調性を検査する。
- `row_within_tail == 0` の `target_step_delta` は `target_tvt_i - last_known_tvt`
- それ以降の `target_step_delta` は `target_tvt_i - target_tvt_{i-1}`
- OOF prediction は `pred_step_delta` として保存し、`pred_tvt = last_known_tvt + cumsum(pred_step_delta)` に復元する。

## 再現性設計

- seed policy: LightGBM config seed と GroupKFold seed 42 を固定する。
- stochastic 処理の有無: LightGBM の bagging / histogram training がある。custom RNG は `max_train_rows` sampling のみだが今回は `null`。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。既存 exp072 / exp145 Kaggle output を input source として読む。
- 並列処理と乱数の関係: CPU `deterministic: true`、`force_col_wise: true`、`n_jobs: 8`、`num_threads: 8`。
- CPU/GPU runtime: CPU only。Kaggle metadata は `enable_gpu: false`。
- train cache / test feature regeneration の SHA 記録方針: helper が input file SHA / gzip decompressed SHA を summary / manifest に記録する。train-side only なので hidden test regeneration はしない。
- model manifest / prediction / submission SHA 記録方針: model manifest、各 booster SHA、OOF prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と bootstrap 内 config が CPU / lgb0 / input sources を指していることを確認する。

## リスク

- リークリスク: 前 row true TVT は label 作成に必要だが、feature には使わない。評価復元は predicted step delta だけで行う。
- CV/LB 不一致リスク: target 変更は過去 exp095/117/138 で long-tail を壊した例がある。global CV が近くても near row、worst-well、hidden-like stress、cumulative drift を guard にする。
- ランタイム/メモリリスク: CPU lgb0 5 folds でも exp148 full surface のため重い。`max_rows` は本番では `null`、debug でだけ制限できる。
- 再現性リスク: CPU deterministic flags でも環境差はありうるため、deterministic submission anchor とは扱わない。
