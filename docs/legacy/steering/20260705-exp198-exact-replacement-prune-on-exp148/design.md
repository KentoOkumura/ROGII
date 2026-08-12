# 設計

## アプローチ

exp148 の compact self-contained train 実装を親にして、feature group 生成と LightGBM config family はそのまま使う。差分は `exact_replacement_prune` feature group を追加し、active variant では exp148 の `projection_correction`、`u_disagreement`、`learned_likelihood_confidence` を維持したうえで、17 列を `drop_columns` として active model features から除外する。

この実験では replacement 候補の TVT 値を予測・混合に使わない。目的は「exp148 の add-only feature surface に混入した完全重複/符号反転/定数列を落としても anchor が保てるか、または改善するか」を train-side CV で確認することに限定する。

## 実験範囲

- 対象実験: `exp198_exact_replacement_prune_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: active model feature list からの 17-column drop-only prune
- 固定する変数: exp072 base feature cache、exp092 U-projection settings、exp145 learned likelihood feature cache、target、GroupKFold by well、LightGBM lgb0/lgb1/lgb2 configs、seed 42、GPU deterministic mode
- 比較基準: exp148 `lgb_mean` CV `8.50128118189582` / Public LB `7.960`

## 再現性設計

- seed policy: exp148 と同じ seed 42、GroupKFold by well、LightGBM config family を使う。
- stochastic 処理の有無: 新規 stochastic feature generation はない。train は LightGBM GPU 学習のみ環境差の可能性がある。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072/exp092/exp145 の保存済み deterministic cache を読むだけで、この実験では再生成しない。
- 並列処理と乱数の関係: feature assembly は保存済み cache join と deterministic transform。LightGBM は exp148 と同じ fixed thread settings を使う。
- CPU/GPU runtime と deterministic flags: Kaggle train は `gpu_repro_guard_dp_threads8`、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`num_threads=8` を使う。control 再学習はしない。
- train cache / test feature regeneration の SHA 記録方針: train 実行後に feature schema、OOF prediction gzip の decompressed SHA、model manifest SHA を記録する。inference は train CV が支持された場合に同じ exp198 内で port する。
- model manifest / prediction / submission SHA 記録方針: train では model manifest と OOF prediction SHA を記録する。inference / submission はこの初回実装範囲外で、実施時に prediction SHA、submission SHA、kernel version を追記する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に生成 notebook bootstrap 内の `config.yaml` が exp198 の selected variant / drop list / kernel sources と一致することを確認する。

## リスク

- リークリスク: 新規特徴量は足さず既存 target-free cache を読むため低い。drop list の列名取り違えが主リスクなので、`corr_prune_sanity_readout_on_exp148` の YAML/JSON fragment と照合する。
- CV/LB 不一致リスク: exp162 など exp148 派生では CV 改善が Public LB に転移しない例がある。CV が改善しても near-row、`1000_plus`、worst-well、hidden-like stress、schema parity を確認するまで inference / submit へ進めない。
- ランタイム/メモリリスク: exp148 と同等以下の feature count なので train runtime は exp148 GPU train と同等以下を想定する。CPU fallback は約 10 時間規模のため既定では使わない。
- 再現性リスク: LightGBM GPU は bitwise reproducible と決めない。採用候補にする場合は kernel version、feature/model/prediction SHA を記録し、必要なら CPU deterministic control を別途相談する。
