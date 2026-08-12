# 設計

## アプローチ

exp188 の exp148 add-only 学習器を足場にし、selector feature block を exp184 heatmap compact に差し替える。exp148 の base/projection/learned-likelihood feature assembly と LightGBM 3 config x 5 fold の学習 flow は維持するが、Kaggle runtime は CPU にし、学習 notebook は `lgb0` / `lgb1` / `lgb2` の 3 本に分割する。

exp184 selected path は OOF best Viterbi artifact から読み、selected candidate/family、`likpf_mean` との差分、exp148 OOF との差分、segment stability を作る。heatmap confidence は exp182 validation prediction の predicted topK TVT/score と sparse sample row location だけから再生成する。

## 実験範囲

- 対象実験: `exp184_heatmap_selector_compact_addonly_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- heatmap 親: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- 変更する変数: compact heatmap/selector confidence feature group を追加する。
- 固定する変数: exp148 base 196 features、projection correction、u disagreement、learned likelihood confidence、GroupKFold 5 folds、LightGBM config family。
- 実行形態: `cpu_deterministic_threads8` を active mode にし、各 split notebook は 1 LightGBM config だけを学習する。
- 範囲外: control retraining、exp184 selected TVT direct replacement、blend、postprocess、hard gate、inference port、submit。

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM config seed は exp148/exp188 と同じ。
- stochastic 処理の有無: feature merge には新規乱数なし。CPU LightGBM は deterministic flags 付きだが bitwise deterministic anchor と扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072/exp145/exp184 artifact を読むだけで、この実験では再生成しない。
- 並列処理と乱数の関係: feature generation は row-order aligned merge と within-well interpolation。global RNG は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle GPU disabled、LightGBM CPU `deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: exp072、exp145、exp184 OOF、exp182 heatmap prediction の file/decompressed SHA を summary / manifest に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest、model SHA、OOF prediction SHA、feature schema SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train_lgb0/train_lgb1/train_lgb2` 後に kernel sources、`enable_gpu=false` metadata、selected variant、active mode、selected lgb config を確認する。

## リスク

- リークリスク: exp184/exp182 artifact には true/error/oracle columns が含まれるため、reader で usecols を限定し、禁止列が feature usecols に入ったら失敗させる。
- CV/LB 不一致リスク: exp184 は train-side heatmap validation artifact 依存のため、positive でも raw-test/current-test feature generation parity が必要。
- ランタイム/メモリリスク: exp148 full rows に heatmap interpolation と selected path merge を追加する。exp188 の fold-time `float32` materialization と preview 軽量化を維持する。
- 再現性リスク: upstream exp182 は GPU PyTorch artifact。downstream LightGBM は CPU deterministic flags 付きだが、deterministic submission anchor ではなく train-side audit として扱う。
