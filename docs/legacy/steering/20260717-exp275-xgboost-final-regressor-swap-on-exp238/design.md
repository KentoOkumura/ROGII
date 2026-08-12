# 設計

## アプローチ

`exp238` の final train と同じ入力再構成を行い、保存済み outer-fold nested selector score から各 fold 専用の35 rank-slot特徴を作る。base 380列と結合した415列を XGBoost に入力し、`last_known_tvt` からの residual を予測する。

XGBoost は `cdeotte/xgb-starter-cv-15` version 3 の full-run設定をそのまま使う。公開 notebook は early stopping を使っていないため、450 trees を各 fold で固定実行する。`n_jobs=-1` は公開値として parameter audit に保存するが、Kaggle GPU の非決定性は残るため deterministic anchor とは扱わない。

## 実験範囲

- 対象実験: `exp275_xgboost_final_regressor_swap_on_exp238`
- Route: `ml_model`
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: 最終 TVT 回帰器を LightGBM 3 configs 平均から XGBoost 1 public config に差し替える。
- 固定する変数: `exp238` の outer fold role、residual target、sample weight policy、380 base特徴、35 nested rank-slot特徴、candidate順序、行順序、stress surface、保存済み LightGBM OOF。

## 再現性設計

- seed policy: 公開 notebook 固定 `random_state=42`。fold は保存済み `exp238` outer role を正とする。
- stochastic 処理の有無: XGBoost GPU histogram training のみ。特徴と selector score は保存済み入力から決定的に再構成する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。親特徴 cache の保存済み値を再構成ロジックで読むだけ。
- 並列処理と乱数の関係: `n_jobs=-1` は公開設定を維持する。Python global RNG は使わない。GPU/thread scheduling による bitwise 再現性は主張しない。
- CPU/GPU runtime と deterministic flags: Kaggle Nvidia T4、XGBoost `tree_method=hist`, `device=cuda`。公開設定にない deterministic flag は追加しない。
- train cache / test feature regeneration の SHA 記録方針: foldごとの415列 schema SHA、float32 train/valid matrix content SHA、selector score artifact decompressed SHA を記録する。test regeneration は未実装。
- model manifest / prediction / submission SHA 記録方針: 5 model の各 SHA、manifest SHA、OOF decompressed SHA を記録する。submission は生成しない。
- Kaggle package bootstrap 確認方針: strict prepare 後に kernel source、GPU/internet metadata、bootstrap内 `config.yaml` と公開パラメータの一致を確認する。

## リスク

- リークリスク: foldごとに outer-valid 向け selector score と outer-train 向け inner-OOF score を分離する。score artifact の role、row coverage、id/well alignment、well overlap 0 を fail-closed にする。
- CV/LB 不一致リスク: `exp257` で CV/LB 方向反転があるため、CV guard 通過だけで提出しない。
- ランタイム/メモリリスク: 3.78M × 415 float32 を foldごとに生成し、同時に1 foldの train/valid matrixだけを保持する。XGBoost `DMatrix` の追加メモリを考慮し、fold終了ごとに解放する。
- 再現性リスク: GPU XGBoost は bitwise deterministic とみなさない。採用候補になった場合だけ rerun で prediction SHA とスコア差を監査する。

## 2026-07-18 参考推論・スコアリング設計

- exp274 reference inferenceで成立したexp238 current-test再生成を同じ入力sourceで使い、final estimator loaderだけをXGBoostへ置き換える。
- train version 2のsummary SHA `12fbac5...bb143`、manifest SHA `0ecffa5...0d5a`、feature schema SHA `85d57f2...805c`、5 model SHAを検証する。
- 各outer foldで保存済みselector 4本を平均し35 rank-slot特徴を作り、そのfoldのXGBoost 1本とparent LightGBM 3本を同じ415列matrixへ適用する。
- primaryは5 XGBoost residual予測の平均。parent 15本平均と固定`0.75 parent + 0.25 XGBoost`はprediction drift監査用にだけ保存する。
- root `submission.csv`はraw XGBoostのみ。notebook自体はsubmitせず、output取得後のローカルsubmit-checkを通過してからKaggle CLIで1件だけsubmitする。
- raw guard不通過、reference override、submission authorization、0 training、0 fallbackをsummaryへ明記し、LBは採用根拠ではなく参考値として記録する。
