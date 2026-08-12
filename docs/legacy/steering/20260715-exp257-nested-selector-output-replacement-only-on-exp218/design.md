# 設計

## アプローチ

exp218の380特徴には`ll_*` learned-likelihood blockが54列ある。そのうちselectorの
rank/probability/error/spread/weighted-pathに相当する29列だけを、exp238 nested selectorの
11候補scoreから生成した値で上書きする。multi-observation 15列とlegacy 5候補の
anchor/likPF差10列はselector入力診断なので維持する。列の追加・削除・並べ替えは行わない。

exp238 scoreはpredicted absolute errorであるため、`1 / max(score, 1e-3)`を行ごとに正規化し、
既存probability slotへ入れる。error slotには元scoreを入れる。rank/spread/weighted TVTは
HMM・self-GR HMM・exp226を含む11候補全体で計算する。candidate別の既存slotはlegacy 5候補だけを
更新し、新候補専用one-hotや`nsel_*`列は作らない。

## 実験範囲

- 対象実験: `exp257_nested_selector_output_replacement_only_on_exp218`
- Route: `ml_model`
- 親実験: feature surface `exp218_gr_wavelet_rotation_confidence_features_on_exp148`、selector source `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: 既存selector出力29列の値だけ
- 固定する変数: 380列schema/order、残り351列、outer fold contract、11 candidate bank、selector scores、LightGBM 3 config family、seed 42

## 再現性設計

- seed policy: exp218/exp238と同じseed 42。新規selector/PF/Beam生成は行わない。
- stochastic 処理の有無: 新規stochastic feature generationなし。最終GPU LightGBM学習のみ。
- PF/Beam / likelihood-PF / seed bagging の有無: exp238固定OOF candidate/score artifactを読むだけで再生成しない。
- 並列処理と乱数の関係: LightGBMはexp218のdeterministic、force_col_wise、threads 8、gpu_use_dpを継承する。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU、1 variant × 3 configs × 5 folds。parent/control再学習なし。
- train cache / test feature regeneration の SHA 記録方針: selector score gzipはdecompressed SHA、fold/id/well/role coverageを検証する。
- model manifest / prediction / submission SHA 記録方針: 15 model SHA、380 schema SHA、OOF decompressed SHA、metrics SHAを保存する。CV通過前はsubmissionなし。
- Kaggle package bootstrap 確認方針: package config/source SHA、kernel source、GPU/internet/run_on_pushをpush前後で照合する。

## リスク

- リークリスク: outer-validにはouter-train inner 4 model平均scoreだけを使い、同fold真値からrank/errorを作らない。
- CV/LB 不一致リスク: exp218 historical OOFとはfold assignmentが異なるため因果比較に使わない。exp238 add-only OOFとは同じfold contractで比較する。
- ランタイム/メモリリスク: 約378万行×380列。fold単位score遅延読込、float32列chunk充填、boosterごとの解放を使う。
- 再現性リスク: exp238 selector artifactのversion/SHAを固定し、違えばfail-fastする。
