# 設計

## アプローチ

exp111 の OOF candidate-long learned likelihood を再学習せずに読み、exp099 v2 wide cache から評価用 true TVT を復元する。5 候補の candidate matrix を作り、posthoc に以下を比較する。

- `likpf_mean_single` baseline
- exp111 `learned_prob_top1` / `learned_error_top1` diagnostic
- exp099 `multiobs_score_top1` baseline
- target-free multiobs score に learned probability / expected-error signal を弱く加える PF weight alpha
- `likpf_mean` fallback を保った conservative verifier gate
- oracle candidate

同時に、exp092 系 ML add-only audit に使える target-free row-level learned likelihood feature cache を保存する。

## 実験範囲

- 対象実験: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- Route: `pf_beam`
- 親実験: `exp111_learned_pf_observation_likelihood_probe`
- 変更する変数:
  - PF weight alpha: `0.05, 0.1, 0.2, 0.4`
  - verifier gate の probability / margin / delta cap
  - ML feature artifact の learned likelihood summary
- 固定する変数:
  - exp111 OOF likelihood predictions
  - exp099 fixed candidate values
  - 候補集合: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`

## 再現性設計

- seed policy: 新規 RNG なし。
- stochastic 処理の有無: exp112 内ではなし。上流 exp111 LightGBM と exp072/exp099 PF/Beam cache は stochastic 由来。
- PF/Beam / likelihood-PF / seed bagging の有無: exp112 では再生成しない。上流 cache を固定入力として扱う。
- 並列処理と乱数の関係: 新規並列 RNG なし。
- CPU/GPU runtime と deterministic flags: CPU notebook。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: exp111 OOF long と exp099 wide cache の raw/decompressed SHA を summary に記録する。raw-test regeneration は行わない。
- model manifest / prediction / submission SHA 記録方針: 新規 model なし。OOF prediction と ML feature cache の raw/decompressed SHA を記録する。submission SHA は対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に generated package の config と kernel source を確認する。

## リスク

- リークリスク: exp099 true TVT は評価専用に使う。ML feature cache には target-derived columns を含めない。
- CV/LB 不一致リスク: exp111 first-fold OOF の train-side surface のみであり、hidden test への転移証拠ではない。
- ランタイム/メモリリスク: exp111 long cache は約 127MB gzip。pivot 後の行列は candidate 数 5 に限定する。
- 再現性リスク: 上流 cache の再現性に依存するため、exp112 単体を deterministic submission anchor とは呼ばない。
