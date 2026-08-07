# 設計

## アプローチ

exp072のdeterministic PF ANCC / PF-Z実装を厳密に再現し、各wellで元seedを含む64 seedを生成する。
全seed trajectoryを恒久保存せず、well処理中にseed別診断、path quantile、nested aggregateを計算して
メモリとKaggle output量を抑える。元seedがseed分布内でどれほど極端か、良好状態が何seedで再現するか、
seed平均・中央値が収束するかをalgorithm別・well別・距離帯別に判定する。

## 実験範囲

- 対象実験: `exp266_pf_ancc_pf_z_multiseed_stability_audit`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較実験: `exp106_strict_exp072_pf_z_multiseed_scale_cache`、`exp205/209 exact HMM`、`exp226`、先行candidate-path condition audit
- 変更する変数: PF ANCC / PF-Zのseed countを1から64へ増やす。集約readoutだけを追加する。
- 固定する変数: raw input、評価行、particles 600、PF transition / observation / clamp / resampling、初期化、全algorithm parameter、seed以外のcandidate path。
- seed counts: `[1, 4, 8, 16, 32, 64]`。64 seed順序は事前固定し、prefix集約を結果で並べ替えない。
- aggregation: arithmetic mean、row-wise median、row-wise 10% trimmed mean。likelihood weightingやtargetによるweightingは行わない。

## 再現性設計

- seed policy: index 0は`stable_seed("pf_ancc", well)` / `stable_seed("pf_z", well)`。index 1〜63は`stable_seed(exp266, split, algorithm, well, seed_index)`。
- stochastic 処理の有無: particle初期化、process noise、resamplingに乱数を使うが、Numba kernelへ明示seedを渡す。
- PF/Beam / likelihood-PF / seed bagging の有無: PF ANCC / PF-Zのみ。Beamとlikelihood-PFは再生成せず、保存済み比較値を読む。
- 並列処理と乱数の関係: well-level `joblib` thread並列を使い、各well/algorithm/seedの乱数列を事前固定する。global Python/NumPy RNGへ依存しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、8 workers、GPUなし。exp106実測10,111.57秒/64 PF-Z seedsを根拠に両方式約5.5〜7時間を見込む。
- train cache / test feature regeneration の SHA 記録方針: canonical exp072 raw/decompressed SHAとschema SHAをhard guardする。train-side diagnosticのみでtest再生成はしない。
- model manifest / prediction / submission SHA 記録方針: model/inference/submissionなし。seed-level metrics、aggregate row paths、summaryのSHAとgzip decompressed content SHAを保存する。
- Kaggle package bootstrap 確認方針: source、loose support、bootstrap内config/source SHAをpush前に一致確認する。
- parity guard: seed index 0の両algorithmをexp072保存列と全行exact比較し、max abs / RMSE差が0でなければfail closedとする。
- occurrence readout: 全wellのseed分布を凍結後、`11d0f5ac`とstrong phenotypeをラベル付けし、prefix長、eval長、距離、GR/typewell quality、PF particle std、candidate disagreementとの関連を評価する。

## リスク

- リークリスク: true TVTをseed/path生成、seed順序、集約weightへ戻すとoracleになる。全well path生成完了後の診断に限定する。
- CV/LB 不一致リスク: train pseudo-tail診断でありLB改善を主張しない。inference / submitへ自動昇格しない。
- ランタイム/メモリリスク: 2方式 × 64 seeds × 600 particlesで約5.5〜7時間。well単位で処理し、全seed全row tensorを全well分保持・保存しない。
- 再現性リスク: exp072実装とのdtype、seed `+1`、resampling順序差でparityが崩れる可能性がある。exact parityをfull実行の採用条件にする。
- 多重比較リスク: thresholdは5/10 ft、seed countと集約規則を事前固定し、結果後のseed選別・threshold追加・parameter gridを禁止する。
