# 設計

## アプローチ

1. raw trainとvisible testのhorizontal GR availabilityからmissing well/run inventoryを作る。visible testは分布監査だけに使う。
2. exp209/exp205の保存済みraw exact-HMM cacheから`hmm_mean_tvt`、`hmm_std`、`hmm_loglik`を読み、
   row ID、well、target/last-known TVT、SHAを固定controlとして検証する。
3. exp209 exact HMM kernelへexp247で検証済みの`raw_gr_missing` mask境界だけを移植する。補間GRはcontrolと
   同じように計算し、GR emission構築後にraw-missing rowを`emission_ll[row, :] = 0.0`へ置換する。
4. Stage 1 variant 1本を全773 wellsで生成し、固定controlとpaired評価する。LGB/self-GR/追加unaryは使わない。
5. row/group/by-well/finite/divergence/posterior診断とinput/output SHAを保存する。
6. 事前guardを判定し、全通過なら`pf_stage_eligible=true`だけを記録する。likelihood-PFコード実行、
   raw-test inference、submissionは別承認まで行わない。

## 実験範囲

- 対象実験: `exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- control artifact ancestor: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- 実装・診断参照: `exp247_missing_gr_masking`
- 変更する変数: evaluation suffixのraw horizontal GR欠損rowにおけるtypewell-GR emission contributionのみ。
- 固定する変数: exp209 HMM grid/transition/rate/initialization/GR sigma/interpolation、score rows、hidden-like定義。
- 学習コスト: active variant 1、LightGBM config 0、fold 0、booster 0、parent/control再生成なし。
- Stage 2: disabled。HMM guard通過後も自動実行せず、別run前にユーザー承認を得る。

## 診断と事前guard

- missing-run bucket: `observed`, `1_4`, `5_31`, `32_127`, `128_255`, `256_999`, `1000_plus`。
- post-gap bucket: `post_gap_001_128`, `post_gap_129_256`, `post_gap_257_plus`。
- distance bucket: `000_050`, `050_100`, `100_250`, `250_500`, `500_1000`, `1000_plus`。
- divergence: `abs(variant-control) > 1e-6 ft`の連続segment、missing overlap、最大絶対差。
- focus well `11d0f5ac`は評価readout専用で、maskやpath生成には使わない。
- Stage 1 passは次をすべて満たすこととし、結果後に変更しない。
  - overall RMSE delta `<= -0.02 ft`。
  - raw-missing RMSE delta `<= 0`、observed RMSE delta `<= +0.02 ft`。
  - `1000_plus`とhidden-like 2群のRMSE deltaが各`<= +0.02 ft`。
  - worst-well RMSE regression `<= +0.25 ft`。
  - prediction/std finite coverage 100%、controlとのID mismatch 0。
- 1項目でも不通過ならPF Stage 2をfail-closeする。

## 再現性設計

- seed policy: `no_new_rng_raw_exact_hmm_missing_gr_neutrality_ablation`。
- stochastic 処理の有無: Stage 1は新規RNGなし。固定controlのupstream HMMもRNGなし。
- PF/Beam / likelihood-PF / seed bagging の有無: Stage 1ではなし。Stage 2はdisabled。
- 並列処理と乱数の関係: wellをsortし、outer workers 2 / Numba threads 2を記録する。乱数系列はない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false、モデル学習なし。
- train cache / test feature regeneration の SHA 記録方針: raw train file inventory SHA、control raw/decompressed SHA、
  variant raw/decompressed SHA、row/group/by-well/finite/divergence SHAをsummaryへ保存する。
- model manifest / prediction / submission SHA 記録方針: model/submissionなし。variant prediction content SHAを記録する。
- Kaggle package bootstrap 確認方針: prepare後にmetadata、bootstrap内config、kernel source、CPU/GPU/internet、
  active variant/PF disabledを照合する。

## リスク

- リークリスク: raw GR availabilityだけをmaskへ渡す。true TVT/error/oracle/hidden-like/focus IDは生成関数へ渡さない。
- CV/LB 不一致リスク: train-side paired auditであり、positiveでもraw-test inference/submitへ自動昇格しない。
- ランタイム/メモリリスク: exact HMMは数時間級。controlを再生成せずvariant 1本だけ生成し、row outputはgzipにする。
- 再現性リスク: NumPy/Numba/thread環境で末尾差があり得るため、version/threadとdecompressed content SHAを記録する。
- 原因分離リスク: self-GR HMMを親にするとGR window補間まで変更対象が広がるため、本実験ではexp209 raw HMMに限定する。
- 解釈リスク: exp247のtiny gainを事前根拠にせず、固定guard不通過ならPF Stage 2を閉じる。
