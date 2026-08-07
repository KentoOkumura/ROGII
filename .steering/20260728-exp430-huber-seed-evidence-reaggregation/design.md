# 設計

## 結論

exp404 x1.0 の PF を固定 128 seed で一度だけ再生し、保存した同一 trajectory bank に対して `gaussian_matched` と `huber_delta_1p345` の二つの evidence readout を計算する。PF 内部の particle likelihood は Gaussian のまま変更しない。科学候補は Huber 集約だけで、Gaussian は再生・集約の matched control とする。

## 系譜

- PF 親: `exp404_scale5_likpf_selector`
- seed 集約の比較根拠: `exp417_scale5_seed_aggregation_promotion_audit`
- Huber 式と失敗根拠: `exp389_exp209_huber_exact_hmm_emission`
- 変更点: PF 軌跡生成後の seed evidence を Huber 化するだけ
- route: `pf_beam`

exp417 の保存済み exp404 artifact は集約列だけで per-seed 軌跡を含まない。このため、保存 artifact の zero-PF 再集約は不可能であり、同一設定の trajectory bank を一度再生する。別候補として Gaussian 用と Huber 用に二度 PF を走らせることは禁止する。

## 固定する計算

各 well、seed、後半の有限観測行について、exp404 と同じ TypeWell 基準の標準化残差を `z_t` とする。

```text
gaussian_score = Σ_t -0.5 * min(z_t^2, 600)

huber(z; δ) =
    0.5 * z^2                         if |z| <= δ
    δ * |z| - 0.5 * δ^2              otherwise

huber_score = Σ_t -huber(z_t; 1.345)
```

各 readout 内で 128 seed の score を最大値で中心化し、固定 `T=5.0` で重み付けする。

```text
w_s = softmax((score_s - max_s score_s) / 5.0)
prediction_t = Σ_s w_s * trajectory_prediction_(s,t)
```

有限観測 support、suffix、欠損処理は exp404/exp417 と同じにする。Huber の追加 clip、weight floor、top-k、ESS rescue は導入しない。比較用に arithmetic mean も同じ bank から出力する。

## 入力と成果物

固定入力は exp417 の契約を引き継ぐ。

- exp072 feature: decompressed SHA `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp404 保存予測: decompressed SHA `00fe1b90fce84bd601b4b91442d9fc698200aafadd48658f7d8c26ec1fbe0d00`
- exp226 fold OOF: decompressed SHA `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp115 hidden assignment SHA `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`

実装時は最低限、per-seed trajectory bank、per-well/seed の二種 evidence、seed weight、ESS、集約予測、by-well/fold/scope/tail metrics、config、input/feature/prediction SHA を保存する。gzip は decompressed content SHA を主証拠にする。

## 実行量

### technical preflight

- 固定 4 well
- PF trajectory variant: 1
- PF well-runs: 4
- seed-well trajectories: 512
- particle starts: 256,000
- evidence readout: 2（Gaussian、Huber。PF の追加実行ではない）

### full CV readout

- wells: 773
- PF trajectory variant: 1
- particles: 500
- seeds: 128
- PF well-runs: 773
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- CPU shards: 4
- LightGBM config / folds / boosters: 0 / 0 / 0
- HMM / Beam / GPU: 0 / 0 / 0
- 保存済み親 control の独立再実行: 0

一度の trajectory replay が matched Gaussian control と Huber candidate の両方を供給する。

## gate

### 技術 gate

1. 固定 4 well で同じ seed の Gaussian/Huber が同一 trajectory SHA を参照する。
2. `gaussian_matched` が exp404 の score 式、有限 support、T=5 集約を再現する。
3. 並列 shard 数を変えても seed trajectory と集約 prediction の logical SHA が一致する。
4. 後半 truth が score 評価以外の軌跡生成、PF update、欠損補完に入らない。
5. NaN/Inf weight がなく、各 well の weight sum が `1±1e-12`。

technical gate を一つでも外した場合は full run を行わない。

### 科学・昇格 gate

Huber を `gaussian_matched` と保存済み exp404 T=5 の双方に比較する。

- overall RMSE gain: `>= 0.10`
- reporting fold: 5 fold 中 4 fold 以上で非劣化
- deep / shallow、missingness、roughness scope: 全 scope 非劣化
- paired per-well squared-error delta p95: `<= 0`
- worst paired per-well RMSE delta: `<= 0.25`
- arithmetic mean より overall RMSE が良い

上記を全部満たした場合だけ推論・提出候補化を別途判断する。preflight は promotion evidence に使わない。

## 再現性

- stable seed key: immutable `well_id` と seed index 0..127 の SHA-256
- joblib/thread の global RNG を使わず、well-seed ごとのローカル RNG を使う
- shard 順や再開順で seed stream を変えない
- trajectory bank は再採点前に凍結し、Gaussian/Huber が同一 logical SHA を参照する
- Kaggle kernel id/version、package versions、Numba/CPU 情報、config SHA、input SHA、artifact SHA を記録する
- CPU/Numba の完全決定性が再実行で確認されるまでは deterministic anchor と呼ばない

## 判断済みの分岐

- zero-PF 再集約: per-seed artifact 不在のため不採用
- Gaussian/Huber の別 PF run: common-trajectory 比較を壊すため不採用
- Huber delta 探索: exp389 との仮説分離が崩れるため不採用
- particle filtering 自体の Huber 化: 軌跡生成と再集約効果が混ざるため不採用

## 実装開始条件

ユーザーの明示承認後にだけ compact self-contained notebook/helper を実装する。本 steering の作成は実装・Kaggle push を承認しない。

## 2026-07-28 実装反映

- ユーザーの明示実装依頼を受け、compact self-contained train / inferenceを実装した。
- trajectoryはfullを4つのdeterministic LPT shardへ分け、各shard内でfloat64
  `.npy` memmapへ保存する。evidenceはbank SHA確定後にだけ計算する。
- technical preflightはSHA-first固定4 wellsを各1回だけPF再生し、同じbankを
  1 worker / 4 workersで再採点してprediction/evidence SHA parityを確認する。
- arithmetic parityは保存exp404の同一bank `likpf_mean_x1p0`列へ照合する。
  保存exp072 deltaのabsolute再構成は既知の丸め差があるため診断だけにする。
- roughness scopeは、frozen arithmetic-mean trajectoryのwell別二階差分RMSを
  全well中央値で二分するtarget-free定義に固定した。
- full shardはpreflight summary SHA、mergeは4 shard rootとsummary SHAを
  すべて固定しない限り開始しない。
- 2026-07-29にfixed 4-well technical preflightのpackage、push、runのみ
  ユーザー承認を得た。full shard、merge、inference、submissionは未承認のまま。

## 2026-07-29 technical preflight結果

- version 1は4 PF well-runsを完了したが、親float32予測をCSVからfloat64として
  再読込した比較器の表現差で2 parity checksがFAILした。
- 保存CSV同士は18,055行完全一致したため、toleranceを変更せず、両辺を親保存dtypeの
  float32へ正規化した。同じcanonical kernelのversion 2で12 / 12 checks PASSした。
- v1/v2のtrajectory、prediction、evidence raw SHAは完全一致した。
- preflightはpromotion evidenceではない。full 4 shard、merge、inference、
  submissionは別承認までfail closedを維持する。
