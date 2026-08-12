# 設計

## 結論

`exp431` は `exp427` の technical/scientific AND gate が完全 PASS した場合だけ実装可能な条件付き設計とする。exp404 x1.0 の固定 128 seed PF を一度だけ再生し、同一軌跡に exp427 の fixed 2x2 likelihood family を適用する。scientific candidate は `affine_ar1` のみである。

設計時点の exp427 は `stage_0_v2_running` で結果未確定だったため、本実験を
`design_frozen_waiting_exp427_gate` として実装・push禁止にした。

## 2026-07-29 gate結果

exp427 version 2は完走したが、technical / scientific AND gateをともにFAILした。
technicalではeligible block fraction `0.721073584 < 0.75`、scientificでは
primary `affine_ar1` MRR `0.386090045`がmatched `0.388002620`とsaved exp280
`0.388146378`を下回った。決定は`stage_0_failed_close_without_rescue`である。

したがって必須先行gateの4番を適用し、exp431を
`closed_prerequisite_failed`として未実装のまま閉じる。ユーザーの実装依頼は
別途実装承認には相当するが、technical/scientific完全PASSという科学的先行条件を
置き換えない。same-OOF rescue、support/gate緩和、PF replay、推論、提出は行わない。

## 系譜

- PF 親: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- seed 集約の比較根拠: `exp417_scale5_seed_aggregation_promotion_audit`
- 尤度親・必須先行実験: `exp427_affine_ar1_whitened_gr_likelihood_readout`
- route: `pf_beam`
- 変更点: PF 軌跡生成後の seed evidence を exp427 の affine + AR(1) block posterior-predictive density に置換

## 必須先行 gate

実装開始には次をすべて要求する。

1. exp427 technical AND gate が PASS。
2. exp427 scientific AND gate が PASS。
3. exp427 の scientific contract、prefix affine posterior、fold rho、eligibility、target-free scores、gate JSON の content SHA が固定済み。
4. exp427 が terminal FAIL の場合、本実験も `closed_prerequisite_failed` とし、same-OOF rescue を行わない。
5. PASS 後も本実験の実装・Kaggle push は別のユーザー承認を必要とする。

## 固定 2×2 尤度

各 seed の suffix TVT 軌跡で Type Well GR を評価した系列を `x`、同じ raw row の horizontal GR を `y` とする。current-well known prefix だけから、exp427 と同じ affine posterior を一度計算する。

```text
y = intercept + slope * x + epsilon
theta_0 = [0, 1]
prior std = [20, 0.25]
sigma_w = clip(std(y_prefix - x_prefix), 10, 60)
```

- finite prefix pair 64 以上
- Type Well GR std 5 以上
- posterior mean slope > 0
- 不適格 well は `affine_ar1` を出さず、identity fallback で coverage を水増ししない

AR(1) は exp427 の outer-train-fold known-prefix residualから得た `rho_fold` をそのまま用いる。outer-valid well を rho source に入れず、`rho` は `[-0.8, 0.8]`、元 missing をまたぐ innovation は作らない。

block は suffix 先頭から非重複 512 raw rowsとし、raw finite GR 128 行以上かつ block の 50%以上を要求する。各 contiguous run で

```text
u_0 = sqrt(1-rho^2) * residual_0
u_t = residual_t - rho * residual_(t-1)

Sigma = sigma_w^2 I + Z V_w Z^T
log p_run = -0.5 * (
    n log(2*pi) + logdet(Sigma) + z^T inv(Sigma) z
)
```

を計算する。production は rank-2 Woodbury を使い、dense 式との max absolute error `<=1e-8` を要求する。

| readout | affine | covariance | 役割 |
| --- | --- | --- | --- |
| `identity_iid_matched` | identity、`V=0` | iid | factorial matched control |
| `affine_iid` | prefix posterior | iid | affine-only 診断 |
| `identity_ar1` | identity、`V=0` | fold-safe AR(1) | AR1-only 診断 |
| `affine_ar1` | prefix posterior | fold-safe AR(1) | 唯一の科学候補 |

## seed evidence と集約尺度

exp427 の block ranking は block 間比較のため mean log predictive density を使う。本実験では一つの well 内の seed posterior weight を作るため、各 eligible run/block の proper log predictive densityを合計する。

```text
seed_total_log_evidence = Σ_block Σ_run log p_run
w_s = softmax(
    (seed_total_log_evidence_s - max_s seed_total_log_evidence_s) / 5.0
)
prediction_t = Σ_s w_s * trajectory_prediction_(s,t)
```

`mean_log_predictive_density = total / finite_count` も監査列として保存するが、weight には使わない。readout 間で score の center は共有せず、それぞれの 128 seed 内で中心化する。temperature、score length normalization、weight clip、top-k、ESS rescue は探索しない。

保存済み exp404 T=5 は strong reference とするが、exp427 proper density は exp404 の capped Gaussian と同一式ではないため、bitwise matched と主張しない。`identity_iid_matched` は exp427 family 内の matched control、保存 exp404 は別 control として両方に勝つことを要求する。

## 入力と成果物

exp430 と同じ PF/input SHA 契約に加え、PASS した exp427 の以下を content SHA 固定で読む。

- scientific contract
- prefix affine posterior
- fold AR(1) prior
- eligibility
- gate JSON

実装時は trajectory bank、四つの per-seed total/mean evidence、finite count、weight、ESS、集約予測、factorial by-well/fold/scope/tail metrics、artifact SHA を保存する。

## 実行量

### technical preflight

- 固定 4 eligible well
- PF trajectory variant: 1
- PF well-runs: 4
- seed-well trajectories: 512
- particle starts: 256,000
- likelihood readout: 4（PF の追加実行ではない）

preflight は dense/Woodbury parity、support、common trajectory、weight finite、truth-late を調べるだけで、科学 gate に使わない。

### full CV readout

- wells: 773（affine eligibility は別途台帳化）
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

## technical AND gate

1. exp427 prerequisite の gate と全 expected content SHA が一致する。
2. 四 readout が同じ per-seed trajectory logical SHA を参照する。
3. support、512-row block、missing run、sigma、affine posterior、fold rho が exp427 contract と一致する。
4. dense/Woodbury score parityの max absolute error `<=1e-8`。
5. eligible score、weight、prediction に NaN/Inf がなく、weight sum が `1±1e-12`。
6. shard 順変更で trajectory/evidence/prediction logical SHA が一致する。
7. suffix truth、error、formation、hidden-like role の score freeze 前読込が 0。

一つでも FAIL なら full run をしない。

## scientific AND gate

`affine_ar1` を `identity_iid_matched`、保存済み exp404 T=5、二つの single factor に比較する。

- overall RMSE gain vs `identity_iid_matched`: `>=0.10`
- overall RMSE gain vs saved exp404 T=5: `>=0.10`
- overall RMSE gain vs `affine_iid`: `>=0.05`
- overall RMSE gain vs `identity_ar1`: `>=0.05`
- 各主要 control に対し reporting fold 4/5 以上で非劣化
- deep / shallow、missingness、roughness scope は全て非劣化
- paired per-well squared-error delta p95: `<=0`
- worst paired per-well RMSE delta: `<=0.25`
- eligible well fraction: `>=0.90`

すべて満たした場合だけ推論・提出候補化を別途判断する。partial main-effect PASS は `affine_ar1` の昇格理由にしない。

## 再現性

- PF seed: immutable well id × seed index 0..127 の stable SHA-256
- exp427 real score artifactは RNG なしで content SHA 固定
- well-seed ごとの local RNGを使い、shard/再開順に依存させない
- trajectory bank を readout 前に凍結
- score reduction は block/run/row の固定順
- Kaggle kernel id/version、package、CPU/Numba、input、exp427 artifact、trajectory、evidence、prediction SHA を記録
- cross-rerun SHA parity まで deterministic anchor としない

## 判断済みの分岐

- exp427 結果を待たずに実装: 不採用
- affine+AR1 だけを別 PF run: factorial common-trajectory 比較を壊すため不採用
- block mean scoreをそのまま weight に利用: suffix 長によって実効 temperature が変わるため不採用
- Huber/Student-t 併用: observation family の効果分離を壊すため不採用
- rho/affine prior/temperature 探索: exp427 の事前固定契約を壊すため不採用
