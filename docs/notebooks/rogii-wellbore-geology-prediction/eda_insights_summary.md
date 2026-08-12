# ROGII EDA / insight notebook survey

調査日: 2026-05-28

更新:

- 2026-06-03 に vote 上位を再取得し、`vote_top/kernel_listing.csv` を更新した。
- 同日に `scoreAscending` 最新上位と `dateRun` 最新上位を追加取得し、差分を `latest_public_notebooks_20260603.md` に整理した。

対象:

- Kaggle 公開 notebook の vote 上位 20 件。
- 既存の score ascending 上位 20 件メモ: `score_ascending/solution_summary.md`。

現在の再取得コマンド:

```bash
task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction EXTRA_ARGS="--limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/vote_top --sort-by voteCount"
```

`task`が利用できない環境では、同じ変数と引数で`make fetch-kaggle-notebooks`を使う。

保存先:

- `docs/notebooks/rogii-wellbore-geology-prediction/vote_top/kernel_listing.csv`
- `docs/notebooks/rogii-wellbore-geology-prediction/vote_top/*/*.ipynb`

## 結論

EDA / 重要知見として最初に読む価値が高いのは次の 5 系統。

2026-06-03 時点の追加観察では、public score 順の上位は PF/physical sel15、Sunny physical + 生成物を使った積み上げ、AeroRidge/PF/beam/TabICL family に寄っている。ただし EDA / 実験設計の基本方針は変わらない。PF/beam は完成 submission のコピーではなく、fold-safe な candidate path / likelihood / divergence features として取り込むのが安全。

1. `pilkwang/rogii-eda-target-free-alignment-for-tvt`
   - もっとも包括的な EDA とリーク境界整理。
   - strict drilling-time と offline batch の情報境界を明確に分けている。
   - task を「`TVT_input` 既知 prefix から hidden tail を予測する prefix-conditioned forecasting」と整理している。
   - anchor、trajectory、GR texture、typewell path、formation plane、row ANCC、beam/PF/DTW を同一の target-free signal family として整理。

2. `cdeotte/eda-starter`
   - 純 EDA として最も読みやすい。
   - task deck の重要点、データ構造、`TVT_input` の役割、TVT が増加/減少/flat になり得る点、well 単位 validation の必要性を短く確認できる。

3. `konbu17/rogii-plane-fit-formation-top-knn`
   - formation surface を使う強い物理仮説を提示。
   - notebook 記載の key insight: `TVT = -(Z - ANCC) + b_well` が well 内でほぼ完全線形、Pearson `-1.0000`、residual std `~0.007 ft`。
   - test に formation columns はないので、`(X, Y) -> formation top` を fold-safe に impute し、prefix から `b_well` を推定する。

4. Mitch の 2 本:
   - `mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore`
   - `mitchgansemer/gr-features-outlier-detection-rogii-wellbore`
   - 8.905 LB の解法 writeup と outlier analysis。
   - 最大の改善は target を raw TVT ではなく `TVT - last_anchor_tvt` にしたこと。
   - GR/NCC、formation KNN、beam/PF、estimator divergence、HGB blend まで、改善の因果が最も明確。

5. `svanikkolli/aeroridge-engine-v2`
   - EDA ではないが、DTW の扱いに関する重要な実験知見。
   - stochastic / PF snapshot を壊さず、deterministic DTW を add-only で足す設計が安全。

## 重要知見

### 1. 問題設定

- 予測対象は各 horizontal well の `TVT_input` が欠損する tail block。
- `submission.csv` は sample order の `id,tvt` のみ。visible test では 3 wells / 14,151 rows。
- train は 773 wells。Pilkwang notebook は train tail rows を 3,783,989 と記録している。
- 行独立の tabular 問題ではなく、well ごとの prefix-conditioned sequence forecasting として扱うべき。

### 2. Target は drift/residual が基本

Mitch writeup の改善表が強い根拠。

| 変更 | OOF RMSE |
| --- | ---: |
| `last_anchor_tvt` constant | 15.91 |
| raw TVT target tree | 19.50 |
| `TVT - last_anchor_tvt` + GR xcorr + formation KNN | 14.99 |
| + beam / particle filter | 13.96 |
| + 163 features / Optuna / NNLS | 10.01 |
| + v4 features / HGB | 9.85 |

raw TVT は 11,000-12,000 ft の per-well offset が支配的で、木モデルが信号より offset を学びに行く。`last_known_TVT` からの drift にすると、flat well を anchor で守りつつ drifting well を補正できる。

### 3. Validation は GroupKFold by well が必須

- same well の rows は GR fingerprint と trajectory が強く相関している。
- random row split は同じ well を train/val に混ぜるため leakage。
- 評価は hidden-tail rows のみで行う。
- formation imputer や nearby-well signal も fold ごとに validation well を除外して作る必要がある。

### 4. Feature の情報境界

Pilkwang notebook の整理が実装前の checklist として有用。

Allowed:

- row の `MD`, `X`, `Y`, `Z`, `GR`
- prefix の `TVT_input`
- typewell logs
- future GR / trajectory を使う offline batch features
- train formation columns から作る fold-safe spatial reference model

Excluded:

- hidden tail の true `TVT`
- direct train-only formation columns as test features
- validation wells を含めて fitted formation imputer
- `TVT_input_bfill`
- target-derived tail summaries

### 5. 強い signal family

- Anchor / prefix TVT:
  - `last_known_TVT` は強い null model。
  - prefix slopes, prefix range, recent slope, fade-in が効く。
- Typewell GR alignment:
  - horizontal GR を typewell GR に TVT 軸で合わせる。
  - Mitch は multi-scale NCC を主力にしている。DTW より amplitude-invariant な Pearson/NCC が安定しやすい。
- Formation / ANCC:
  - `TVT ~= -Z + formation_depth + b_well` が強い物理式。
  - direct formation columns は hidden test にないため、plane-fit KNN / row-level ANCC imputer 経由にする。
- Beam / PF / DTW:
  - standalone 予測器としては外すことがある。
  - tree model に入れる orthogonal drift signal、または disagreement / uncertainty feature として価値が高い。
- Estimator divergence:
  - NCC、formation、beam、PF が食い違う well は hard case。
  - divergence features は outlier well の検知に使える。

### 6. Postprocess は保守的に効く

- target は smooth だが monotonic ではない。
- slope clipping、residual shrinkage、prediction-start 近傍の fade-in、Savitzky-Golay smoothing が複数 notebook で出ている。
- Karnak notebook は postprocess search の注意点として、PF signals を 2 種類使う、SavGol parameters も Optuna objective に含める、per-fold PP RMSE を正しく見る、と整理している。

### 7. 主な failure mode

- GR missingness / noisy GR。
- typewell と horizontal GR の ambiguity。似た GR pattern が複数 formation に出る。
- large drift wells。Mitch outlier analysis では median per-well RMSE 6.2 ft、一方で 30 / 773 wells が 20 ft 超。
- long tail。小さな slope bias が tail 全体で蓄積する。
- public visible test の coordinate overlap に寄せた postprocess。Mitch は coordinate-overlap postprocess を generalizable ではないと明記。
- stochastic feature snapshot mismatch。AeroRidge は PF snapshot を変える full retrain で悪化し、deterministic DTW add-only に切り替えている。

## Notebook 別メモ

| Notebook | votes | 役割 | 読む価値 |
| --- | ---: | --- | --- |
| `pilkwang/rogii-eda-target-free-alignment-for-tvt` | 90 | EDA + leakage / feature policy | 最重要。情報境界、target-free alignment、signal map がまとまっている。 |
| `cdeotte/eda-starter` | 91 | pure EDA | 最初の把握に良い。task deck と validation の要点。 |
| `konbu17/rogii-plane-fit-formation-top-knn` | 53 | formation physics | ANCC / formation plane formula の核心。 |
| `mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore` | 81 | solution writeup | drift target, NCC, formation KNN, GroupKFold の因果説明。 |
| `mitchgansemer/gr-features-outlier-detection-rogii-wellbore` | 60 | GR features + outliers | hard wells と estimator divergence の理解に良い。 |
| `svanikkolli/aeroridge-engine-v2` | 92 | deterministic DTW | DTW を add-only で検証する実験設計が有用。 |
| `romantamrazov/rogii-super-solution-lb-top-3` | 160 | polished pipeline | Numba beam, multi-scale NCC, WLS `b_well`, uncertainty features。 |
| `romantamrazov/rogii-better-solution-lb-9-956` | 82 | beam/PF speedup | Python beam bottleneck、PF N=600、score-weighted NCC。 |
| `shinyanagai123/triple-signal-beam-search-dual-pf-lightgbm` | 56 | signal list | beam/PF/ANCC を簡潔に確認できる。 |
| `karnakbaevarthur/physics-informed-baseline` | 94 | postprocess improvement | PF signals と Optuna postprocess の改善観点。 |
| `cdeotte/xgb-starter-cv-15` | 127 | starter baseline | safe schema, residual target, GroupKFold の実装雛形。 |
| `cdeotte/nn-starter-cv-15-5` | 92 | NN baseline | TCN 方向の baseline。ただし Mitch は BiLSTM/TCN は OOF ~14.6 と記録。 |
| `ravaghi/wellbore-geology-prediction-hill-climbing` | 174 | runnable tree pipeline | 実装コピー元として有用。EDA は少ない。 |
| `nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` | 483 | DWT/DTW runnable | vote は最大だが markdown の知見は少なめ。 |
| `nina2025/rogii-h-blend-v1/v2` | 150 / 74 | public blend | final blend 参考。EDA / hidden-safe 実験の主軸にはしない。 |
| `needless090/score-10-081-score-lb-32-rank` | 56 | TabICL diversity | OOF leakage fix、TabICL diversity、runtime の参考。 |
| `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl` | 55 | artifact inference | PF/beam/formation/TabICL stack の整理。artifact 依存が強い。 |

## 次に試すなら

1. `exp002_drift_minimal`
   - `TVT - last_known_TVT` target。
   - prefix slopes, row geometry, GR rolling, simple typewell residual。
   - GroupKFold by well, hidden-tail rows only。

2. `exp003_formation_plane_ancc`
   - fold-safe `(X,Y) -> ANCC / formations` plane-fit KNN。
   - `tvt_formula = -Z + ANCC_hat + b_well_prefix` と uncertainty を追加。

3. `exp004_ncc_multiscale`
   - multi-scale NCC を add。
   - amplitude-invariant な Pearson/NCC を先に確認し、DTW は後回し。

4. `exp005_beam_pf_divergence`
   - beam/PF を standalone ではなく auxiliary features と divergence features として使う。

5. `exp006_hgb_nnls_blend`
   - XGB/CatBoost/HGB の OOF NNLS。
   - HGB は model diversity として価値がある可能性が高い。

6. `exp007_deterministic_dtw_addonly`
   - AeroRidge 方針で deterministic DTW だけを add-only。
   - stochastic DTW / PF snapshot mismatch を避ける。
