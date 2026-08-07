# ROGII public notebooks solution summary

取得日: 2026-05-27

更新:

- 2026-06-03 に `scoreAscending` 上位 20 件を再取得し、`docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/` に保存した。
- 最新差分は `docs/notebooks/rogii-wellbore-geology-prediction/latest_public_notebooks_20260603.md` を参照。

対象: Kaggle public notebooks を `scoreAscending` で並べた上位 20 件。

取得コマンド:

```bash
kaggle kernels list --competition rogii-wellbore-geology-prediction --sort-by scoreAscending --page-size 20 -v
```

注意:

- このコンペの metric は RMSE なので、Kaggle API の `scoreAscending` は小さい順、つまり良い順として扱う。
- Kaggle CLI / Python API の一覧レスポンスには数値スコア列が含まれなかった。したがってこのメモの順序は API の `scoreAscending` の返却順、スコア値は notebook title / markdown に明記されていたものだけを記録する。
- `kernel_listing_score_ascending.csv` は同じディレクトリに保存済み。

## まず読むべき結論

現時点で一番価値が高いのは Mitch の 8.905 系 writeup。単なるブレンドではなく、target を `TVT - last_anchor_tvt` の drift に変える、GR の NCC、formation KNN、PF / beam、DWT / estimator divergence、HGB を含む NNLS blend まで、改善の因果が一番よく説明されている。

2026-06-03 の最新 `scoreAscending` では、title 上 `LB 8.860/8.863` の PF/physical sel15 family と Sunny physical + 生成物を使った積み上げ が前面に出ている。ただしこれらは public visible branch、artifact dataset、GPU / rerun stability、hidden runtime の影響が強い。実験方針としては、Mitch 系の因果が明確な drift / NCC / formation / ensemble を軸にしつつ、PF/beam family の scale別予測、likelihood、GR interpolation、候補間の食い違い を add-only で検証する。

次に実装候補として価値が高いのは Ravaghi / DWT / AeroRidge 系。LightGBM + CatBoost、GroupKFold、PF / beam / NCC / formation KNN、hill climbing、Optuna postprocess という再現しやすいパイプラインになっている。AeroRidge は stochastic DTW を避け、deterministic DTW だけを追加する方針が特に参考になる。

h-blend / sidecar 系は public LB を押し上げる探索としては有効だが、static public CSV と hidden sample の id mismatch を強く意識している。実験の核にするより、最後の提出候補として扱うのが安全。

## Score-Ascending 上位 20 件

| 順位 | Notebook | 既知スコア | 種類 | 要点 |
| ---: | --- | --- | --- | --- |
| 1 | [GR Features / Outlier Detection](https://www.kaggle.com/code/mitchgansemer/gr-features-outlier-detection-rogii-wellbore) | LB 8.905 | writeup + inference | GR feature / outlier analysis 付きの 8.905 系。184 features、XGB + CB + HGB NNLS。 |
| 2 | [Drift Targeting + NCC](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore) | LB 8.905 | writeup + inference | 同じ 8.905 系の解法説明。drift target と NCC の説明が最も読みやすい。 |
| 3 | [9.251 DWT-based](https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based) | title 9.251 | runnable fork | DWT / DTW、PF / beam、LGB + CB、hill climbing、Optuna postprocess。 |
| 4 | [h-blend v1](https://www.kaggle.com/code/nina2025/rogii-h-blend-v1) | table 9.409 for earlier v1 | blend | 9.537 / 9.765 / 9.956 の rank-aware blend。current code は 0.81/0.15/0.04。 |
| 5 | [Inference Stack with PF, Beam and TabICL](https://www.kaggle.com/code/kojimar/rogii-inference-stack-with-pf-beam-and-tabicl) | not exposed | artifact inference | Hoang v10 artifacts + TabICL dominated stack。PF / beam / formation / dense ANCC features。 |
| 6 | [rogii-0525-nannicha-dwt-hc](https://www.kaggle.com/code/beicicc/rogii-0525-nannicha-dwt-hc) | not exposed | DWT / HC fork | Ravaghi artifact route の DWT + hill climbing 系 candidate。 |
| 7 | [AeroRidge Engine v2](https://www.kaggle.com/code/svanikkolli/aeroridge-engine-v2) | expected 9.18-9.25 | DTW extension | cached LGB+CB に deterministic DTW 21 cols と LGB-DTW を追加。 |
| 8 | [Wellbore Geology Prediction / Hill Climbing](https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing) | not exposed | base runnable route | LGB + CB + hill climbing + Optuna。多くの fork の元になっている。 |
| 9 | [H-Blend + Model-Package Sidecar](https://www.kaggle.com/code/pilkwang/rogii-h-blend-model-package-sidecar) | not exposed | blend + sidecar | h-blend に OOF-weighted sidecar を 0.5-2% 足す設計。見えない test well 用の代替処理 を持つ。 |
| 10 | [Ultra Sub-9 RMSE](https://www.kaggle.com/code/raunakdey07/rogii-ultra-sub-9-rmse) | title/heading 9.251 | DTW / Optuna | multi-scale DTW、stochastic DTW、PF / beam、LGB + CB + XGB、Optuna postprocess。 |
| 11 | [rogii-0525-yoshi74-exact080](https://www.kaggle.com/code/beicicc/rogii-0525-yoshi74-exact080) | not exposed | tree-only fork | CNN artifacts 不可のため tree/DWT fallback。 |
| 12 | [h-blend v2 fork](https://www.kaggle.com/code/han21400/rogii-h-blend-v2) | table v4 9.530 | blend | Nina h-blend v2 の fork。current code は 0.90/0.08/0.02。 |
| 13 | [TreeOnly Blend 63/67](https://www.kaggle.com/code/yoshimurakoei/rogii-70-ulas-treeonly-blend-63-67-gc2) | not exposed | seed blend | seed 777 と 271828 の tree-only fallback を 0.65/0.35 blend。 |
| 14 | [h-blend v2](https://www.kaggle.com/code/nina2025/rogii-h-blend-v2) | table v4 9.530 | blend | 12 と同一系統。current code は 0.90/0.08/0.02。 |
| 15 | [v10 Fresh Artifact Infer](https://www.kaggle.com/code/thbdh5765/rogii-v10-fresh-artifact-infer) | referenced 9.537 | artifact inference | LGB + CB + TabICL stack、postprocess、exact coordinate overlap optional。 |
| 16 | [9.538 Training](https://www.kaggle.com/code/rauffauzanrambe/9-538-rogii-wellbore-geologyprediction-training) | title 9.538 | artifact route | Ravaghi artifacts を使う LGB + CB / hill climbing route。 |
| 17 | [DWT tau22 sgw25 candidate](https://www.kaggle.com/code/beicicc/rogii-dwt-tau22-sgw25-candidate) | markdown 9.674 for related route | DWT fork | score honesty / validation 重視の DWT artifact ensemble fork。 |
| 18 | [DWT sgw25 candidate](https://www.kaggle.com/code/beicicc/rogii-dwt-sgw25-candidate) | not exposed | DWT fork | 17 と同じ説明文の sibling candidate。 |
| 19 | [DWT fixed tau25 wpf007 02](https://www.kaggle.com/code/beicicc/rogii-dwt-fixed-tau25-wpf007-02) | not exposed | DWT fork | 17 と同じ DWT artifact ensemble 系。 |
| 20 | [TreeOnly 70 seed final](https://www.kaggle.com/code/yoshimurakoei/rogii-72-ulas-treeonly-70-seed271828-final) | not exposed | seed blend | 13 と同じ seed-block tree-only family。 |

## 解法ファミリー別の整理

### 1. Mitch 8.905: drift target + GR/NCC + formation + ensemble

該当: 1, 2

核:

- target を raw TVT ではなく `drift = TVT - last_anchor_tvt` にする。
- baseline は `last_anchor_tvt` 一定で OOF 15.91 ft。raw TVT target は 19.50 ft で悪化。drift target + GR xcorr + formation KNN で 14.99 ft。
- GR を stratigraphic barcode とみなし、typewell GR と horizontal GR の normalized cross-correlation を multi-scale に計算する。
- formation top columns は test にないので、train wells の (X, Y) 近傍から formation plane を推定し、anchor zone で per-well bias を合わせる。
- Viterbi beam / particle filter を standalone predictor ではなく、tree model の補助特徴として使う。
- v4 では estimator divergence、short-window slopes、DWT GR を追加。
- 最終は XGBoost + CatBoost + HistGradientBoosting の NNLS blend。最終重みは XGB 20.8%、CB 43.2%、HGB 36.0% と記載。
- postprocess は Savitzky-Golay smoothing window 17, order 3。
- validation は GroupKFold(5) by well_id。OOF 9.85 ft、LB 8.905 ft。

転用方針:

- まず exp002 として `last_anchor_tvt` drift target + GroupKFold + simple tabular model を作る。
- 次に NCC / formation KNN / PF or beam を段階追加し、OOF で 1 変更ずつ見る。
- HGB は LightGBM より多様性が出る可能性があるため、CatBoost と合わせて OOF blend を見る価値が高い。

リスク:

- pre-trained artifacts 前提の notebook なので、そのまま提出 kernel にするには artifact dataset が必要。
- formation top は test にないため、直接特徴として使わず KNN / plane imputation 経由にする。
- coordinate overlap postprocess は見かけの public test には効いても hidden では一般化しない、と notebook 側も否定している。

### 2. Ravaghi / DWT / hill-climbing: reproducible tree pipeline

該当: 3, 6, 8, 10, 16, 17, 18, 19

核:

- `ravaghi/wellbore-geology-prediction-artifacts` に feature table / saved models / OOF を置き、Kaggle 上で再利用する。
- GroupKFold(5) by well、target は drift delta。
- feature は PF / beam / multi-scale NCC / formation plane KNN / dense ANCC / anchor extrapolation / GR rolling が中心。
- LightGBM 3 seed + CatBoost 3 seed を基本に、fork によって XGBoost、DWT、DTW、stochastic DTW を追加。
- ensemble は hill climbing。Ravaghi base は `allow_negative_weights=True`、DWT-based の一部は false。
- postprocess は `alpha`, `tau`, `w_pf` を Optuna で最適化し、`last_known_tvt + postprocessed_delta` に戻す。SG smoothing を使う fork と外す forkがある。

転用方針:

- 実装の出発点としてはこの family が一番コピーしやすい。
- ただし artifact を使うだけでは自分の CV 確認にならないため、まず local feature builder を再現し、small subset で `id,tvt` alignment と GroupKFold を確認する。
- DWT / DTW は Mitch writeup では OOF 悪化の記載がある一方、AeroRidge は deterministic DTW だけで改善余地を見ている。入れるなら stochastic ではなく deterministic DTW から。

リスク:

- artifact route は実行できても、どの fold / feature / postprocess が効いているかが見えにくい。
- stochastic PF / DTW は feature snapshot の再現性を崩しやすい。AeroRidge の note でも v36 が random PF snapshot mismatch で悪化したと書かれている。

### 3. AeroRidge deterministic DTW

該当: 7

核:

- cached LGB + CB の feature snapshot は変更しない。
- 21 個の deterministic DTW features だけを追加し、LGB-DTW x3 を追加で学習する。
- DTW は Sakoe-Chiba band の multi-scale DP。stochastic DTW は使わない。
- cached LGB/CB は元の cached feature columns だけで予測し、新しい LGB-DTW は widened feature set を見る。
- hill climbing で cached models と LGB-DTW を blend。

転用方針:

- ROGII の次実験としてかなり現実的。PF/beam/NCC の base が固まった後に、deterministic DTW features を add-only で入れる。
- add-only にして既存 OOF と alignment を壊さないのがポイント。

リスク:

- runtime は notebook 上で 2xT4 55-65 min と記載。CPU local では重い。
- expected LB は notebook の見積もりで、API から actual score は取れていない。

### 4. Hoang v10 / TabICL 生成物を使った積み上げ

該当: 5, 15

核:

- feature は PF / beam / multi-scale self-correlation / formation-plane / dense ANCC / GR rolling / trajectory。
- LightGBM + CatBoost + TabICL を stacking。Kojimar notebook は saved linear stack が TabICL dominated と明記。
- `ROGII v10 Fresh Artifacts` と `TabICL mirror` が必要。
- `ROGII_INFERENCE_ONLY` を使い、Kaggle 側では基本 inference に寄せる。
- exact coordinate overlap blend は optional。

転用方針:

- TabICL は強い public route として観察価値があるが、まず自前 baseline を作るまでは主軸にしない。
- hidden / offline reproducibility を考えるなら、TabICL context と artifacts の入力契約を明確にする必要がある。

リスク:

- GPU / TabICL の environment 差でスコアが揺れる可能性が notebook に明記されている。
- saved artifacts 依存が強く、改善実験の因果が追いにくい。

### 5. h-blend / sidecar

該当: 4, 9, 12, 14

核:

- h-blend は単純平均ではなく、行ごとに予測値を sort し、順位に応じて base weight に correction を足す。
- h-blend v1 current code:
  - base: 9.537 / 9.765 / 9.956
  - weights: 0.81 / 0.15 / 0.04
  - rank correction: +0.10 / -0.03 / -0.07
  - asc/desc: 0.30 / 0.70
- h-blend v2 current code:
  - base: 9.537 / 9.765 / 9.956
  - weights: 0.90 / 0.08 / 0.02
  - rank correction: +0.031 / -0.014 / -0.017
  - asc/desc: 0.30 / 0.70
- sidecar notebook は h-blend anchor に OOF-weighted model package prediction を late-linear 0.5-2% 程度で足す。id mismatch 時は original 見えない test well 用の代替処理 に切り替える設計。

転用方針:

- 自前モデルが複数できた後に、rank-aware blend を OOF 上で再現してみる。
- public-only CSV 依存の h-blend は hidden sample で壊れやすいので、提出候補にするなら全 member が current sample に対して再実行できる形にする。

リスク:

- static public CSV は hidden sample の ids と一致しない。
- 見えない test well 用の代替処理 では sidecar blend を無効化する設計になっており、public score と hidden rerun の実体が違う可能性がある。

### 6. tree-only / CNN-disabled seed blends

該当: 11, 13, 20

核:

- original は tree/DWT predictions と CNN artifacts の blend だったが、CNN artifacts が attach できないため tree-only fallback に固定。
- Yoshimura notebook は seed 777 weight 0.65、seed 271828 weight 0.35 の final blend。
- LGB + CB + hill climbing + Optuna postprocess という基本形は Ravaghi/DWT family と同じ。

転用方針:

- artifact が欠けた状態でも valid submission を出す設計、seed blend の整理として参考になる。
- スコア改善の本命ではなく、提出安定性・代替処理設計の参考にする。

## 実験優先順位

1. `exp002_drift_ncc_baseline`
   - drift target、GroupKFold by well、anchor / GR rolling / simple NCC / formation KNN の最小セット。
   - 目的: Mitch 8.905 系の最大ジャンプ R3 をローカルで検証する。

2. `exp003_pf_beam_hgb_blend`
   - PF / beam features と HGB を追加し、CatBoost / XGB or LightGBM と OOF NNLS blend。
   - 目的: 13.9 -> 10.x に効いている orthogonal estimators と model diversity を確認する。

3. `exp004_deterministic_dtw_addonly`
   - AeroRidge 方式で deterministic DTW features を add-only 追加。
   - 目的: stochastic feature snapshot を壊さず、DTW の寄与だけを測る。

4. `exp005_rank_aware_blend`
   - 自前の複数 submission に h-blend 風の rank correction を OOF 上で最適化。
   - 目的: public-only CSV に依存せず、hidden-safe な blend だけを検証する。

## 読む順番

1. Mitch `Drift Targeting + NCC`
2. Mitch `GR Features / Outlier Detection`
3. Ravaghi `Wellbore Geology Prediction | Hill Climbing`
4. AeroRidge Engine v2
5. Hoang v10 Fresh Artifact Infer
6. Nina h-blend v1/v2
7. Pilkwang sidecar

この順番なら、まず解法の因果を理解し、その後で artifact route と blend route の使いどころを判断できる。
