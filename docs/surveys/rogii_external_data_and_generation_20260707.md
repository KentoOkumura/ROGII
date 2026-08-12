---
title: ROGII external data and data generation survey
date: 2026-07-07
types:
  - survey
  - literature_review
  - comparison
experiments:
  - exp082
  - exp179
  - exp182
  - exp193
  - exp204
  - exp206
  - exp214
  - exp215
topics:
  - external_data
  - public_artifact
  - data_generation
  - hidden_safe
  - licensing
status: final
summary: "外部データ、公開生成物、コンペ内候補生成を比較し、hidden-safe性とライセンスを踏まえた利用優先度を整理した。"
---

# ROGII external data and data generation survey

- 対応する上位仮説: なし

調査日: 2026-07-07

## 結論

今回のコンペでスコア改善に使えそうな「外部」は、実測の追加地質データよりも、Kaggle 上で公開されている ROGII 用 artifact / model package と、コンペ内データから生成する候補 path / confidence feature が中心。

優先度順:

| 優先度 | アイデア | 期待値 | 状態 / 次アクション | 主リスク |
| ---: | --- | --- | --- | --- |
| 1 | public artifact / model package を ensemble route の比較基準・部品として使う | 既に `exp082` で Public LB 7.601。直接の外部入力としては最も強い | 新規調査より、hidden-safe source-port と dependency audit を継続 | artifact 依存、static public CSV、CV 因果が追いにくい |
| 2 | raw GR + typewell から PF/Beam/likelihood landscape を生成し、候補 confidence feature として使う | direct は弱いが oracle headroom と selector 材料はある | `exp214` raw scale control を固定し、selector / uncertainty feature に限定 | direct PF/Beam は過去に悪化、seed/runtime/reproducibility |
| 3 | CNN/SDF/MTP/MDN heatmap で topK path を生成し、selector の候補または confidence に使う | exp179/182 で real GR signal は確認済み。full-tail artifact が成立すれば再開価値あり | `exp204` は `exp215` full-tail artifact 待ち | fallback tail、worst-well、direct replacement の破綻 |
| 4 | typewell late-range / formation parallel prior を context feature として使う | `exp193` は CV 改善、Public LB 小幅改善。hard window / direct correction は危険 | context-only / candidate prior として限定採用 | early-range exception、PF_Z regression、hard invalid |
| 5 | 一般公開 well-log datasets で GR encoder / denoising / missingness pretraining | 直接 TVT には効きにくいが、低リスクな事前学習候補 | まずは no-training feature readout か小さな encoder smoke | 地域・目的変数・depth reference が違う。ライセンス注意 |

採用しない / 低優先:

- Texas / Eagle Ford / Austin Chalk / Buda の外部実測データを探して hidden well に照合する方向。hidden well は hash ID で、X/Y も匿名化に近く、同一井照合や地域 surface の外部補間は現実的ではない。
- Volve 系 dataset。Kaggle copy は `CC-BY-NC-SA-4.0` と表示され、商用利用を妨げない外部コード/データ要件と相性が悪い。
- static public submission CSV。hidden rerun で ID が変わる code competition では壊れやすい。
- dz/dTVT `b` peak cluster を単独 baseline にする方向。`exp206` で Public LB 29.193、CV 35.300 までしか行かず、要件未達。

## 公式制約

確認元:

- Kaggle competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- Rules page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
- Data page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data
- Evaluation page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation

Kaggle CLI `competitions pages` で確認した要点:

- Submit は Notebook-only。
- CPU/GPU runtime は 9 hours 以下。
- Internet access disabled。
- Freely and publicly available external data is allowed, including pre-trained models。
- ルール上、外部データ / 外部モデルは許可。ただし無料で全参加者が同等にアクセスできる、または合理的にアクセス可能であることが必要。
- Open source code を使う場合は OSI-approved license で、commercial use を制限しないことが必要。
- validation/test record に対する hand labeling / human prediction は禁止。

公式 data-description の要点:

- train horizontal: `MD`, `X`, `Y`, `Z`, train-only formation columns, `TVT`, `GR`, `TVT_input`。
- test horizontal: trajectory + `GR` + `TVT_input`。formation columns は Training only。
- test は約 200 wells。ローカルに見える 3 wells は authoring 用 example で、hidden run では差し替え。
- metric は RMSE。

## Kaggle 内の外部 artifact / model package

保存済み public notebook metadata から、`dataset_sources` を集計した。主要 dataset は以下。

| Dataset | 中身 | 利用状況 / 評価 |
| --- | --- | --- |
| `ravaghi/wellbore-geology-prediction-artifacts` | `data/train.csv` 約 7.4GB、LightGBM/CatBoost pickles | 148 notebooks が参照。Ravaghi/DWT/hill-climbing family の中核。大きく、因果追跡は重い |
| `fleongg/rogii-claude-models-pub` | `features.json`, `lgb0.pkl`, `lgb1.pkl`, `lgb2.pkl` | 88 notebooks が参照。fle3n / dual-pipeline 系で強い。`exp082` source-port の一部 |
| `phongnguyn23021656/koolbox-offline` | `koolbox`, `optuna`, `scikit_learn`, `scipy` など wheel | 87 notebooks が参照。internet disabled 対策の依存 wheel 集 |
| `needless090/rogii-tabicl-mirror` | TabICL wheel と checkpoint | TabICL stack 用。GPU / environment / artifact contract に注意 |
| `thbdh5765/rogii-v10-fresh-artifacts` | CatBoost seeds, diagnostics OOF/test predictions | Hoang v10 / TabICL stack 系。static prediction risk がある |
| `pilkwang/rogii-model-package` | feature builders, feature columns, CatBoost/HGB/LGB models, manifest | model-package sidecar / geosteering rebuild 系。feature builder と manifest は参考価値が高い |
| `nina2025/rogii-03` | `9.537.csv`, `9.765.csv`, `9.956.csv` など public CSV | h-blend 用。static public CSV なので hidden-safe ではない |
| `hengck23/hengck23-rogii-cnn-mtp-demo` | CNN/MTP `.pth`, `run_train_sdf.py`, `meta_df.typewell.csv` | MTP/SDF 実装参考として有用。重みそのものの採用は再現性と契約確認が必要 |
| `medali1992/rogii-cnn-mtp-weights` | SDF/MTP checkpoint 多数 | learned heatmap route の参考。直接 submit 部品にはしない |

判断:

- スコアだけなら public artifact route が最強。ただしこの repo ではすでに `exp079` / `exp081` / `exp082` で audit 済みで、`exp082_public_artifact_replay_followup` が Public LB 7.601 の ensemble route anchor。
- 新規に別 artifact を足すなら、まず source-port 可能か、hidden test で生成される current sample に対して実行できるか、static CSV 依存ではないかを確認する。
- ML route の実験因果を追う目的では、artifact を丸ごと積むより、feature builder / confidence feature / candidate set に切り分ける方がよい。

## 一般公開 well-log dataset

Kaggle datasets を検索し、代表ファイルの列を確認した。

| Dataset | License 表示 | 中身の確認 | このコンペへの評価 |
| --- | --- | --- | --- |
| `faresazzam/well-logs-dataset-for-machine-learning` | MIT | FORCE 2020 系。列は `WELL`, `DEPTH_MD`, `X_LOC`, `Y_LOC`, `Z_LOC`, `GROUP`, `FORMATION`, `GR`, `NPHI`, `DTC`, lithofacies など | GR encoder / facies pretraining には使えるが、TVT / typewell-horizontal 対応はない |
| `mahdialfred/multiple-well-logging-datasets` | Kaggle page 要確認。内容は上と同系 | FORCE 2020 系 CSV | 上と同じ。重複候補 |
| `ainiashiqinn/barber-county-well-log-dataset-kansas` | CC0-1.0 | metadata は API, lat/lon, depth, `GR_present`, resistivity, RHOB, NPHI flags。main CSV は約 517MB | GR missingness / log statistics pretraining くらい。地域が Kansas で ROGII とは遠い |
| `imranulhaquenoor/volve-dataset-well-f-9-a` | CC-BY-NC-SA-4.0 | depth/time CSV。MWD raw GR, corrected GR, inclination, azimuth, TVD, drilling params | ライセンスが non-commercial。competition submission には避ける |
| `yohanesnuwara/north-sea-petroleum-data` | MIT | FORCE2020, DLIS, core pore/perm/lithology data。小 CSV は core/pore-perm 主体 | ROGII の TVT には直接結びつかない。pretraining 低優先 |

判断:

- 直接的な外部実測データとしては弱い。理由は、ROGII の核心が「各 test well の typewell GR と horizontal GR / trajectory / known TVT_input prefix から hidden tail の TVT を推定する」問題であり、一般 well-log はこの対応関係を持たないため。
- 使うなら、GR 波形 denoising、欠損補完、normalization、sequence encoder pretraining、facies/log morphology の self-supervised pretraining に限定する。
- ただし現行 `exp148` 系の tabular add-only feature では、既に learned GR matcher (`exp180`) が小幅悪化している。外部 pretraining をやるなら selector route か heatmap route の小さな smoke に限定する。

## データ生成手法

### 1. PF / Beam / likelihood landscape

出典:

- Discussion 699853: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853
- Discussion 721549: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/721549
- exp214: `experiments/exp214_public_raw_gr_residual_scale_control/README.md`

有効性:

- raw GR + typewell GR + known prefix residual scale の public-like control は、`exp214` で 64 wells diagnostic を作成。best non-oracle `pf_raw_scale_12` RMSE 15.223857、oracle diagnostic RMSE 11.104328。
- direct path では現行 ML anchor に遠いが、path confidence、scale agreement、seed dispersion、topK gap は selector feature として使える。

採用方針:

- direct replacement / PF weight replacement ではなく、candidate confidence / selector feature に限定。
- raw control を固定して、denoise / affine / structural prior が本当に上回るかを見る。

### 2. CNN/SDF/MTP/MDN heatmap path generator

出典:

- Discussion 699853
- Alyaev & Elsheikh, Direct multi-modal inversion of geophysical logs using deep learning: https://arxiv.org/abs/2201.01871
- `hengck23/hengck23-rogii-cnn-mtp-demo`
- exp179/182/184/203/212

有効性:

- 論文は gamma-ray log inversion を non-unique / multi-modal として扱い、MDN + MTP で複数 trajectory と確率を出す。ROGII の datum ambiguity / topK candidate 問題に合う。
- `exp179` / `exp182` で real GR は shuffled/no-GR control を明確に上回った。exp182 full-fold base は top3 within10 0.500000、top10 0.808908。
- `exp184` は heatmap path features を selector に入れる方向を支持したが、direct TVT replacement はしない方針。
- `exp203` は heatmap MDN feature-only で exp184 を更新できず。
- `exp212` は full-grid artifact contract は成立したが source coverage 43.0%、fallback 57.0%、stitched-only top5 RMSE 50.085 と弱い。existing + stitched oracle は改善するが、fallback tail が直線化する。

採用方針:

- exp215 で full-tail learned path artifact が成立した場合だけ、`exp204` の selector candidate route を再開。
- direct replacement / softmax average / PF weight replacement はしない。

### 3. Typewell late-range / formation parallel prior

出典:

- Discussion 708167: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167
- exp176/186/191/192/193/196/199

有効性:

- Discussion 708167 は train-only formation columns が typewell geology intervals + parallel formation assumption から派生している可能性を整理。6 formation columns は実質 1 base surface + constant offsets。
- `exp193` の `tlic_` context-only add-only features は exp148 GPU CV 8.501281182 から 8.456665439 へ改善し、Public LB 7.946。exp148 CPU runtime 7.921 には届かないが、signal はある。
- `exp192` pct50 hard-window full cache は `likpf_mean` と `pf_ancc` を改善したが、`pf_z` と early-range bucket が壊れた。`exp196` pct40 は sensitivity として改善/悪化が混在。
- `exp186` soft prior は strongest `likpf_mean` を大きく悪化させた。

採用方針:

- context-only / confidence-only / candidate score feature として使う。
- hard invalid、hard clip、typewell range window の direct generation は避ける。

### 4. Known-prefix supervised GR window matcher

出典:

- exp178/180

有効性:

- exp178 では known `TVT_input` prefix だけから pair dataset を作り、real GR logistic pair AUC 0.765、shuffled 0.662、expected-error top5 within10 coverage 0.960。
- しかし exp180 で exp148 add-only feature にすると `lgb_mean` 8.514526、exp148 8.501281 から悪化。

採用方針:

- exp148 global add-only は不採用。
- PF/Beam selector 側の candidate confidence / expected-error / topK pruning なら再検討余地あり。

### 5. dz/dTVT offset / B-peak cluster

出典:

- Discussion 711308: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308
- exp206

有効性:

- `dTVT ~= a*dZ+b` と `b` peak cluster は説明変数として面白い。
- ただし `exp206` は v3 で Public LB 29.193 / CV 35.300。discussion の LB 約 12.8 には届かなかった。

採用方針:

- standalone baseline / direct submit は不採用。
- 使うとしても offset-well confidence tag、diagnostic readout に限定。

### 6. Synthetic geology / GAN / flow matching

出典:

- Alyaev et al., Probabilistic forecasting for geosteering in fluvial successions using a GAN: https://arxiv.org/abs/2207.01374
- Ghyselincks et al., Synthetic Geology: Structural Geology Meets Deep Learning: https://arxiv.org/abs/2506.11164

有効性:

- GAN geosteering paper は 2D geological sections を生成し、forward model で log predictions に変換し、data assimilation で ahead-of-bit uncertainty を下げる発想。
- SyntheticGeo は 3D lithology model を大量生成して generative model prior にする発想。
- ROGII には train 773 wells と hidden per-well typewell があり、完全な外部 synthetic geology foundation model より、competition-specific target-free generator の方が現実的。

採用方針:

- 今回の提出期限内では低優先。
- ただし小さな synthetic は有用: typewell GR から horizontal GR を forward simulate し、noise/stretch/offset を入れて CNN/MTP や matcher の pretraining に使う。

## 未着手の現実的候補

1. `exp214` raw PF scale/seed landscape を selector confidence feature にする。
   - 仮説: direct PF は弱いが、scale agreement / seed spread / oracle gap proxy は exp158/184 selector の不確実性を改善する。
   - 検証: exp184 selector候補は固定、`rawpf_` confidence features だけ add-only。exp184 best 10.560650325 を超えるか、worst-well / sparse GR bucket を見る。

2. 外部 FORCE/Barber GR で小さな GR denoising/self-supervised encoder smoke。
   - 仮説: 一般 well-log の GR morphology を使って、ROGII known-prefix GR matcher の feature extraction を安定化できる。
   - 検証: ROGII train だけの encoder vs external-pretrained encoder を exp178 pair split で比較。AUC と shuffled/no-GR margin が改善しなければ終了。
   - 注意: exp148 add-only に直行しない。

3. typewell context-only feature の軽い second pass。
   - 仮説: `exp193` は CV/LB 小幅 positive なので、hard prior ではなく exact-prune後の exp198 schema に `tlic_` を再評価すると競合が少ない可能性。
   - 検証: parent/control 再学習なし、1 active variant、15 boosters。exp148 CPU runtime anchor 7.921 を超える見込みが薄い場合は CV-only で止める。

4. full-tail MTP artifact が成立した場合の selector candidate route。
   - 仮説: exp212 は fallback-heavy だが、source-supported topK には oracle headroomがある。full-tail generation が成立すれば exp204 を再開できる。
   - 検証: direct candidate RMSE ではなく existing+new oracle、selector selected rate、sparse/fallback bucket、path switch。

## 調査コマンドメモ

- `kaggle competitions pages rogii-wellbore-geology-prediction --content --page-name rules`
- `kaggle competitions pages rogii-wellbore-geology-prediction --content --page-name data-description`
- `kaggle competitions pages rogii-wellbore-geology-prediction --content --page-name "Code Requirements"`
- `kaggle competitions topics list rogii-wellbore-geology-prediction -s top -v`
- `kaggle competitions topics show rogii-wellbore-geology-prediction 699853`
- `kaggle competitions topics show rogii-wellbore-geology-prediction 702474`
- `kaggle competitions topics show rogii-wellbore-geology-prediction 708167`
- `kaggle competitions topics show rogii-wellbore-geology-prediction 711308`
- `kaggle datasets files <dataset>`
- `kaggle datasets list -s "well logs gamma ray" --csv`
