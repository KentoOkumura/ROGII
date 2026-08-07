# ROGII latest public notebook update

調査日: 2026-06-03

対象:

- Kaggle public notebooks の `voteCount` 上位 20 件を再取得。
- Kaggle public notebooks の `scoreAscending` 上位 20 件を `score_ascending_latest/` に新規取得。
- Kaggle public notebooks の `dateRun` 上位 20 件を `date_run_recent/` に新規取得。

取得コマンド:

```bash
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/vote_top --sort-by voteCount --force
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest --sort-by scoreAscending --force
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent --sort-by dateRun --force --retries 3
```

保存先:

- `docs/notebooks/rogii-wellbore-geology-prediction/vote_top/kernel_listing.csv`
- `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/kernel_listing.csv`
- `docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent/kernel_listing.csv`

注意:

- Kaggle API の notebook listing には public score 列が出ない。score 値は title、markdown、notebook 内の記録に明記されたものだけを使う。
- `dateRun` 上位は直近 fork / probe が多く、0 vote の notebook も多い。読む優先度は `scoreAscending`、高 vote、明示的な validation note の順に置く。
- public visible test は train 由来の 3 wells なので、visible overlap や static CSV blend を hidden-safe な手法として扱わない。

## 重要更新

1. public score 順の先頭は、Mitch 8.905 系からさらに進み、`LB 8.860` / `8.863` を title に掲げる PF/physical 系 fork が出ている。
2. 上位 family は、Aiden / Needless / Safar の `sel15` PF ensemble、Sunny physical model、PF/beam/TabICL 生成物を使った積み上げ、AeroRidge v34 fork に寄っている。
3. 直近 notebook は 2026-06-02 から 2026-06-03 にかけて、Aiden PF scale selector、Needless sel15 256 seeds、Yarukikun golden LGB/Cat、AeroRidge v34 の fork / rerun が多い。
4. 実験に転用するなら、public score の高い完成 submission を直接追うより、PF/beam の candidate path、likelihood、scale selector、GR interpolation、per-well sanity check を fold-safe な特徴・診断として取り込むのが安全。

## Score-Ascending 最新上位

| API 順 | Notebook | listing votes | 既知スコア/記録 | 要点 |
| ---: | --- | ---: | --- | --- |
| 1 | `kojimar/rogii-physical-pf-signal-meets-artifact-stack` | 18 | title score なし | Sunny physical / PF 80% + v10 生成物を使った積み上げ 20%。hidden test 上で両 component を同一 notebook 内実行する設計。 |
| 2 | `needless090/lb-8-860-rogii-sel15-256seeds` | 11 | title `LB 8.860` | hidden は PF ensemble only。GR interpolation、256 seeds、scale selector、beam / hold blend を持つ。 |
| 3 | `safar1/lb-score-8-863` | 18 | title `8.863` | Needless/Aiden 系に近い PF scale selector route。 |
| 4 | `aidensong123/rogii-sel15-rerun` | 112 | score title なし | hidden は 128 seeds の likelihood-weighted PF。visible train wells は physical model branch。 |
| 5 | `ajayrao43/rogii-wellbore-geology-prediction` | 29 | score title なし | PF/beam 系 fork の source family の一つ。 |
| 10 | `beicicc/rogii-0602-aiden-pf-scale3` | 2 | score title なし | Aiden PF scale fork。dateRun 側にも継続的な scale variants がある。 |
| 11 | `yaroslavkholmirzayev/reproduce-strongest-reference-aeroridge-v34` | 33 | route reference | AeroRidge v34 の再現 route。artifact input は `ravaghi/wellbore-geology-prediction-artifacts`。 |
| 12 | `sunnywu27/rogii-wellbore-tvt-physical-model` | 98 | score title なし | physical / PF signal の主要 source。 |
| 13 | `jamienojek/rogii-sunny-physical-pf-v12` | 18 | score title なし | Sunny PF の rerun / fork。 |
| 16-20 | Aiden / Beicicc PF params / AeroRidge weights | 1-4 | score title なし | scale、PF params、AeroRidge weight の public probe。 |

## DateRun 最新上位

| API 順 | Notebook | lastRunTime | 要点 |
| ---: | --- | --- | --- |
| 1 | `beicicc/rogii-0603-yarukikun-golden-lgbcat` | 2026-06-03 04:35 | Golden 6 features + LightGBM/CatBoost + Savitzky-Golay。score 実績は未確認。 |
| 2-3 | `beicicc/rogii-0603-aiden-pf-scale2/scale25` | 2026-06-03 04:31 | Aiden PF scale selector の追加 probe。 |
| 5 | `beicicc/rogii-0603-needless-sel15-256seeds` | 2026-06-03 04:21 | Needless 256 seeds の rerun / fork。 |
| 8 | `pilkwang/rogii-eda-target-free-alignment-for-tvt` | 2026-06-03 01:31 | 高品質 EDA が更新済み。情報境界 checklist として引き続き最重要。 |
| 17 | `afr1ste/rogii-pf-beam-tabicl-stack-guide-9-150` | 2026-06-02 17:29 | PF/beam/TabICL route の validation / submission hygiene note。verified public RMSE 9.150 の account pin を記録。 |

## 手法ファミリー別メモ

### PF / physical sel15 family

該当:

- `aidensong123/rogii-sel15-rerun`
- `needless090/lb-8-860-rogii-sel15-256seeds`
- `safar1/lb-score-8-863`
- `beicicc/rogii-0602-aiden-pf-*`
- `beicicc/rogii-0603-aiden-pf-*`

核:

- hidden test wells は physical model ではなく PF ensemble を主力にする。
- typewell GR と horizontal GR を合わせる Particle Filter を、likelihood weighted multi-seed ensemble にする。
- GR は prediction zone の NaN gap を interpolation してから PF に渡す。
- scale selector は evaluation length、Z span などで `pf_scale_3/5/8/12`、beam weight、hold-last-known weight を切り替える。
- visible train wells branch では physical formation contacts が非常に強いが、hidden test では formation columns / Geology がないため、そのまま使えない。

転用方針:

- まず PF / beam を standalone submission として追うより、OOF 上で candidate path、likelihood、scale別予測、hold-weighted drift、divergence を特徴にする。
- stochastic PF は seed 数で安定化しても実行時間と再現性の問題が出る。`seed`, `n_particles`, `n_seeds`, `scale`, `GR interpolation` を config 化し、fold-safe feature snapshot を残す。
- public 3 wells の physical branch は hidden-safe ではない。train fold 内で validation well を除外した formation/contact imputer として再構成する。

### Sunny physical + 生成物を使った積み上げ

該当:

- `kojimar/rogii-physical-pf-signal-meets-artifact-stack`
- `sunnywu27/rogii-wellbore-tvt-physical-model`
- `thbdh5765/rogii-v10-fresh-artifact-infer`

核:

- Sunny physical / PF prediction を anchor にし、v10 生成物を使った積み上げ を小さく混ぜる。
- Kojimar notebook の明示 weight は `0.80 * Sunny + 0.20 * v10 生成物を使った積み上げ`。
- v10 stack は LightGBM / CatBoost / TabICL artifacts を使う。T4 系 GPU 推奨で、TabICL は GPU 差分に注意。
- hidden test 上で両 component を同じ notebook 内で再実行してから blend する点は、static public CSV blend より安全。

転用方針:

- 今すぐ自前実験の核にするより、PF/physical signal と tabular/artifact signal の非相関性を確認する benchmark として使う。
- 生成物を使った積み上げ を使う場合は dataset input、GPU、version、output hash を記録し、rerun stability を確認する。

### PF / beam / TabICL runnable guide

該当:

- `afr1ste/rogii-pf-beam-tabicl-stack-guide-9-150`
- `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl`

核:

- 2026-06-03 時点の Afr1ste note は verified public RMSE `9.150` を account pin として記録。
- submission hygiene が具体的で、`submission.csv` shape、id order、finite value、kernel ref、version、public score、blank score / 400 error を分けて記録する。
- candidate を pinned best output と比較し、重複 submission や壊れた branch を検出する方針。

転用方針:

- 実験管理側に取り込む価値が高い。提出前に candidate 同士の pairwise distance、per-well range、prediction start 近傍の連続性、smoothing 前後の差を確認する。
- model 本体より、submission monitoring / validation discipline の改善として優先的に使う。

### Golden LGB/Cat simple feature family

該当:

- `beicicc/rogii-0603-yarukikun-golden-lgbcat`

核:

- `Z_MD_Interaction`, `Z_since`, `Deviation_absolute`, `Z_local_norm`, `Tortuosity`, `GR_cumsum_since` の 6 features。
- LightGBM + CatBoost の GroupKFold ensemble、Savitzky-Golay smoothing。

転用方針:

- score 実績がまだ見えないため低優先。exp010 で trajectory features が悪化した現状では、単純 trajectory feature を増やすより、悪化条件の audit を先に行う。

## 次に反映する実験観点

1. `trajectory_feature_error_audit` を先に実施し、trajectory / GR missing / eval length / Z span と悪化 well の関係を見る。
2. PF / beam は `exp011_tracker_divergence_features` の候補として、scale別 PF、beam、hold-last-known、likelihood、推定器間 divergence を add-only にする。
3. 提出候補を作る段階では、Afr1ste / Jiayi Du の checklist を `submit-check` 側に寄せる。特に hidden rerun で sample shape が変わる前提、id merge、per-well range、pairwise distance を見る。
4. public score の高い `LB 8.860/8.863` family は観察価値が高いが、見えている train well 用の物理処理 と static public probe に寄りすぎないようにする。
