# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `segment_local_negative_space_gr_corridor_audit` 実装契約を、既存の `exp249` とは切り離して `exp250_segment_local_negative_space_gr_corridor_audit` として実装する。

exp246 の full-tail global surface と `valid_after_history` を再利用せず、official evaluation tail を MD 256 ft / stride 128 ft の overlap segment に分割する。well 全体で robust-z 化した horizontal/typewell GR から 4 ft grid の mismatch cost を作り、左から右だけに進む DAG 上の minimum-bottleneck path と near-optimal corridorを監査する。

## 固定する科学契約

- Route: `pf_beam`。
- 親: `exp246_negative_space_gr_barrier_audit`。固定候補は `exp072_exp063_full_replay_feature_cache`、hidden-like 集計は `exp115_hidden_like_spatial_holdout_from_ppt` を使う。
- horizontal segment: 256 ft、stride 128 ft、4 ft/bin、通常 64 columns。1 bin は有限 GR 2 点以上で support。
- short tail は実在 MD 範囲だけを可変 column にし、padding・端値複製をしない。16 columns 未満は topology primary から除外する。
- typewell grid: target-free flat-Z prior を中心に ±256 ft、4 ft/state、129 states。範囲外 extrapolation はしない。
- horizontal/typewell GR は well 全体で別々に median/IQR robust-z、`[-8, 8]` clip。segment/candidate/truth 単位の再正規化は禁止する。
- diagnostic surface は `real_gr` と stable SHA circular shift による `shuffled_typewell_gr` の 2 面。
- edge は右向きの `dy=-1/0/+1` のみ。単独 unsupported column だけ `dx=2, dy=-2..+2` の gap edge を許す。
- first segment は `last_known_tvt ±8 ft` source の anchored graph、全 segment は全左端 state source の spanning graphを保存する。segment 間で path/history を持ち越さない。
- primary threshold は固定 grid でなく `tau_star = min_path max(D)`。同値は累積 cost 最小で tie-break する。
- corridor は `D <= tau_star + 0.25` 上の forward/backward reachability。primary path の上下 1 state mask 後の second path gap も 1 回だけ計算する。
- fixed 5 candidate と truth を後段 readout に通す。truth/error/hidden-like role は surface、segment、grid、source/sink、path、thresholdの選択に使わない。
- primary risk は `corridor_outside_fraction`。bad label は candidate absolute error `>10 ft`。
- overlap view は個別 sample のまま評価し、OR/AND/平均/min/max/majority で row rule に統合しない。

## 実行コスト・禁止事項

- active diagnostic surfaces: 2。
- LightGBM / CNN / HMM config: 0、fold: 0、booster: 0。
- PF/Beam/likPF 再生成: 0。parent/control 再学習なし。
- Kaggle CPU、GPU/internet disabled、single process。
- candidate 値変更、prune、blend、fixed threshold grid、HMM/PF/Beam edge cut、raw-test inference、submission を禁止する。
- inference notebook は train-side-only guard で停止する。

## Stage 0 / Stage 1

- Stage 0 は stable SHA normal 6 wells、raw-only high-GR-missing 2、flat-GR 2、long-tail 2を選び、first/middle/last segmentを最大1枚ずつ描く。
- plot は real/shuffled mismatch、source/sink、primary path、corridor、固定5候補、anchor、評価layerのtruthを表示する。`vmin=0`, `vmax=3` とするが DP cost は表示clipしない。
- Stage 0 は synthetic DAG/DP contract、axis/grid、real-vs-shuffled support parity、no padding/no target crop/no history、config/input/plot SHA を検証する。
- Stage 1 は Stage 0 の manual parity確認後だけ同一configで 773 wells を処理する。結果を見て grid/slack/stride/half-width を変更しない。

## 受け入れ基準

- `.steering/`、`config.yaml`、compact self-contained Jupytext train/inference notebook、`SESSION_NOTES.md`、`README.md`、`result.md`、`metrics.json` が揃う。
- candidate cache の ID/well/row coverage、duplicate、finite、raw/decompressed SHA を実行前に assert する。
- synthetic contract が left-only component、broken component、1-column gap、2-column gap、anchor-outside component、tie-break を assert する。
- Stage 0 と Stage 1 の両方を実装し、初期 active mode は `stage0_preview`、Stage 1 は manual gate で無効にする。
- segment、candidate-segment、candidate、group、overlap、by-well、plot manifest、summary の所定生成物を出せる。
- gzip は mtime 0 で保存し、raw SHA と decompressed content SHA を分け、後者を主証拠にする。
- fixed input に対する audit determinism と upstream exp072 cache の stochastic provenance を分け、deterministic prediction/submission anchor とは扱わない。
