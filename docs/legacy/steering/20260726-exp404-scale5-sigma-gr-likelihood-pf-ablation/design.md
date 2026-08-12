# 設計

## 結論

exp400のcandidate-only secondary診断で`gs × 1.3 + scale_5`が
RMSE `11.174614846889103`だった一方、同じrunの算術`pf_mean`は
`12.221810980460939`だった。ただし保存exp072にはx1.0のscale 5列がなく、
この差から`gs × 1.3`の効果は判定できない。

そこでexp072-compatible likelihood-PFを全773 train wellsで2回実行し、
seed集約をtemperature 5へ固定して次のpaired A/Bだけを比較する。

```text
control   = gs_x1p0 + likelihood_weighted_scale_5
candidate = gs_x1p3 + likelihood_weighted_scale_5
gain      = RMSE(control) - RMSE(candidate)
```

2 variantは同じwell別seed baseとseed indexを使うcommon-random-number設計とする。
`gs`変更でparticle weight、ESS、resampling時点、その後の乱数消費とtrajectoryが
分岐することは介入効果に含める。

## 実験範囲

- 対象実験: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- Route: `pf_beam`
- scientific parent: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- PF kernel parent: `exp072_exp063_full_replay_feature_cache`
- reporting folds:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- hidden-like roles: `exp115_hidden_like_spatial_holdout_from_ppt`
- 変更する変数: `gs` multiplierの`1.0 / 1.3`のみ
- 固定する変数: base `gs` estimator、clip、scale temperature 5、particles、
  seeds、seed label、PF dynamics、likelihood、resampling、roughening、
  Type Well補間、GR missing補間、初期position/rate、fold、score rows

## 外部根拠とローカル根拠

- Discussion:
  `https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712`
- Linked Notebook:
  `https://www.kaggle.com/code/hjyact/ultimate-pf-config-strategy-a-reproducible-score`
- discussionは公開Notebookの`gs`約1.3倍でscoreが改善したと述べるが、
  数値、評価面、Notebook version、active branchは示していない。
- 現行公開Notebookのx1.3 `lik_pf`はscale 5を後段blendへ使う一方、
  別selector PFはx1.0であり、Notebook全体は複数PF、Beam、ML、
  calibrationを持つ。
- exp404は公開Notebook全体を再現せず、local exp072 kernel上で
  `scale_5`と`gs`倍率の交互作用だけを原因分離する。
- exp400のscale 5結果は仮説根拠とtechnical parityに使うが、
  x1.0対照の代用やpromotion結果には使わない。

## 固定するlikelihood-PF

| 項目 | 固定値 |
| --- | --- |
| particles | 500 |
| seeds | 128 |
| seed indices | 0--127 |
| seed aggregation | `exp((loglik-max(loglik))/5)`で正規化 |
| Type Well TVT grid step | 0.2 ft |
| initial position spread | 4.5 ft |
| initial rate spread | 0.01 |
| momentum | 0.998 |
| rate noise | 0.002 |
| position noise | 0.005 |
| rough position / rate | 0.1 / 0.001 |
| resampling | ESS < 0.5 × particles |
| emission | capped Gaussian、z²上限600 |
| particle TVT clamp | Type Well TVT範囲±100 ft |
| eval GR missing | 両方向線形補間後、Type Well GR mean |

開始positionはknown prefix末尾の`TVT_input + Z`、初期rateは末尾30 rowsの
`(ΔTVT_input + ΔZ)/ΔMD` medianとする。

## `gs`介入

```text
gs_base = clip(
    nanstd(fillna(horizontal_GR_known_prefix, 0)
           - typewell_GR_at_TVT_input),
    10,
    60,
)
gs_x1p0 = 1.0 * gs_base
gs_x1p3 = 1.3 * gs_base
```

base clip後に倍率を正確に1回だけ掛け、どちらもpost clipしない。
有効範囲はx1.0が`[10,60]`、x1.3が`[13,78]`である。

## paired seed契約

```text
key = "likpf::train::<well_id>"
seed_base = int(SHA256(key)[0:16], 16) mod 2_147_483_647 + 1
seed_s = seed_base + s, s=0,...,127
```

variant名とmultiplierをseed keyへ含めない。各wellでx1.0、x1.3の順に同じ
seed labelを渡す。well単位のthread並列は`n_jobs=8`、well順は辞書順へ固定し、
thread内のNumba kernel以外でglobal RNGを消費しない。

## primary endpointとparity出力

primary:

- control: `likpf_scale_5_x1p0`
- candidate: `likpf_scale_5_x1p3`
- gain: `RMSE(x1p0) - RMSE(x1p3)`。正がx1.3改善。

同じtrajectoryから`pf_mean_x1p0`と`pf_mean_x1p3`も保存するが、
次のtechnical parityにだけ使う。

- x1.0 mean対saved exp072 mean: RMSE差の絶対値`<=1e-5 ft`
- x1.3 mean対exp400 `12.221810980460939`: `<=1e-5 ft`
- x1.3 scale5対exp400 `11.174614846889103`: `<=1e-5 ft`

算術meanをscientific primary、救済候補、blendへ使わない。
scale 3/8/12は生成しない。

## truth-late契約

PF入力はhorizontal `MD / Z / GR / TVT_input`とType Well `TVT / GR`だけ。
両variantの全prediction、row identity、scientific contract、logical content SHAを
freezeしてreadbackした後にだけ、horizontal unknown suffix `TVT`、exp226 fold、
exp115 hidden-like roleをjoinする。freeze前のtruth / error / fold score /
hidden-like role read countは0を要求する。

## technical gate

すべてANDで判定する。

1. raw identity、exp072、exp226、exp115の期待SHAが一致する。
2. score surfaceが3,783,989 rows / 773 wells / 5 foldsで一致する。
3. 2 variantsが各773 wellsをfallbackなしで完走する。
4. variants間でwell、seed base、seed indicesが完全一致する。
5. `gs_x1p0/gs_base=1.0`、`gs_x1p3/gs_base=1.3`が全wellで
   absolute tolerance `1e-12`以内、post clip countが0。
6. particles、seeds、scale 5、PF constants、stable orderがcontractと一致する。
7. primary/parity predictionのfinite coverageが1.0、ID/orderが一致する。
8. 3つのRMSE parity checkが各`1e-5 ft`以内。
9. truth/error/fold score/hidden-like roleのfreeze前read countが0。
10. 実行量とinput/prediction/artifact manifestがcontractどおりでSHAを持つ。

## scientific promotion gate

`likpf_scale_5_x1p3`対`likpf_scale_5_x1p0`について全項目をAND判定する。

1. pooled RMSE gain `>=0.05 ft`
2. 5 folds中4 folds以上でx1.3がx1.0以下
3. raw GR observed rowsのgain `>=0.05 ft`
4. raw GR missing rowsのregression `<=0.00 ft`
5. high missing-fraction wellsのregression `<=0.00 ft`
6. suffix 1000 ft以上のregression `<=0.00 ft`
7. hidden-like spatial valid scopeのregression `<=0.00 ft`
8. hidden-like typewell-purged valid scopeのregression `<=0.00 ft`
9. by-well `RMSE_x1p3 - RMSE_x1p0` p95 `<=0.00 ft`
10. worst-well regression `<=+0.25 ft`

PASSしても、scale5 x1.3を推論・提出・公開Notebook blendへ自動昇格しない。
別のdownstream / raw-test再現性設計と明示承認を必要とする。
FAIL時はtemperature、multiplier、clip、seed、particle、well gate、blendを
同じOOFで救済せず、global x1.3 under scale5を閉じる。

## 実行量とruntime

- scientific variants: 2
- PF well-runs: 1,546
- seed-well trajectories: 197,888
- particle starts: 98,944,000
- primary / parity readouts: 2 / 2
- reporting folds: 5
- LightGBM config / trained fold / booster: 0 / 0 / 0
- HMM / Beam well-runs: 0 / 0
- GPU / inference / submission: 0 / 0 / 0

exp400の1 variant実測`10,496.300 sec`からの線形見積りは
`20,992.600 sec`（約5.83時間）。Kaggle CPU上限を`30,600 sec`、
peak RSS上限を25 GBとする。1 Notebook、8 well workersで実行し、
shardや別versionへの分割を初回設計に含めない。

## 再現性設計

- seed policy: SHA256 per-well common seeds + seed index
- stochastic 処理: particle初期化、process noise、systematic resampling、
  resampling後roughening
- PF/Beam: likelihood-PFのみ。Beam/HMM/modelなし
- 並列: 8 well threads、kernelへ明示seed、variant間共通seed、
  scheduling非依存のwell順と出力sort
- runtime: Kaggle CPU、GPU/AMPなし、internet off
- train生成物: schema、row/well count、logical content SHA、
  raw gzip SHAとdecompressed content SHAを分離
- raw test: 対象外。train PASS後も別のdeterminism設計なしに推論しない
- model/submission: 生成しないためSHA非該当
- Kaggle bootstrap: 実装・package承認後にmetadata、embedded config、
  kernel sources、CPU、internet、seed contractをpush前照合する
- deterministic anchor: 初回train runではsubmission anchorを主張しない

## リスク

- リークリスク: suffix GRはtest時にも観測可能だが、suffix TVTは不可。
  truth-late freezeとsafe-column allowlistで防ぐ。
- CV/LB不一致: train unknown suffixはhidden testや公開Notebook full pipelineと
  異なる。PASSしてもLB改善を意味しない。
- post-hocリスク: scale 5はexp400結果後に選ばれたため、この実験内では
  x1.0対x1.3だけを検定し、scale 5対meanのpromotion主張はしない。
- runtime: 2 variantでexp400の約2倍。30,600秒を超える可能性があり、
  実装時に小型fixtureとper-well見積りを記録する。
- 再現性: `gs`差でresampling後の乱数消費が分岐する。共通seedは同じ初期乱数
  labelを保証するがtrajectoryのrow-wise一致は保証しない。
- 外部再現: 公開Notebookはseed_base=0と複数の後段処理を持つため、
  exp404単体からdiscussionのscore差を断定しない。

## 禁止事項

- temperature、multiplier、clip、particles、seeds、initial spread、
  resampling thresholdのgrid
- scale 3/8/12の生成と実行後best選択
- arithmetic meanへのprimary差替え
- well/row-adaptive multiplier、missingness gate、true-error selector
- HMM、Beam、ML、selector、hold、Ridge、projection、calibration、
  model-package、blend weight探索
- same-OOF rescue
- 全gate PASSと別承認前のinference / submission

## 現在の承認境界

初回の2026-07-26指示はbacklog、experiment scaffold、steering、
design確定だけを承認した。その後の「exp404を実装してください」により、
別名compact self-contained source / Notebook候補と専用testの実装まで承認済み。
さらに「実行してください」により、正規Notebook編集、Kaggle private CPU
package、push、train version 1 runを承認済みとする。inferenceとsubmissionは
引き続き未承認である。
