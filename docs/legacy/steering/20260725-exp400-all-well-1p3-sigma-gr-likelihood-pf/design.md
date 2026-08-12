# 設計

## 結論

exp072の決定論的128-seed likelihood-PFをscientific parentとし、全773 train wellsで
GR観測ノイズ幅だけを次のように変更する。

```text
gs_base = clip(
    nanstd(fillna(horizontal_GR_known_prefix, 0)
           - typewell_GR_at_TVT_input),
    10,
    60,
)
gs_candidate = 1.3 * gs_base
```

`1.3` はbase clip後に正確に1回だけ掛け、再clipしない。したがって候補の有効範囲は
`[13, 78]` である。変更は全well一律で、selectorやfallbackは置かない。

primary predictionは既存PF候補の中で固定anchorになっている `likpf_mean` とする。
保存済みexp072 `likpf_mean`をx1.0 controlとしてload-onlyで使い、候補x1.3だけを
新規生成する。

## 外部根拠の範囲

- Discussion:
  `https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712`
- Linked Notebook:
  `https://www.kaggle.com/code/hjyact/ultimate-pf-config-strategy-a-reproducible-score`
- 投稿は、公開Notebookの `lik_pf` 内で `gs` をおよそ1.3倍するとscoreが改善したと共有している。
- 投稿本文には比較score、CV、Notebook version、対象branch、再現runの詳細がない。
- 2026-07-25に確認した現行Notebook sourceでは、`* 1.3` は後半の
  `lik_pf` にある一方、最終selectorから呼ばれる別の `run_particle_filter` は
  x1.0のままである。したがって投稿の変更が現行Notebook全体の最終予測へ
  どの経路で寄与するかはsourceだけでは確定できない。

本実験は公開Notebook全体の再現ではない。投稿で示された1行のmechanismを、
同型kernelを持つlocal exp072へ移した原因分離実験である。公開Notebookのdefault
`seed_base=0`も再現せず、exp072で検証済みのstable per-well seedを維持する。

## 実験範囲

- 実験: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- Route: `pf_beam`
- scientific parent: `exp072_exp063_full_replay_feature_cache`
- fixed reporting reference:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- fold identity:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- hidden-like roles:
  `exp115_hidden_like_spatial_holdout_from_ppt`
- sibling negative reference:
  `exp398_all_well_1p3_sigma_gr_exact_hmm`

exp398は同じ数値倍率をexact HMMへ適用して悪化したが、PFは逐次重み更新、
ESS resampling、particle roughening、128-seed aggregationを持つ別algorithmである。
exp398のFAILをPFの結論へ流用せず、同時にexp398を救済・再分類もしない。

## 固定するlikelihood-PF

exp072の `lik_pf` / `_pf_lik_allseeds` contractを固定する。

| 項目 | 固定値 |
| --- | --- |
| particles | 500 |
| seeds | 128 |
| seed indices | 0--127 |
| seed weighting scales | 3, 5, 8, 12 |
| Type Well TVT grid step | 0.2 ft |
| initial position spread | 4.5 ft |
| initial rate spread | 0.01 |
| momentum | 0.998 |
| rate noise | 0.002 |
| position noise | 0.005 |
| rough position | 0.1 |
| rough rate | 0.001 |
| resampling threshold | ESS < 0.5 × particles |
| emission | `exp(-0.5 * min(((GR-observed_expected)/gs)^2, 600))` |
| eval GR missing | linear interpolation both directions, then Type Well GR mean |
| particle TVT clamp | Type Well TVT range ±100 ft |
| seed aggregation | scale-weighted outputs 3/5/8/12 and arithmetic `pf_mean` |

開始positionは最後のknown `TVT_input + Z`、初期rateはknown prefix末尾30 rowsの
`(ΔTVT_input + ΔZ) / ΔMD` medianとする。Type WellはTVTでsortし、GR missingを
Type Well meanで補う。ここを含め、`gs` 係数以外は変更しない。

## seed契約

exp072 deterministic v2と同じpolicyを使う。

```text
key = "likpf::train::<well_id>"
seed_base = int(SHA256(key)[0:16], 16) mod 2_147_483_647 + 1
seed_s = seed_base + s,  s = 0, ..., 127
```

well IDを文字列として読み、辞書順に固定する。well単位のthread並列は
`n_jobs=8`に固定し、各Numba kernelへ明示seedを渡す。global Python /
NumPy RNGをthread側から追加利用しない。

x1.3によりresampling時点が変わり、その後のtrajectoryがx1.0から分岐することは
treatmentの一部である。seed labelと初期seed policyはpairedに保つが、
seed pathのrow-wise一致は要求しない。

## 出力とprimary endpoint

同じcandidate PF実行から次を保存する。

- primary: `likpf_mean_x1p3`
- secondary diagnostic:
  `likpf_scale_3_x1p3`、`likpf_scale_5_x1p3`、
  `likpf_scale_8_x1p3`、`likpf_scale_12_x1p3`
- per-well audit:
  `gs_base`、`gs_candidate`、multiplier、prefix rows、seed_base、
  eval rows、runtime、status

primary controlは保存exp072の
`last_known_tvt + likpf_mean_d`で、train-side RMSEの期待値は
`11.594897672217703`。secondary scaleは同名の保存exp072列と対比するが、
実行後にbest scaleを選んでprimaryを差し替えない。

exp209保存Gaussian exact-HMMとのfixed 50:50は次のreporting guardだけに使う。

```text
parent_blend    = 0.5 * saved_exp209_hmm + 0.5 * saved_exp072_likpf_mean
candidate_blend = 0.5 * saved_exp209_hmm + 0.5 * likpf_mean_x1p3
```

blend weightを探索しない。これはPF interventionのdownstream safety checkであり、
新しいensemble experimentやprediction routeにはしない。

## 保存入力

| 入力 | 固定証拠 / 用途 |
| --- | --- |
| exp072 train cache | raw gzip SHA256 `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18` / decompressed SHA256 `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`、x1.0 control |
| exp209 HMM cache | decompressed SHA256 `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`、fixed blend |
| exp226 OOF | decompressed SHA256 `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`、row/fold identity |
| exp115 role assignment | SHA256 `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`、late-join stress |

exp072 cacheは3,783,989 rows / 773 wellsを期待する。IDとwellはCSV parse時から
string dtypeに固定する。saved controlのrow order、prediction coverage、expected RMSE、
decompressed content SHAが一致しなければPF開始前にfail closeする。

## truth-late契約

candidate生成時のhorizontal fileは `MD / Z / GR / TVT_input` だけを読む。
unknown-suffix target `TVT`、error、fold score、hidden-like roleは読まない。
Type Wellの `TVT` はreference curveの座標として必要であり、horizontal suffixの
target truthとは別物である。

全candidate predictionをID順にfreezeし、logical content SHAを保存した後だけ、
horizontal unknown-suffix `TVT`、exp226 fold、exp115 roleを評価用にjoinする。
truth-before-freeze read countは0でなければならない。

## technical gate

すべてANDで判定する。

1. raw identity、exp072、exp209、exp226、exp115の期待SHAが一致する。
2. score surfaceが3,783,989 rows / 773 wells / 5 reporting foldsで一致する。
3. candidateが773 wellsすべてを1回ずつ処理し、failure / fallback wellが0。
4. `gs_candidate / gs_base` が全wellでabsolute tolerance `1e-12`以内に1.3。
5. `gs_base` が `[10,60]`、candidateが `[13,78]` で、post clipが0。
6. particles 500、seeds 128、seed indices 0--127、scale outputs
   3/5/8/12、seed policy、PF constantsがcontractと一致する。
7. primary / secondary predictionのfinite coverageが1.0で、row identityが一致する。
8. truth / error / fold score / hidden-like roleのread countがprediction freeze前に0。
9. actual execution countが773 PF well-runs / 98,944 seed-well trajectories /
   49,472,000 particle startsで、control PF / HMM / Beam rerunが0。
10. input manifest、scientific contract、prediction content、artifact manifestのSHAを保存する。

実装時には、exp072 parent kernelとcandidate側kernelのmultiplier x1.0が
合成fixture上でexact一致するcontract testを作る。これはfull parent control
再実行には数えない。

## scientific promotion gate

primary `likpf_mean_x1p3`について全項目をANDで判定する。
gainは `control RMSE - candidate RMSE` とし、正が改善である。

1. saved exp072 `likpf_mean`比pooled RMSE gain `>= 0.05 ft`。
2. 5 reporting folds中4 folds以上でcandidate RMSEがcontrol以下。
3. raw GR observed rowsのRMSE gain `>= 0.05 ft`。
4. raw GR missing rowsのRMSE regression `<= 0.00 ft`。
5. high missing-fraction wellsのRMSE regression `<= 0.00 ft`。
6. suffix distance 1000 ft以上のRMSE regression `<= 0.00 ft`。
7. hidden-like spatial valid scopeのRMSE regression `<= 0.00 ft`。
8. hidden-like typewell-purged valid scopeのRMSE regression `<= 0.00 ft`。
9. by-well RMSE delta p95 `<= 0.00 ft`。
10. worst-well RMSE regression `<= +0.25 ft`。
11. fixed exp209-HMM / candidate-PF 50:50がparent fixed blend比で
    regression `<= 0.00 ft`。

secondary scale 3/5/8/12、MAE、bias、within 10 ft、ESS / resampling回数、
seed likelihood spread、clip position、well別gainは診断保存するが、
上のprimary gateへ追加・置換しない。

PASS時は同じexp400内のfail-closed inference実装候補にできるだけで、
inference、submission、full public Notebook replayを自動承認しない。
FAIL時はx1.3 global likelihood-PF interventionをnegative resultとして閉じる。

## 実行量とruntime

- scientific variants: 1
- candidate PF well-runs: 773
- seeds per well: 128
- seed-well trajectories: 98,944
- particles per seed: 500
- particle starts: 49,472,000
- prediction readouts: 5
- reporting folds: 5
- LightGBM config / trained fold / booster: 0 / 0 / 0
- parent PF control / HMM / Beam rerun: 0 / 0 / 0
- GPU / inference / submission: 0 / 0 / 0

exp072 deterministic v2はbase featureを含む全cache generationに
15,380.262秒、Notebook全体に17,728.972秒を要した。exp400はlikelihood-PF
candidateだけだが、事前runtime上限はKaggle CPUの30,600秒とする。
1本のCPU Notebook、8 well workersを前提とし、shardや複数versionへの分割は
実装時に別判断しない。

## 再現性設計

- stochastic components:
  particle initialization、process noise、systematic resampling、
  post-resampling roughening。
- seed policy:
  exp072 deterministic v2のSHA256 per-well base + seed index。
- parallel policy:
  well-level thread並列、固定`n_jobs=8`、kernelへ明示seed、stable well order。
- deterministic anchor:
  初回runだけでは主張しない。fixed source/seed/content SHAを持つ
  deterministic candidateとして記録し、submission anchorにはしない。
- gzip:
  raw gzip SHAとdecompressed content SHAを分け、後者を主証拠にする。
- record:
  Kaggle kernel ID/version、CPU/GPU/internet、source/config/contract SHA、
  input manifest、row/schema/content SHA、prediction SHA、artifact manifest。
- model/submission SHA:
  modelとsubmissionを作らないため対象外。
- raw test:
  train-side PASS後もtest再生成のdeterminismは未検証なので、
  inference実装前に別途同じseed contractとcontent SHAを設計する。

## 禁止事項

- multiplier、base clip、post clip、particle数、seed数、weighting scale、
  initial spread、resampling thresholdのgrid
- well/row-adaptive multiplier、missingness gate、truth/error/worst-well selector
- scale 3/5/8/12の実行後best選択によるprimary差替え
- parent x1.0 PF full rerun、exp398 HMM rerun、Beam再生成
- Huber / Student-t / mixture、ESS reliability、adaptive outlier rescue
- selector、hold blend、Ridge、projection、contact guard、ML feature再学習
- same-OOF rescue、blend weight search、Public LBによる設定選択
- train-side全gate PASS前のinference、submission

## 現在の承認境界

2026-07-25の最初のユーザー指示は、backlog、steering、experiment scaffold、
design確定だけを承認した。その後の「exp400を実装してください」により、
別名compact self-contained train候補、fail-closed inference候補、専用contract
testのimplementation-onlyを承認済みとする。

正規Notebook採用、Kaggle package、push、train run、train-side PF実行、
inference、submissionは引き続き未承認である。

## 実行承認とterminal結果

2026-07-25のユーザー指示「実行してください」により、正規train Notebook採用、
private CPU package / push / runを追加承認した。canonical kernel
`kentookumura/exp400-all-well-1p3-sigma-gr-likelihood-pf-train`
version 1 / id_no `128585102`を実行し、`10496.299889 sec`で完走した。

technical gateはPASSした。primary `likpf_mean_x1p3` RMSE
`12.221810980460939`はsaved exp072 control `11.594894395642696`から
`0.6269165848182432 ft`悪化し、科学gateはFAILした。fixed HMM 50:50も
`0.39027517454298355 ft`悪化した。

設計済みFAIL contractをそのまま適用し、decision
`all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`でbranchを閉じる。
secondary scale readoutはcandidate-only診断のままとし、最良scaleのpost-hoc
採用、別設定での救済、inference、submissionを行わない。

## 実装時に確認した保存入力の制約

固定raw SHA
`14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
のexp072 cache headerには`last_known_tvt`と`likpf_mean_d`があるが、
`likpf_scale_3/5/8/12_d`は保存されていない。

primaryのsaved x1.0 `likpf_mean`比較とpromotion gateは設計どおり維持する。
x1.3のscale 3/5/8/12は同一candidate PF runから保存してcandidate-onlyの
nonselective diagnosticとして評価する。x1.0 scale別比較を得るためのparent PF
再実行は行わず、scale別readoutをprimaryへ差し替えない。
