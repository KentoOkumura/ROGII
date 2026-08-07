# exp335_signed_residual_meta_on_exp264 結果

## 状態

Kaggle T4 Stage D version 2は15/15 modelsを完了した。pooled OOF RMSEはsaved exp264の`8.460811237612477`から`8.146107755881022`へ`0.314703 ft`改善し、4/5 foldsと全5 scopeで非悪化だった。一方、事前固定したby-well tailとclean273 promotion guardを通過しなかったため、scientific-support / promotion gateはいずれもFAILとする。gateは緩和しない。その後のユーザー明示overrideにより、保存済みmodelのCPU推論だけを実施し、version 3でsubmit-checkまでPASSした。ユーザーによるcode submissionはPublic LB `7.517`を記録したが、train-side非promote判断は維持する。

## 仮説と単一変更

corrected exp264のclean273 + saved selector compact 74列を固定し、12候補の`true_tvt - candidate_tvt`をstrict nestedで予測した23列だけをadd-onlyした。既存control、fold、seed、候補順、LightGBM 3 configは変更・再学習していない。

## 実行契約

- Route: `ml_model`
- Stage S: 1 objective × outer 5 × inner 4 = 20 CPU boosters
- Stage D: 1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters
- 最終特徴: clean273 + saved74 + signed23 = 370列
- saved exp264 control再学習: 0 boosters
- Stage D runtime: Kaggle T4、internet off、`gpu_use_dp=true`、threads 8
- canonical kernel: `kentookumura/exp335-signed-residual-meta-on-exp264-tvt-train` version 2、id_no `128232946`
- 実行時間: `20,017.036 sec`（約5時間33分37秒）

## 主要結果

| 指標 | Saved exp264 | exp335 | 判定 |
| --- | ---: | ---: | --- |
| Pooled OOF RMSE | `8.460811` | `8.146108` | `-0.314703 ft`、PASS |
| Nonworse folds | - | `4 / 5` | PASS |
| 最大scope delta | - | `-0.090418 ft` | 全5 scope PASS |
| By-well delta p95 | - | `+1.728657 ft` | FAIL（要件`<=0`） |
| Worst-well delta | - | `+10.238752 ft` | FAIL（要件`<=+0.25`） |
| Scientific-support gate | - | FAIL | tail guard不通過 |
| Train-side promotion gate | - | FAIL | clean273 tail悪化 |

### Fold別 downstream OOF

| Fold | Saved exp264 | exp335 | Delta |
| ---: | ---: | ---: | ---: |
| 0 | 8.468093 | 8.000562 | -0.467532 |
| 1 | 8.309700 | 8.685334 | +0.375633 |
| 2 | 8.249391 | 7.672685 | -0.576706 |
| 3 | 8.450573 | 7.832720 | -0.617853 |
| 4 | 8.814966 | 8.493250 | -0.321716 |

### 固定scope

| Scope | Saved exp264 | exp335 | Delta |
| --- | ---: | ---: | ---: |
| 0--250 ft | 1.583151 | 1.492732 | -0.090418 |
| 250--1000 ft | 4.099686 | 3.910356 | -0.189330 |
| 1000+ ft | 9.302283 | 8.959723 | -0.342560 |
| Hidden-like spatial | 9.420315 | 8.852269 | -0.568045 |
| Hidden-like typewell-purged | 9.341391 | 8.809396 | -0.531995 |

## Tail / promotion readout

- 773 wells中、428 wellsが改善または同等、345 wellsが悪化した。delta中央値は`-0.074115 ft`だがp95は`+1.728657 ft`だった。
- worst wellは`fb03ae90`。saved exp264 `29.631678`からexp335 `39.870430`へ`+10.238752 ft`悪化した。
- clean273比worst deltaはsaved exp264 `+14.482873 ft`からexp335 `+17.774910 ft`へ悪化した。
- clean273比`+1/+3/+5 ft`悪化well数は`135/39/14`から`150/53/21`へすべて増えた。
- signed 23列のgainは`1.454105e11`で非ゼロ。最大1特徴のshareは`25.296%`で、単一特徴への極端な集中は見られない。平均改善のsignalは実在するが、well-tailで安全に一般化しなかった。

## Stage S

Stage S version 3は20/20 models、25/25 partitions、technical gateをPASSした。signed residual pooled RMSEはprior `10.974122864313635`に対して`8.430777428355306`、5/5 folds改善だった。この結果は方向signalの学習可能性を支持するが、Stage Dのtail FAILを上書きしない。

## 技術監査と再現性

- Stage D model manifestは15個の`fold × config` slot、15 unique model SHA、全model 370特徴を記録した。best iterationは251--10,000、median 1,895。
- 3,783,989 rows / 773 wells、pooled/fold/scope/by-well/promotion count/signed gainを取得artifactから再計算し、保存metricsと一致した。
- Stage C/Sの25 partition、Stage Sの20 models、saved exp264 OOF/metrics、clean273 allowlist/schemaはfit前SHA gateを通過した。
- 大容量OOF/model payloadはローカルへ一括取得せず、Kaggle側manifestのSHAを証拠とした。small metrics、manifest、readout、logだけを保存した。
- OOF prediction SHA: `8b28a3f29b981cbba118c9f98a5e7dd92e75613d87dddce39c2d162fb6a769b1`
- Metrics SHA: `dd4502a5f8620820a023d7663b8335b20ea4e26ad847ff5c45757b6935c42ae1`
- Model manifest SHA: `bfe917ba446096026c6e8bc6f0ac0a0a33c69b5d5602e140152caccc5d2d3bcd`
- Reproducibility manifest SHA: `85ead119035604bd5559de566f57d0c5088e8f3c3cfbd4f6f5a13d8f07e21cba`
- Downloaded kernel log SHA: `48609a0899a2999a71aabeb9abbc51375009ebc3e6157073bb3068f57f6b3829`
- deterministic anchor: false。GPU LightGBMのbitwise rerunは未実施。

## 結論

signed residual 23列はpooled/fold/scopeを大きく改善したが、345 wellsを悪化させ、固定tail guardを大幅に超えた。技術的に信頼できるnegative promotion resultとして保存し、本実験内でobjective/grid/threshold/特徴追加による救済は行わない。2026-07-23のユーザーoverrideでは、このFAILを保持した保存済みmodel CPU inferenceだけを例外実施する。

## CPU inference override

- Kaggle CPU / internet off、学習booster 0
- corrected exp264 Stage C parent selector 40 models + exp335 Stage S signed selector 20 models + exp335 Stage D TVT 15 models
- current-test featureは同一run内で再生成し、final 370列とouter対応を維持
- canonical kernel: `kentookumura/exp335-signed-residual-meta-on-exp264-inference` version 3、id_no `128358534`
- runtime: `387.808 sec`、14,151 rows / 3 wells
- 40/20/15 model SHA、88/74/23/273/370 feature契約、formula/top-1 parity `0.0`を確認
- `submission.csv`: `id,tvt`、14,151 rows、sampleとのID内容・順序完全一致、重複/NaN/Infなし
- submit-check: PASS、WARN/FAIL 0
- submission SHA: `9d163b11fbea5c6a1e807f9681aaf39916bb5682e35ea874d3acd981f922a14f`
- user-submitted ref: `54928806`、status `COMPLETE`、Public LB `7.517`
- Public LB改善: exp287 `7.530`比`-0.013`、exp264 `7.562`比`-0.045`
- agentによるcompetition submitは実行していない

## 次のアクション

train-side guard FAILと非promote判断を維持し、Public LB `7.517`をsubmitted reference anchorとして扱う。
