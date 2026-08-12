# 設計

## アプローチ

exp263 Stage 0 candidate-major cache から exp264 が使用する6 primitiveを読み、5 pairと1 fixed formulaを
同じ式で再構成して既存12候補bankを固定する。truthを持たないこの段階でwell集約gate featureを作り、
outer-fold内 percentile rankの等重み平均でtop 25%を凍結・保存する。その後にだけexp279 OOFの
`geop_hmm`とreadout-only truthをID joinし、full 13候補unionとgate付きsparse unionを監査する。

Stage 0でfull 13候補oracleのheadroomは確認できた一方、固定gateだけが失敗した。ユーザーの
追加指示に従い、Stage Bではgateを廃止し、保存済み`geop_hmm`を全wellの正式な13番目candidateに
変換してexp264と同じdual-objective candidate-long selectorをouter well 5-foldで再学習する。
exp264 Stage B v5の12候補OOF metricsは固定baselineとして読み、controlは再学習しない。

## 実験範囲

- 対象実験: `exp286_geop_hmm_sparse_addonly_candidate_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 追加候補: `exp279_exp226_geop_centered_exact_hmm_redecode/geop_hmm`
- 変更する変数: 13番目候補の有無と、truth-free固定gateによるcandidate availabilityだけ。
- 固定する変数: 既存12候補、2 legal domains、fallback、candidate formula、outer fold、truth、
  512-block定義、hidden-like assignment、guard閾値。

Stage Bでは変更する変数を13番目候補の有無だけに限定する。既存12候補、2 legal domains、
fallback、candidate formula、outer fold、raw-test-safe context、LightGBM設定、目的関数、samplingを
exp264 Stage B v5と同じに固定する。

## Stage B candidate-long設計

- ID: `geop_hmm`（string artifact + `id__candidate__geop_hmm` one-hot、ordinal index禁止）。
- kind / family: `primitive` / `geop_centered_exact_hmm`。
- availability: exp279 OOFとexp263 keyが一致する全行で1。gateなし。
- native confidence: `geop_hmm_std -> sigma_tvt`、`geop_hmm_loglik -> source_loglik`、
  `geop_hmm_loglik / evaluation rows in well -> loglik_per_row`、finite flag、validity flag。
- universal proxy: candidate TVT/anchor差、local shape 32/128/512、bank disagreement、legal-domain統計を
  共有pipelineで全候補同様に生成する。
- 禁止: `true_tvt_readout_only`、candidate error、oracle、catalog RMSEをfeature/confidenceへ渡さない。

Stage Aで候補13本のfeature schemaを再監査・freezeし、Stage Bで`pred_abs_error`と`p_within10`を
5 foldsずつ学習する。合計10 CPU boosters、HMM/PF well-run 0、親/control booster 0とする。

## Stage B比較設計

12候補baselineはexp264 corrected Stage B v5 artifactsとSHAで固定する。13候補版について、
hard primary pathのpooled/fold RMSE、rank regret、top-3 oracle coverage、candidate選択率を直接比較する。
candidate-long pooled scoreは候補数が異なるため補助指標とし、shared-12 candidate/foldごとのscore差を
apples-to-apples比較として別CSVへ保存する。13候補自身のprior対score guardも従来通り判定する。

## Stage B実測

Kaggle CPU version 4で10 modelsを完走し、hard RMSEはparent12 `8.587004`からnew13 `8.477740`へ
改善、3/5 folds改善、`geop_hmm` pred-error選択率19.50%、ID/native confidence coverage 1.0、
13候補score guard 5/5で事前addition guardをPASSした。fixed fallback 8.238332には届かないため、
hard inferenceには採用せず、Stage C以降は別承認とする。

## 2026-07-19 Stage C/D昇格

ユーザーは判断基準を元selectorとの比較と明示し、Stage Bの`8.587004 -> 8.477740`改善を受けて
Stage CからStage Dまでの実行を承認した。fixed fallbackはこの昇格判断に使わない。

- Stage C: full13 selectorをouter 5 x inner 4 x 2 objectivesでnested学習し、40 CPU modelsと
  25 compact partitionsを生成する。outer-trainはinner OOF、outer-validはinner 4 model ensembleとする。
- Stage D: clean 273 base + full13 compact 77 = 350 featuresのadd-onlyだけを3 configs x 5 foldsで
  15 GPU models学習する。保存済みexp264 control 273列と12候補add-only 347列をbaselineにし、
  control再学習は0とする。
- Stage Cのscore/leakage guardを通過した生成物だけをStage Dへ渡す。inference/submissionはscope外。

### Stage C/D実測

- Stage Cは40/40 CPU models、25 partitions、77 compact featuresを生成し、nested hard RMSEを
  `8.652532 -> 8.448682`へ改善した。4/5 folds、score/leakage guardはPASSした。
- Stage Dは15/15 T4 boostersを生成し、parent12 compact add-only `8.460811`に対してfull13
  `8.403784`、delta `-0.057027 ft`だった。near / mid / 1000+とhidden-like 2面は改善した。
- 一方、fold改善2/5、400/773 wells悪化、worst `+5.862833 ft`で総合guardはFAILした。
  pooled改善と安定性不足を分離して記録し、inference/submissionは無効のままとする。

## Gate設計

事前gateは次の8 well featureのouter-fold内rankを等重みする。NaN rankは中立値0.5とし、
score降順で各fold `floor(0.25 * n_wells)` 以下を選ぶ。cutoff境界で同scoreが複数wellに
またがる場合は境界tie全体を除外し、well IDで選ばない。

| feature | 集約 | 高rank方向 | 理由 |
| --- | --- | --- | --- |
| `geometry_gr_delta_abs_median` | exp226 GR delta絶対値median | 小 | geometry anchorの整合 |
| `known_prefix_rows` | suffix開始row | 大 | geometry donor/prefix support |
| `exact_sigma_tvt_p90` | exact HMM sigma p90 | 大 | existing exactの不確実性 |
| `exact_neg_loglik_per_row_median` | `-loglik_per_row` median | 大 | exact emissionの弱さ |
| `exp226_exact_abs_median` | path差median | 大 | geometry/exactの相補余地 |
| `exp226_selfgr_abs_median` | path差median | 大 | geometry/self-GRの相補余地 |
| `existing_bank_std_median` | 既存bank row std median | 大 | bank disagreement |
| `tail_rows` | evaluation suffix rows | 大 | long-tail復帰余地 |

feature名・方向は実行前にconfigへ固定し、結果を見て追加、削除、方向反転、重み変更しない。

## Oracle/readout設計

- row oracle: 各rowで最小二乗誤差候補を選ぶ。
- 512-block oracle: suffix先頭から非重複512行blockを作り、block SSE最小候補をblock全体へ選ぶ。
- whole-well oracle: well SSE最小候補をwell全体へ選ぶ。
- full unionは保存済み`geop_hmm`を全wellで診断に使う。sparse unionはgate外`geop_hmm=NaN`、
  `candidate_available=0`とし、既存12候補を常に保持する。
- full/sparse双方のpooled、fold、1000+、hidden-like、by-wellを保存する。unique-bestは
  `geop_hmm`誤差が既存bank bestより固定tolerance以上小さいrow/block/wellだけを数える。
- gate retentionはwhole-well oracleのadditive SSE reductionで計算する。

## Runtime設計

Stage 0の保存OOF監査はhidden runtimeの根拠にしない。別途作る200-well paired shadow manifestに、
raw-test schema、well/fold、base秒、geop追加秒、gate選択、selector/TVT/save固定overheadを記録する。
local `default_rng(seed)`だけを使う2,000 bootstrapでp50/p95を算出する。manifest欠損、truth列混入、
200 wells未満、coverage超過はfail-closed。hard guardはselected `<=min(floor(.25N),50)`、各fold`<=30%`、
geop追加p95`<=2700秒`、total p95`<=27000秒`。

## 再現性設計

- seed policy: gate/oracleはRNGなし、runtime bootstrapだけ固定local RNG seed 42。
- stochastic 処理: runtime manifest bootstrapのみ。global RNGは使わない。
- PF/Beam / likelihood-PF / seed bagging: 保存済み候補を読むだけで再生成0。
- 並列処理: Stage 0はsingle process。乱数とthread schedulingの関係なし。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off。LightGBM/HMM/PF実行0。
- input SHA: exp263 manifest file SHA、各読込partition SHA、exp279 gzip raw/decompressed SHA、
  hidden assignment SHAを保存する。
- feature SHA: truth-free gate CSVのschema/content SHAをtruth join前に確定する。
- model/prediction/submission SHA: model/prediction/submissionを生成しないため対象外。
- Kaggle bootstrap: prepare後にloose config/sourceとembedded bootstrap SHAを照合する。

## リスク

- リークリスク: gate構築関数はtruth/error/oracle/geop/well ID派生feature名を拒否し、gate artifact保存後だけ
  exp279 truthを読む。
- CV/LB不一致: oracleは候補headroom診断であり実モデルCVではない。Stage D guard通過前は採用しない。
- selector再学習結果はOOF selector診断であり、そのままinference/submissionを有効化しない。
- ランタイム/メモリ: 3.78M rows x 12候補をfloat32 matrixで保持し、誤差は候補ごとにreduceして
  大きなcandidate-long frameを作らない。
- 再現性: gzipはdecompressed content SHAを主証拠にし、1回のdiagnosticをdeterministic submission
  anchorとは呼ばない。
