# 設計

## アプローチ

exp375のfixed13 selector実装を構成参照元にし、candidate loaderだけを
exp374 Student-t HMMの列契約へ置き換える。exp374予測は全train unknown suffixを
各wellの既知prefixだけから生成済みで、unknown truthはprediction SHA freeze後に
付与されている。この保存済み予測をglobal key index化し、exp263の各selector
outer foldへrepartitionしてfixed12 bank末尾へ追加する。

Student-t候補のnative confidenceにはposterior std、well単位HMM log-likelihood、
log-likelihood / evaluation rowsを使う。selector scoreをfreezeした後だけ、
H512/whole-well add-one oracle headroomを非gating診断として計算する。

## 実験範囲

- 対象実験: `exp388_exp374_fixed13_dual_selector_on_exp264`
- Route: `ensemble`
- 親実験: selector=`exp264_exp263_candidate_confidence_dual_selector`、
  candidate=`exp374_exp209_student_t_exact_hmm_emission`
- 変更する変数: primary candidate inventoryをfixed12から
  fixed12 + `student_t_exact_hmm`へ変更
- 固定する変数: selector objectives/folds/sampling/LightGBM、raw-test-safe
  context、fixed fallback 7候補、parent score、gate閾値
- 実行: Stage A + Stage Cのみ、1 variant × 2 objectives × 5 outer ×
  4 inner = 40 CPU boosters。parent/control再学習0、GPU booster 0

## 再現性設計

- seed policy: global seed 42とimmutable key由来のstable sampling。
- stochastic 処理: selectorのLightGBM row/column samplingとcandidate-long
  samplingのみ。exp374候補生成は保存済み決定的出力を読む。
- PF/Beam / likelihood-PF / seed bagging: 本実験では再生成しない。
- 並列処理と乱数: samplingをfit前にstable keyで固定し、worker内global RNGを
  使用しない。LightGBMは`deterministic=true`,`force_col_wise=true`。
- CPU/GPU runtime: CPU 8 threads、GPU不使用。bitwise deterministic anchorは
  主張せずmodel SHAとprediction SHAを実行ごとに記録する。
- train cache: exp374入力raw gzip SHA
  `ea6f95334b2d75ab8c96f705c453f66b856397fd222770cd15f3ae0b7fef221e`、
  decompressed content SHA
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
  を固定する。
- model/prediction/submission: feature schema/content、40 model manifest、
  outer-valid score SHAを保存する。submissionは生成しない。
- Kaggle bootstrap: prepare後にmetadataと埋込configのkernel source、
  CPU/internet設定を確認する。

## リスク

- リークリスク: exp374のtruth/error列はファイルに存在しないが、
  loader usecolsをallowlistへ限定し、truth load count 0をmanifest化する。
- CV/LB不一致: exp374単独の平均改善とtail悪化が混在する。selector gateで
  pooled/fold/scope/by-wellを評価し、単独改善だけでは昇格しない。
- ランタイム/メモリ: 3.78M行×13 candidateのouter-valid long scoreは約49.2M行。
  exp375と同じchunk/capを維持する。
- 再現性: Kaggle CPU/LightGBMの環境差はmodel SHAで追跡し、
  保存済みexp374 decompressed SHA不一致時は即停止する。
