# exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation

## 状態

Kaggle CPU Stage 1 version 1を完了した。neutral variantはcontrol RMSE 11.938287に対して13.348499（`+1.410212 ft`）へ悪化し、事前固定guardを不通過。likelihood-PF Stage 2はfail-closeし、推論・提出も未実行のまま本branchを終了した。

## 目的

exp209 の raw typewell-GR exact HMMを固定対照にし、raw horizontal GR が欠損する評価rowだけを観測中立化する。補間済みGRから作る対照のemissionと異なり、対象rowのGR log-likelihoodを全TVT stateで厳密に0にする。

## 仮説

raw GR欠損rowに補間GRを観測値として与えると、根拠のないtypewell alignmentへposteriorを寄せる。欠損rowのGR観測だけを中立化すれば、exp209のtransitionによる連続性を保ったままmissing区間の誤差を下げられる。

## 親と1変更

- scientific parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- fixed artifact ancestor: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- implementation/diagnostic reference: `exp247_missing_gr_masking`
- route: `pf_beam`
- variant: `raw_hmm_missing_gr_neutral` 1本
- 固定: exp209のgrid、rate lattice、transition、initialization、sigma、GR補間、score rows
- 変更: raw GR non-finiteの評価rowだけGR emissionを0にする
- 使用しないもの: exp223 self-GR、exp221/exp148 LightGBM unary、run-length gate、PF

保存済みcontrolはdecompressed SHA `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`を必須とし、exp209生成物とexp205参照物のどちらを解決しても同一contentであることを確認する。

## Stage 1 guard

全条件を事前固定し、1つでも失敗または診断欠落ならPFを閉じる。

- overall delta RMSE `<= -0.02 ft`
- raw-missing `<= 0`、observed `<= +0.02 ft`
- distance `1000_plus`とhidden-like 2群が各`<= +0.02 ft`
- worst-well regression `<= +0.25 ft`
- prediction/std finite coverage 100%、ID mismatch 0

guard通過時も`pf_stage_eligible=true`を記録するだけで、likelihood-PFの実行には別承認が必要。

## 検証方針

保存済みexp209/exp205 raw-HMM cacheを固定controlとして全train unknown-suffix rowをpaired比較する。overall、observed/missing、missing-run、post-gap、distance、hidden-like、focus well、by-well、posterior、finite coverage、divergenceを保存し、事前固定guardをfail-closeで判定する。visible testはmissing分布の記述だけに使う。

## 所見

raw GR missing 1,200,837 rowsではRMSEが11.948064から14.496321（`+2.548257 ft`）へ悪化し、observed rowsも`+0.846115 ft`悪化した。HMM smoothingにより変更が欠損row外へ伝播し、全773 wellsの3,783,349 rowsでpredictionが変わった。hidden-like 2面は`+3.462999 / +3.556545 ft`、worst wellは`+51.167455 ft`で、blanket observation neutralityは不採用とする。

prediction/std finite coverage 100%とID mismatch 0は通過し、control decompressed SHAも事前固定値と一致したため、実装不良ではなく科学仮説のnegative resultと判断する。詳細は`result.md`と`metrics.json`を参照。

## 次の扱い

本branchではlikelihood-PF Stage 2、run-length gate、sigma/temperature救済、mask grid、raw-test inference、submissionへ進まない。missing-GR処理を固定した既存`exp270_exact_hmm_posterior_mode_candidate_audit`を独立したraw-HMM候補監査として継続する。

## 主なファイル

- `config.yaml`: 親、HMM固定値、guard、成果物、PF fail-close契約
- `exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_train.py`: Jupytext正本
- `exact_hmm_smoother.py`: Numba exact HMMとraw-GR emission中立化境界
- `exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation_inference.py`: PF/inference停止契約
- `test_observation_neutrality.py`: emissionと親設定のtargeted contract test

## 実行コスト契約

active variant 1、LightGBM config 0、fold 0、booster 0、control再生成0、GPU false、PF false。Kaggle CPUでouter workers 2 / Numba threads 2を使う。

実測runtimeは19,573.731秒（約5時間26分14秒）。kernelは`kentookumura/exp269-raw-hmm-missing-gr-neutrality-train` version 1、id_no `127592556`。
