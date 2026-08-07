# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `raw_hmm_likpf_missing_gr_observation_neutrality_ablation` を
`exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation` として実装する。

最初の Stage 1 では `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` の
raw typewell-GR exact HMM を固定 control とし、raw horizontal GR が欠損している評価 row
だけ GR emission contribution を state-neutral な 0 にする。Stage 1 の事前固定 guard を
全通過した場合だけ、別承認・別 Kaggle run で exp072 likelihood-PF の同じ観測中立化へ進む。

## 制約

- Route: `pf_beam`。
- scientific parent / fixed control は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
  の raw exact HMM とし、exp221 LGB unary HMM や exp223 self-GR HMMを親にしない。
- exp247 は欠損maskの実装・診断参照に限定する。LGB unary、LGB OOF、exp221 predictionは使わない。
- 新規 variant は Stage 1 `raw_hmm_missing_gr_neutral` 1本。LightGBM config 0、fold 0、booster 0、
  parent/control再学習・再生成0とする。
- control は exp209/exp205 保存済み raw exact-HMM cacheを固定入力として読み、ID、row、well、
  decompressed content SHAを検証する。
- 欠損判定は補間前のraw horizontal `GR` non-finiteだけから作る。true TVT、error、oracle、
  focus-well ID、hidden-like roleはmask生成・HMM state更新に使わない。
- GR補間値、prefix calibration/sigma、grid、rate lattice、transition、initial rate、HMM grammarを固定する。
- observed rowではexp209 GR emissionと同一、raw-missing rowでは全stateのGR emissionを厳密に0にする。
- HMM Stage 1 と PF Stage 2 を同じvariantや集約scoreとして混ぜない。
- run-length gate、補間法、sigma/temperature、particle数、seed数、process noise、initial rateを探索しない。
- Stage 1 guard通過前はlikelihood-PF生成、raw-test inference、submissionを無効にする。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- raw-missing rowでGR emissionが全state 0、observed rowでcontrol計算と一致するsynthetic assertionがある。
- fixed control とvariantのID coverage、row count、well count、finite prediction/std coverageが一致する。
- overall、observed/missing、missing-run長、post-gap、distance、`1000_plus`、exp115 hidden-like 2群、
  focus well `11d0f5ac`、by-well、worst-well、posterior std/loglik、divergence segmentを保存する。
- Stage 1 guardを実行前にconfigへ固定し、結果後にthresholdを変更しない。
- `config.yaml`、Jupytext train/inference source、通常`.ipynb`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`を作成し、Jupytext、py_compile、Ruff F821、strict experiment validationを通す。
- Kaggle package作成前にactive variant 1、LightGBM 0 config、fold 0、booster 0、control再生成なし、
  PF Stage 2 disabledを`SESSION_NOTES.md`へ記録する。
- deterministic submission anchorとは扱わない。kernel version、input/control/output SHA、runtime/thread設定を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
