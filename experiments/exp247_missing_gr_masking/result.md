# exp247_missing_gr_masking 結果

## 仮説

exp221 exact HMMでraw horizontal GR欠損rowのGR emissionだけをstate-neutralにすると、長い欠損区間へ補間値が作る人工的な観測を除き、固定LGB unaryとtransitionによるposteriorが欠損区間と直後で改善する可能性がある。

## 設定

- 親: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- Route: `ensemble`
- fixed control: exp221 train v3、`sigma=20/lambda=0.50`
- active variant: `mask_only` 1本
- LightGBM config / fold / booster: `0 / 0 / 0`
- parent/control再学習・再生成: なし
- 変更: raw evaluation GR欠損rowのGR unaryを0にする1点のみ
- inference / submission: 無効

## 実装結果

| 項目 | 値 |
| --- | --- |
| train notebook | 実装済み |
| train-side-only inference guard | 実装済み |
| synthetic GR unary assertion | 実装済み |
| raw train/test missing-run readout | 実装済み |
| fixed-control paired diagnostics | 実装済み |
| py_compile / Ruff | pass / pass |
| Jupytext convert/test | pass |
| strict experiment/template validation | pass / pass |
| canonical Kaggle CPU package | pass |
| Kaggle full audit | CPU train v1完了 (`version=1`, `id_no=127064272`) |
| Public / Private LB | - / - |

## Kaggle結果

Kaggle CPUで773 wells / 3,783,989 rowsを11,409.172秒で完走した。固定controlの再学習・再生成はなく、1 variant、LightGBM 0 config / fold 0 / booster 0である。

| slice | control RMSE | mask RMSE | ΔRMSE | ΔMAE |
| --- | ---: | ---: | ---: | ---: |
| overall | 8.327728213 | 8.322894658 | -0.004833555 | +0.042731938 |
| raw GR missing | 8.258335070 | 8.253294362 | -0.005040709 | +0.066277985 |
| missing run 1-31 | 8.255837996 | 8.251989133 | -0.003848864 | +0.067176896 |
| post-gap 1-128 | 8.373273302 | 8.368494772 | -0.004778531 | +0.032130143 |
| post-gap 129-256 | 7.366781488 | 7.364965862 | -0.001815626 | -0.000997128 |
| post-gap 257+ | 5.955891709 | 5.960079203 | +0.004187495 | +0.004025675 |
| distance 1000+ | 9.130472317 | 9.124143987 | -0.006328330 | +0.048925680 |
| hidden-like spatial | 9.572230856 | 9.578192456 | +0.005961600 | +0.047406615 |
| hidden-like typewell-purged | 9.545375249 | 9.540505814 | -0.004869435 | +0.040814351 |

- by-well RMSE: 改善386 / 悪化387、median ΔRMSE +0.000063584 ft。
- worst well `c66be2b8`: 7.491665913 -> 10.068646557、ΔRMSE +2.576980644 ft。longest missing runは8 rows。
- prediction / std finite coverage: control・maskとも3,783,989 / 3,783,989 rows、773 / 773 wells。
- maskは3,782,870 rowsでcontrolから変化し、divergenceは1,797 segments、最長10,052 rowsだった。

## 再現性

- seed policy: `no_new_rng_exact_hmm_deterministic_ablation`
- new stochastic process: なし
- runtime: Kaggle CPU、GPU/internet disabled
- input/output SHA: `SESSION_NOTES.md`と`metrics.json`に記録。downloadしたgroup/by-well/finite/divergence artifactはnotebook summaryのSHAと一致
- model/submission SHA: 対象外

## 解釈

maskはraw GR availabilityだけで決まり、true TVT、error、hidden-like roleを参照しない。欠損rowでもLGB unaryとtransitionは通常どおり残すため、「GR更新なし」という意味を全missing runで統一した。契約どおりの実装でfinite coverageも維持したが、RMSE gainは0.0048 ftに留まり、MAEはoverall / missing / short-runで悪化した。hidden-like spatialの悪化と+2.577 ftのworst-well回帰もあり、改善386 wells対悪化387 wellsで頑健な利得ではない。

## 採否

`reject_uniform_missing_gr_mask_close_branch`。一律missing-GR maskは不採用とし、短runの小さなaggregate gainを根拠にrun-length gateやthreshold gridへ進まない。raw-test inference、selector、submissionも行わない。

## 次

このbranchは完了・不採用として閉じる。この結果だけから派生backlogは追加しない。
