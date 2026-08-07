# exp245 セッションノート

## 2026-07-14 設計・実装

exp238 inference で45 context特徴が全行NaNになった問題を調査した。内訳は、
test-equivalent generatorを持たないtrain OOF専用 `copcf_*` 41特徴と、exp226推論が
最終`tvt`だけを保存していたため欠落した診断4特徴だった。hidden testのwell数が増えても
schema欠落は解消しないため、別実験exp245でtrain/test parityを修正する。

実装内容:

- `ranker.cluster_prior_features` をselector trainで無効化し、`copcf_*` をschemaから除外。
- exp226 full-train/current-test predictionを一度だけ実行し、候補値と診断4列を同じ
  `PredictionResult`から生成。
- context 143列、候補11列についてmissing/nonfinite 0をfail-fastで強制。
- train summaryへcontext schema SHA、20 model SHA、fold manifest SHA、nested scoreの
  decompressed SHAを保存。
- inferenceはtrain guard通過済みsummaryと20 saved selectorsを必須とし、学習を行わない。

## Kaggle実行コスト確認

- variant数: 1 (`parity_safe_selector`)
- selector config数: 1
- fold数: outer 5 × inner 4
- selector boosters: 20
- final LightGBM boosters: 0
- 合計 boosters: 20 CPU
- 親/control再学習: なし
- GPU使用: なし

この時点ではKaggleへpushしていない。`config.yaml`の
`runtime.kaggle.push_requires_user_approval: true`を維持し、明示承認後にselector trainを実行する。

## 静的検証

- `.py`構文検査: pass
- Ruff `F821`: pass
- Jupytext notebook変換: pass
- notebook round-trip test: pass
- `make validate-exp EXP=exp245_selector_context_parity_on_exp238`: strict pass
- Kaggle train/inference package生成: pass
- bootstrap manifestにexp237 helper/config、exp218 replay source/config、exp226 source/configを確認

## 次段

exp245 train guard通過とcurrent-test missing/nonfinite 0の確認後、別実験でselector top1候補の
直接採用を評価する。exp245ではadd-only final LightGBMもsubmissionも作らない。

## 2026-07-14 Kaggle CPU train実行承認

ユーザーからKaggle実行の明示承認を受領した。push前の再確認結果は以下。

- active variant: `parity_safe_selector` 1件
- selector config: 1
- folds: outer 5 × inner 4
- selector boosters: 20 CPU
- final boosters: 0
- 親/control再学習: なし
- internet: disabled
- canonical kernel: `kentookumura/exp245-selector-context-parity-on-exp238-train`

canonical titleを`exp245 selector context parity on exp238 train`へそろえ、
`run_on_push=true`で同kernelへpushする。

## 2026-07-14 Kaggle CPU train v1開始

- push時刻: `2026-07-14 12:10:28 UTC`
- Kernel: `kentookumura/exp245-selector-context-parity-on-exp238-train` v1
- Kaggle id_no: `127057623`
- URL: `https://www.kaggle.com/code/kentookumura/exp245-selector-context-parity-on-exp238-train`
- runtime: CPU、internet disabled、20 selector boosters、final 0、control再学習なし
- package notebook SHA256: `eadde4c2299ed906fab248a7587f2cca615893a558ebb8ac41fcd66e194afffa`
- package metadata SHA256: `85bf341b607a06854ddc31e9547a17207ea461a176c0fa155af9b42e0294e566`
- `kaggle kernels pull ... -m`で同canonical kernelの存在を確認。
- push直後のstatus: `KernelWorkerStatus.RUNNING`

実行中のCLI logsは空になり得るため監視は行わない。完了連絡後に通常logsを取得し、
143 context、20 model、safety guard、schema/model/prediction SHAを監査する。

## 2026-07-15 Kaggle CPU train v1完了監査

- status: `KernelWorkerStatus.COMPLETE`
- notebook runtime: 約11,634.47秒（約3時間13分54秒）
- context: 143、`copcf_*` 0、exp226診断4列存在、missing/nonfinite 0
- models: outer 5 × inner 4 = 20、fold pair完全被覆、20 model SHA一致
- pooled selector top1 RMSE: 8.558916610
- pooled likPF fallback RMSE: 11.594897537、delta -3.035980927
- near `000_050`: 0.622346410、fallback比 -0.566531083
- `1000_plus`: 9.393873960、fallback比 -3.309116772
- worst well `fb03ae90`: +38.016696930 RMSE
- regressed wells: 215 / 773、+0.25超184、+5超28、+20超2
- guard: fail、`selector_guard_failed_inference_forbidden`

必要最小限のoutputとしてsummary、safety/by-well、fold/context/model manifest、20 saved
selectorを取得した。約1GBのnested score 5面は取得していない。fold/context/safety manifest SHAと
全model SHAを検証した。

exp238 selectorとのwell別delta相関は0.990883で、NaN特徴除外後もworst-well構造は同じ。
exp245 hard top1はexp218 finalより+0.083123、exp238 add-only finalより+0.622227悪いため、
無条件direct replacementには進まない。current-test parityだけを確認するaudit-only inferenceと、
outer-trainでgateを決めるwell-risk付きdirect auditを次候補とする。
