# exp260_matched_early_late_attribution_on_exp244 セッションノート

## 目的

exp244 mixed augmentationの効果をearly-only / late-onlyへmatched attributionし、late viewがearly-onlyの
悪化を独立に補償するか、またworst-well崩壊がどちらの方向に由来するかを確定する。

## 現在の状態

- Route: `ml_model`
- 状態: train-side attribution完了・不採用
- CV / LB: late-only 8.489116155 / 未提出
- inference / submission: 禁止

## 2026-07-16 設計・承認

### 固定条件

- parent: `exp244_bidirectional_prediction_start_pseudotail_augmentation` v4。
- official: 3,783,989 rows、weight 1.0。
- early-only: `m1000/m250`、1,537 views / 384,250 rows、weight 0.5。
- late-only: `p250/p1000`、1,544 views / 385,907 rows、weight 0.5。
- sampled row、380-feature schema、cache v1 SHA、exp218互換GroupKFold、3 LightGBM configs、
  early stopping、GPU mode、official-start validation、stress surface、guardをexp244から固定継承。
- outer-valid source well由来pseudo rowは対応foldのtrainから除外する。
- frozen exp218 / exp244 mixed OOFは比較入力であり再学習しない。

### GPUコスト

- active variants: 2 (`early_only`, `late_only`)
- LightGBM configs: 3 / variant
- folds: 5
- boosters: 15 / variant、合計30
- parent/control再学習: なし
- inference prediction / submission: なし

前turnで30 boostersと別途承認が必要なことをユーザーへ提示し、2026-07-16の
「これに進んでください。同じ条件でearly-onlyとlate-onlyを分けるmatched attributionです」を
上記固定計算量の明示承認として記録する。

### 実装

- `docs/legacy/steering/20260716-exp260-matched-early-late-attribution-on-exp244/`を作成。
- exp260実験ディレクトリを作成。
- exp244 official + 4 pseudo cacheを1回だけmemmapへstreamするJupytext trainを実装。
- variantごとにoffset maskを適用し、各foldでouter-valid source wellを除外する。
- exp218 frozen OOF decompressed SHA `5f3fc951...2976`をhard assertion。
- exp244 mixed OOF decompressed SHA `3c460056...e98b`、model manifest SHA
  `d93612c1...b830`、row ID、target、raw exp218、fold identityをhard assertion。
- variant別30 models、training metrics、OOF、distance/hidden/fold/by-well、importance、summary、
  model/prediction SHAを保存する。
- notebook上でconfig、approval、入力、variant、fold学習、metrics、生成物を追える構造にした。
- parent exp244 integrated notebookは949 lines / 9章、exp260 trainは768 lines / 7章。cache streamingと
  generic evaluationはSHA固定したparent helperへ残し、variant選択、30-booster学習、mixed identity、
  attribution metrics、artifact orchestrationはexp260 notebookへ展開した。
- Jupytext変換 / `--test`、py_compile、ruff、strict experiment validation: pass。

## 再現性

- seed/mode: exp218 `gpu_repro_guard_dp_threads8`を固定継承。
- mode guard: GPU、double precision、deterministic、force_col_wise、threads/jobs 8をassertする。
- 新規stochastic feature generation / PF / Beam / seed bagging: なし。
- input: exp244 v1 cache manifest/schema/request/content SHAをpin。
- gzip OOFはdecompressed content SHAを主証拠にする。
- GPU LightGBM rerun一致は未確認のためdeterministic anchorとは呼ばない。
- test regeneration、submission SHA: 対象外。

## 2026-07-16 Kaggle GPU train v1 push

- canonical kernel: `kentookumura/exp260-matched-early-late-exp244-train` v1。
- URL: https://www.kaggle.com/code/kentookumura/exp260-matched-early-late-exp244-train
- push成功、run-on-pushで実行開始。
- pull metadata: id_no `127431158`、machine shape `Gpu`、private、internet off。
- input kernel sources: exp239 official cache、exp244 offset cache 4本、exp218 train artifact、
  exp244 mixed train artifactの計7本。
- bootstrap: 12 files。exp260 config、parent cache/evaluation helper、exp218 helper/config、hidden-like
  assignmentを含み、2 variants / 3 configs / 5 folds / 30 boosters / control再学習0を確認。
- package configと正のconfigはbyte一致。id/title slugも一致。
- status: `KernelWorkerStatus.RUNNING`。
- template validation: pass。repository pytestは44 passed / 2 failed。失敗2件は既存exp251 configの
  `raw_test_safe` / `copcf_`旧契約を期待するテストで、exp260の変更とは無関係。exp260 strict validation、
  Jupytext、py_compile、ruffはpassしている。
- inference / submission: なし。

## 2026-07-16 Kaggle GPU train v1完了

### 実行確認

- status: `COMPLETE`。Traceback / Error / Exceptionなし。
- 2 variants / 3 configs / 5 folds = 30 boostersを完走。parent/control再学習0。
- elapsed: 26,501.519秒（約7時間21分42秒）、peak RSS: 20,371.730 MiB。
- sklearn feature-name warningとnbconvert warningだけで、学習結果に影響する例外はない。
- full output archiveは取得せず、metrics / training metrics / by-well / summary / logだけをselective downloadした。

### matched attribution

| 候補 | overall RMSE | raw exp218差 | mixed差 | 改善fold |
| --- | ---: | ---: | ---: | ---: |
| exp218 raw | 8.475793752 | - | +0.003414021 | - |
| exp244 mixed | 8.472379731 | -0.003414021 | - | 3/5 |
| early-only | 8.513933814 | +0.038140063 | +0.041554084 | 2/5 |
| late-only | 8.489116155 | +0.013322404 | +0.016736425 | 2/5 |

- late-onlyはearly-onlyより`0.024817659` ft良いが、raw exp218 / mixedのどちらにも届かない。
- early-onlyは1000+を`+0.036479760`悪化させる一方、hidden-like spatial / typewell-purgedを
  `-0.332981056 / -0.325110885`改善した。
- late-onlyは1000+を`+0.013735077`、hidden-like 2面を`+0.052783536 / +0.058460626`悪化させた。
- rawからの改善foldは両variantとも2/5。

### well安定性

- early-only: 394改善 / 379悪化、+2 ft超悪化17 wells。worst `059c8f24`は
  `7.655552450 -> 26.278710530`（`+18.623158080`）。
- late-only: 388改善 / 385悪化、+2 ft超悪化2 wells。worst `7850c72e`は
  `18.002609793 -> 21.411061241`（`+3.408451448`）。
- late-onlyはearly-onlyより安全だが、worst-well +2 ft以内guardを通らない。

### guardと解釈

- early-only: overall / 1000+ / 3-fold / worst-wellがFAIL、adoption false。
- late-only: overall / 1000+ / hidden-like 2面 / 3-fold / worst-wellがすべてFAIL、adoption false。
- `late_independent_compensation_supported=false`。
- mixed exp244だけがoverallをraw比`-0.003414021`改善し、hidden-like改善もearly-only単独より大きい。
  方向単独の効果を足した説明とは整合せず、両方向同時学習の非加法的相互作用が示唆される。ただし
  mixed自身もworst `059c8f24 +16.650567`でguardを失敗しているため、採用や救済探索の根拠にしない。
- hidden-like改善とworst `059c8f24`崩壊はいずれもearly-onlyで再現される。同wellのlate-onlyはraw比
  `-0.434121538`なのでlateは崩壊源ではなく、mixed内で部分緩和している可能性はある。ただしlate-onlyの
  overall / stress guardが不通過なので、独立補償とは区別する。

### artifact identity

- training metrics: `62c8031481fa1e7a2eb383fb6c5e952e6f8cc895358df0b4d71f7f783e4eba3e`
- metrics: `104c021e8a26fe6808c5d2d8a261a6109f3766ad48f98e7fe3100991bb3efaa3`
- by-well: `1ee0ef90e6f5d44faff93cd10adf7ec2c9e40d47f118a0b523d883994316e989`
- feature importance: `936182183560474100bda553fd129f700f0e5f6a0e31552df7e3698ebbbe391e`
- feature schema: `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`
- model manifest: `cdfbda0d35f9140f82c0c65374697e4d416c511d7729b715ae2779df84fce00b`
- OOF decompressed: `3e55541d02f4221fc0ea8b48af86f2c36b7368d2f20bb5c39cb489387eaee28a`

### 判断

両variantを不採用とする。matched attributionでlateの独立補償を否定できたため、mixed weight grid、
offset / sampling変更、risk gate、current-test inference、submissionへ進まない。prediction-start augmentation
branchは終了し、低価値な救済候補を新規backlogへ追加しない。

## 次

なし。prediction-start augmentation branchは完了・不採用として閉じる。
