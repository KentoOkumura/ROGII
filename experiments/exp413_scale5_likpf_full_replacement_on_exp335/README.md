# exp413_scale5_likpf_full_replacement_on_exp335

## 状態

- ルート: ml_model
- 状態: Stage 0 / C / S / D PASS・hidden互換推論v4完了・Public LB 7.201
- CV: 7.884802794404715
- Public LB: 7.201
- Private LB: -
- Submit ID: 55080377
- 作成日: 2026-07-26
- 親実験: `exp335_signed_residual_meta_on_exp264`

## 仮説

exp404で得た`scale 5・gs×1.0` likelihood-PFは、exp072互換の
128 seed arithmetic meanよりdirect OOF RMSEを約`0.680376 ft`改善した。
このprimitiveを現行Public-LB reference anchor exp335の候補・特徴量・selector・
downstreamへ一貫して全面置換すれば、13候補目を追加せずML予測を改善できる。

## 変更点

- 固定12候補の`likpf_mean` slotを`likpf_scale_5_x1p0`へ同一IDで置換する。
- `likpf_mean`を使う4 pair/fixed formulaと、clean273、selector compact74、
  signed compact23を依存順に再生成する。
- 40 CPU nested selector + 20 CPU signed selector + 15 GPU downstream、
  合計75 boostersを新variantだけ学習する。saved exp335 control再学習は0。
- `gs×1.3`、scale 3/8/12、13候補化、旧meanのadd-only保持は行わない。

## 検証方針

- Fold: corrected exp264 / exp335と同じouter 5、selectorはinner 4。
- Group: `well_id`。
- Stratification: 親fold/sample keyを変更しない。
- Primary control: saved exp335 OOF RMSE `8.146107755881022`。
- Primary gate: pooled `>=0.03 ft`改善、3/5 folds nonworse、
  near/mid/1000+とhidden-like 2面の各delta `<=+0.02 ft`。
- Leakage Check: exp404 scale5 predictionをtruth/fold/hidden-like読込前にSHA固定し、
  outer-valid target/errorをselector fit・feature選択に使わない。
- by-well p95、worst well、`+1/+3/+5 ft`悪化well数は必須診断だが、
  LB-oriented experimentの自動停止gateにはしない。

## 実行入口

- 学習候補 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_compact_selfcontained_train.ipynb`
- Stage 0実行 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_replacement_preflight.ipynb`
- Stage C実行 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_selector_train.ipynb`
- Stage S実行 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_signed_selector_train.ipynb`
- Stage D実行 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_downstream_gpu_train.ipynb`
- 正規学習 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_train.ipynb`（placeholderのまま）
- 正規推論 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_inference.ipynb`
  （placeholderのまま）
- current-test推論 notebook:
  `exp413_scale5_likpf_full_replacement_on_exp335_current_test_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp413_scale5_likpf_full_replacement_on_exp335`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
- train-side helper、Jupytext候補、専用testは実装済み。Stage 0とStage Cは
  専用packageのKaggle version 3、Stage Sはversion 1でtechnical PASS。
  Stage C score/leakage、Stage S score gateもPASS。Stage Dはversion 2で
  technical / primary gateをPASS。current-test CPU推論version 3のcode submission
  ref `55078306`は公開testの固定row/well assertによりhidden rerunで失敗した。
  assertだけを動的sample/ID契約へ直したversion 4は完了し、Kaggle Notebook
  outputとしてsample互換`submission.csv`を生成・取得してsubmit-checkをPASSした。
  公開出力はversion 3と完全一致。ユーザー実施の修正版code submission
  ref `55080377`は`COMPLETE`、Public LB `7.201`となった。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 technical gate | PASS |
| rows / wells / partitions | 3,783,989 / 773 / 5 |
| unchanged / formula / old-mean parity max abs | 0.0 / 0.0 / 0.0 ft |
| Stage C technical / score / leakage | PASS / PASS / PASS |
| Stage C models / compact partitions | 40 / 25 |
| compact / outer-valid score rows | 18,919,945 / 45,407,868 |
| expected-error MAE / prior | 3.720634 / 5.700200 |
| within10 logloss / prior | 0.349579 / 0.499814 |
| within10 Brier / prior | 0.108064 / 0.160703 |
| Stage S technical / score / total | PASS / PASS / PASS |
| Stage S models / compact partitions | 20 / 25 |
| signed-residual pooled RMSE / prior | 8.291963 / 10.854996 |
| signed-residual pooled improvement / folds | 2.563032 ft / 5/5 |
| candidate別改善数 | 11/12 |
| Stage D technical / primary | PASS / PASS |
| Stage D models / unique SHA | 15 / 15 |
| saved exp335 / replacement RMSE | 8.146108 / 7.884803 |
| pooled gain / nonworse folds | 0.261305 ft / 5/5 |
| 最大scope delta | -0.019498 ft（上限+0.02 ft） |
| by-well p95 / worst delta | +1.228715 / +9.033462 ft（report-only） |
| control再学習 / PF well-runs | 0 / 0 |
| current-test inference | Kaggle version 4 COMPLETE / validation PASS |
| inference rows / wells | 14,151 / 3 |
| saved models / new boosters | 40 + 20 + 15 / 0 |
| scale5 changed rows / abs-delta parity | 14,093 / 0.0 ft |
| prediction file / decompressed SHA | `52ffb491...136` / `875a1334...dc4` |
| submission.csv / submit-check / user submit | Kaggle生成済み / PASS / ref 55080377 COMPLETE |
| code submission ref 55078306 | version 3 hidden rerun error |
| version 4 public output parity | version 3と完全一致 |
| CV | 7.884802794404715 |
| Public LB | 7.201 |
| Private LB | - |

## 所見

### 良かった点

- direct PF根拠は大きく、candidate widthを変えない反証可能な置換設計にできる。
- 旧meanをparity監査だけに隔離し、5 changed / 7 unchanged candidate、
  clean273、selector88→compact74、signed23、final370を依存順に再構築する
  fail-closed実装候補を作成できた。
- Stage Cの3 score指標はprior比で全5 folds改善し、40 model / 25 partition /
  leakage契約と全SHAを確定できた。
- Stage Sも20/20 model、25 signed partition、technical / score gateをPASSし、
  signed-residual RMSEをprior比`2.563032 ft`改善した。
- Stage Dは15/15 GPU modelを完了し、saved exp335比`0.261305 ft`改善、
  5/5 folds nonworse、固定5 scopesすべて改善でprimary gateをPASSした。
- current-testはstable per-well 128-seed trajectoryのtemperature-5へ全面置換し、
  14,151行を保存済み40/20/15 modelでCPU推論してrow/order/finite/SHAをPASSした。
- version 3のhidden非互換な固定row/well assertを動的sample/ID契約へ直し、
  科学条件を変えずversion 4を完了した。

### 悪かった点

- CV `7.884803`に対してPublic LBは`7.201`で、exp335 Public LB `7.517`を
  `0.316`改善した。
- `exp226_w500_50_50`だけはcandidate別prior比`0.123613 ft`悪化した。
- final TVTのby-well tailはp95`+1.228715 ft`、worst`+9.033462 ft`であり、
  train-side robust promotionではなくLB-oriented候補として扱う。

### リスク / 注意

- semantic slot名`likpf_mean`を維持して値sourceをscale5へ変えるため、
  manifestとstale old-mean拒否が必須。
- GPU LightGBMはbitwise deterministic anchorとは扱わない。
- exp335はtrain-side tail guard FAILのLB referenceであり、robust promotionとは分ける。

## 次

- hidden互換version 4のcode submission ref `55080377`はPublic LB `7.201`で完了。
  MLルートのPublic-LB referenceとして後続比較に使う。ただしtrain-side
  by-well tail悪化があるため、robust promotionとは引き続き分離する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
