# exp300 exp264 vs exp274 well / selector readout

## 状態

- ルート: `ml_model`
- 状態: 診断完了（学習・推論・提出なし）
- 対象CV: exp264 corrected Stage D v3 `8.460811238`
- 比較CV: exp274 raw CatBoost OOF `8.183503547`
- 差: `+0.277307691 ft`（exp264が悪化）
- Public LB: 新規提出なし
- 作成日: 2026-07-20
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 比較対象: `exp274_catboost_final_regressor_swap_on_exp238`

## 目的

前のsubmitted ML anchorであるexp274に対してexp264のCVが悪化したwellを特定し、long-tail位置、raw/typewell特徴、予測不一致、corrected Stage C v6 selectorの候補・margin・switchとの関係を調べる。

## 仮説

exp264のexp274比悪化はlong-tail深部の一部wellへ集中し、target-freeなwell品質、予測不一致、selectorの候補family・margin・switchのいずれかに再現可能な偏りがある。

## 検証方針

- 保存済みOOF 3,783,989 rows / 773 wellsをID・truth・row数で照合する。
- exp274とcorrected exp264 Stage D v3をwell、distance、relative-tail、GR、residual bucketで比較する。
- corrected Stage C v6のprimary hard top1分布、well dominant候補、margin、switchを悪化severity別に読む。
- exp264/exp274でfoldが異なる63 wellsを分離し、matched 710 wellsを併記する。
- exp264 Stage C hard top1の切替点近傍と、切替後runで前候補を維持するoracle反実仮想を比較する。
- primary 11候補のactual-error oracle top1とselector top1を比較し、selection regretとStage D downstream効果を分解する。
- target-free特徴とoracle特徴を分け、posthoc診断をrouter承認に使わない。

## 変更点

- 新規学習は行わず、corrected exp264 Stage C/Dとexp274 raw OOFを結合する診断だけを追加した。
- `/tmp`にあった先行集計を実験配下へ移し、6本のreadoutコードと再現性SHAを正規化した。

## 結論

- 773 well中387 wellがexp274比で悪化し、`>1 / >3 / >5 ft`は`194 / 73 / 40 well`。
- 悪化は末尾側に集中し、distance `1000+`だけが`+0.327833 ft`悪化した。relative-tail q9/q10は`+0.598962 / +0.710069 ft`。
- target-freeなraw well特徴だけでは`>3 ft`悪化を識別できない（補助logistic AUC `0.495675`）。予測不一致RMSEは強いposthoc指標（AUC `0.946888`）だが、同じOOFで発見したためrouter承認には使わない。
- selectorは悪化wellで迷っているのではなく、むしろmarginが大きくswitchが少ない。Beam / LikPF / Self-GR-LikPF dominant wellが過剰代表だった。
- 主因はselectorの候補ranking失敗だった。`>3 ft`群はoracle候補RMSE `5.2626`、selected hard `17.1663`、Stage D final `16.6532`で、Stage Dは集約上selected hardを緩和した。
- MSE分解はoracle候補vs exp274 `-88.8676`、selection regret `+266.9864`、Stage D effect `-17.3527`、合計final vs exp274 `+160.7660`。67/73 wellでselected hardがexp274より既に悪かった。
- 候補切替点は主因ではない。`>3 ft`悪化73 wellで切替±5行はrowの`19.85%`、正のSSE悪化の`14.26%`で、悪化の`85.74%`はその外側にあった。
- 切替runのhard-path oracle比較では、新候補が前候補維持よりglobalに改善（MSE `-4.6181`）。`>3 ft`群も36/73 wellでnet悪化、37/73でnet改善だった。一律のswitch suppressionは支持されない。
- ただしhard top1は最終予測ではない。Stage D v3は74 compact selector特徴をdownstream LightGBMへadd-onlyし、この読み出しは候補score landscapeの診断である。

## 実行入口

- 正規診断 notebook: `exp300_exp264_vs_exp274_well_selector_readout_train.ipynb`
- Jupytext source: `exp300_exp264_vs_exp274_well_selector_readout_train.py`
- 集計: `well_feature_readout.py`、`threshold_readout.py`、`row_readout.py`、`selector_readout.py`、`selector_switch_readout.py`、`selector_oracle_attribution.py`
- inference notebookは禁止契約の表示だけで、prediction/submissionを生成しない。

## 主な生成物

- `artifacts/well_comparison_and_features.csv`: 773 wellのRMSE差と特徴
- `artifacts/material_worsened_gt3_vs_exp274_wells.csv`: `>3 ft`悪化73 well
- `artifacts/row_metric_*.csv`: distance / tail decile / GR / residual / fold集計
- `artifacts/selector_by_well.csv`: well差とselector manifestの結合
- `artifacts/selector_dominant_candidate_summary.csv`: dominant候補別悪化率
- `artifacts/selector_metric_effects.csv`: margin/share/switchの群間差
- `artifacts/selector_readout_summary.json`: source contract、SHA、主要数値
- `artifacts/selector_switch_window_summary.csv`: 切替±0/1/5/25/100行のStage D悪化寄与
- `artifacts/selector_switch_run_counterfactual.csv`: 新候補と前候補維持のrun別oracle SSE
- `artifacts/selector_switch_by_well.csv`: well別switch頻度、近傍寄与、前候補維持差
- `artifacts/selector_switch_readout_summary.json`: 切替寄与の主要数値と全入出力SHA
- `artifacts/selector_oracle_scope_summary.csv`: oracle候補・selector選択・Stage DのSSE分解
- `artifacts/selector_oracle_by_well.csv`: well別selection regretとStage D effect
- `artifacts/selector_oracle_confusion.csv`: selected候補×oracle候補の誤ranking集計
- `artifacts/selector_oracle_attribution_summary.json`: 候補選択品質の主要結果とSHA
- `artifacts/selector_*.svg`: selector診断図

詳細は`result.md`を参照する。

## 所見

- long-tail深部で正しい候補が存在するのに別候補を高confidenceにrankすることが主因で、候補の時系列切替自体ではない。
- raw well特徴だけの事前識別は弱く、単純hard gateは支持されない。
- Stage D finalは悪化群でもhard top1を集約上改善している。まずselector ranking failureを主対象とし、Stage D単独failureの6 wellは別診断として扱う。

## 次

既存の高優先0-booster `exp276_corrected_exp264_parent_revalidation`で、固定済みtarget-free risk familyのouter-fold再現性だけを監査する。
