# exp300 exp264 vs exp274 well / selector readout セッションノート

## 目的

exp264 corrected Stage D v3 OOFを前のsubmitted ML anchor exp274 raw CatBoost OOFと比較し、悪化wellの一覧・特徴・selectorの選ばれ方を再現可能な診断生成物としてリポジトリへ保存する。

## 実行契約

- Route: `ml_model`
- 実行variant: 0
- LightGBM config: 0
- 学習fold: 0
- booster: 0
- candidate再生成: 0
- GPU: 0
- inference / submission: 0 / 0
- 使用surface: corrected Stage C v6 strict nested outer-valid / corrected Stage D v3 selector compact add-only
- 無効化済みの旧Stage C/Dは使用しない。

## 変更点

- 先行調査のOOF、by-well、readout scriptをexp300配下へ移設した。
- well特徴、severity、row bucket、selector behaviorを別々の再実行可能なscriptへ整理した。
- corrected Stage C v6 candidate-long scoreだけを選択取得し、row-level switch windowと前候補維持run反実仮想を追加した。
- 同じcandidate-longのactual error oracle top1を用い、selector ranking regretとStage D downstream effectを加法分解した。
- canonical notebookをJupytext sourceから作成し、診断専用・inference禁止契約を明記した。

## コマンドログ

2026-07-20にlocal CPUで保存済みKaggle OOFを決定的に再集計した。Kaggle train/inferenceはpushしていない。

```bash
.venv/bin/python -m py_compile \
  experiments/exp300_exp264_vs_exp274_well_selector_readout/*.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/well_feature_readout.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/threshold_readout.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/row_readout.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/selector_readout.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/selector_switch_readout.py
.venv/bin/python experiments/exp300_exp264_vs_exp274_well_selector_readout/selector_oracle_attribution.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp300_exp264_vs_exp274_well_selector_readout/exp300_exp264_vs_exp274_well_selector_readout_train.py \
  experiments/exp300_exp264_vs_exp274_well_selector_readout/exp300_exp264_vs_exp274_well_selector_readout_inference.py
```

oracle attributionの初回float64実行は完了したが、mechanism列追加後の再実行2回は集計時にexit 137となった。SSE恒等式を変えず、保持する派生row配列を削減してSSE配列をfloat32、集約をfloat64に変更した後に全件再実行し、RMSEとMSE主値が表示桁で一致、恒等式誤差0、exit 0を確認した。

## 結果

- 3,783,989 rows / 773 wellsのcoverageを確認。
- exp274 `8.183503547`、exp264 `8.460811238`、差`+0.277307691 ft`。
- exp274比で387 well悪化、386 well改善。`>1 / >3 / >5 ft`は`194 / 73 / 40 well`。
- exp264/exp274 fold一致は710 well、不一致63 well。一致群でも`+0.279430401 ft`悪化。
- selector primary hard top1は`8.652531956`、Stage D finalは`8.460811238`。hard top1はfinal predictionではない。
- 全体のswitchは205,264（54.245/1000 rows）。切替rowは全rowの`5.42%`、正のSSE悪化の`3.19%`だった。
- `>3 ft`悪化73 wellでは切替±5行がrowの`19.85%`、正のSSE悪化の`14.26%`。悪化の`85.74%`は切替から6行以上離れた領域にあった。
- 新候補と前候補維持のswitch-run oracle比較はglobal MSE `-4.618103`で新候補が改善。`>3 ft`群のwell集約では36/73がnet悪化、37/73がnet改善、中央値`-0.074948`だった。
- hard pathで有害なswitch-runと有益なswitch-runは、`>3 ft`群のStage D正のSSE悪化を`49.18% / 50.80%`に二分した。candidate switch自体をStage D悪化の主因とは判定しない。
- `>3 ft`悪化73 wellのoracle候補 / selected hard / Stage D final / exp274 RMSEは`5.262585 / 17.166280 / 16.653182 / 10.796409`。候補集合には良い候補があるがselector rankingが外していた。
- 同群のMSE分解はoracle-vs-exp274 `-88.867648`、selection regret `+266.986386`、Stage D-vs-selected `-17.352698`、final-vs-exp274 `+160.766040`で恒等式誤差0。主因はselection regret、Stage Dは集約上緩和側。
- 67/73 wellでselected hardがexp274より悪かった。Stage Dは41 wellで緩和、32 wellで追加悪化。後者のうち6 wellはselected hard自体はexp274より良く、Stage D単独failureだった。
- tie-aware oracle正解率は`>3 ft`群`7.917%`、その他`12.983%`。悪化群1000+では`6.555%`、absolute regret平均`11.220 ft`。
- selection-regret SSEの`52.3%`はoracle候補が`exp226_k16`の誤rankingで、最大pairはselected Self-GR/LikPF vs oracle K16だった。
- selected Self-GR/LikPF / oracle K16は悪化群13,331行。selected MAE/RMSE `34.146/36.891 ft`、K16 `6.082/9.363 ft`、selection-regret MSE `1273.287`。
- Beam誤選択は悪化群18,947行、全row比`5.256%`でその他群`2.523%`の`2.083x`。その他群率からの期待9,096行に対しexcess約9,851行、Beam選択内誤選択率`99.22% vs 90.34%`。

## 再現性メモ

- primary aggregation seed policy: no RNG
- auxiliary logistic AUC / spatial KMeans: `random_state=42`。主結論・anchor判断には使わない。
- CPU/GPU runtime: local CPU / GPUなし
- Kaggle kernel id/version: 新規runなし。sourceは既存exp264/exp274 Kaggle生成物。
- exp264 Stage C outer-valid score SHA: `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- exp264 Stage D OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- exp274 OOF raw gzip SHA: `47b1319d50467faca9ceecd0eb70d74bd412b376d26512adc4d96e4f092101ed`
- exp274 OOF decompressed content SHA（主証拠）: `56a7f1bbeef0e703af74650d41e546343aa6f499a71b584f1a16992a5209aa55`
- model manifest / model SHA: N/A（学習なし）
- new prediction SHA: N/A（予測生成なし）
- submission SHA: N/A（提出生成なし）
- output SHA: `artifacts/selector_readout_summary.json`、`artifacts/selector_switch_readout_summary.json`、`artifacts/selector_oracle_attribution_summary.json`へ記録。
- deterministic anchor: false。cross-experiment posthoc診断でありanchor更新に使わない。
- migration cleanup: `/tmp/exp264-well-analysis`、`/tmp/exp264-analysis-exp274`、`/tmp/exp264_row_analysis.py`は削除済み。重複`artifacts/preliminary`も削除し、final artifacts/source inputsだけをexp300配下に保持した。

## 注意

- exp274はsubmitted ML anchorだがtrain-side rejectedであり、新しいCV anchorではない。
- 63 wellでouter foldが異なるため全体比較はcross-experiment診断。matched 710 wellでも同じ悪化方向だが因果ablationとは主張しない。
- target/oracle feature、悪化label、同じOOF上で見つけた閾値をrouterへ流用しない。
- candidate dominanceはwell内hard top1の最頻候補。Stage D finalの直接routing結果ではない。
- exp274にはselectorがない。「exp274から候補が変わった」とは定義せず、exp264内のhard-top1切替とStage D finalのexp274比悪化を比較した。
- 前候補維持counterfactualはactual TVTを使うoracle hard-path診断であり、deployable switch policyでもStage D finalの因果ablationでもない。
- oracle候補もactual TVTを使う候補集合内の上限診断であり、そのままrouting labelやfeatureへ利用しない。

## 次のアクション

1. 既存の高優先0-booster `exp276_corrected_exp264_parent_revalidation`で、事前固定済みtarget-free risk familyが今回の高confidence / low-switch / Beam・LikPF系regimeをfold-stableに捉えるか確認する。
2. exp300だけを根拠にswitch suppression、hard fallback、selector weight、candidate除外、threshold gridを行わない。
