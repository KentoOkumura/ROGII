# exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector

## 状態

Kaggle train v2 完了。提出なし。

## 仮説

exp181 では cluster-outlier gate を通した typewell / spatial prior signal が PF/Beam/likPF 候補上で有効だったが、direct correction としては worst-well regression が大きかった。補正値を予測へ直接足さず、exp157/158 candidate selector の confidence feature として add-only すれば、selector が cluster 外れ well で候補品質をより正しく読める可能性がある。

## 検証方針

exp157 と同じ 8候補、dense enrichment、GroupKFold を維持し、cluster-outlier flags、prior-candidate delta、prior std/count/neighbor、gate flag、well gate ratio、c20/c40 correction magnitude を score feature として追加する。新しい candidate error ranker の OOF predicted-error surface を exp158 と同じ Viterbi grid に渡す。

比較基準は `likpf_mean` RMSE 11.594897672、exp157 row-wise RMSE 10.795799837、exp158 continuity RMSE 10.789163253。

## 実行内容

- active selector variant: 1
- LightGBM configs: 3
- folds: 5
- boosters: 15
- Viterbi variants: 180
- control / parent retraining: なし
- runtime: Kaggle CPU
- kernel: `kentookumura/exp183-copcf-train` version 2

v1 は candidate-long feature 生成中に `DeadKernelError` で落ちた。v2 では long-model train/eval cap を 120k rows/fold、full-valid OOF prediction を 50k row chunk に変更して完了した。

## 判定基準

global OOF が exp158 continuity を上回り、path switch、worst-well regression、distance bucket、exp115 hidden-like subgroup が壊れていない場合だけ follow-up を検討する。positive でも direct correction / inference port / submit はこの実験では行わない。

## 所見

best Viterbi は `viterbi_sw200_bias000_jw100_jf025_d0075_std999999_md0000_seg001`。

- RMSE: 10.601481774
- MAE: 6.386571251
- within10: 0.792418794
- oracle label accuracy: 0.266536716
- path switches: 5,650 / 1.493 per 1000 rows
- delta RMSE vs exp158 continuity: -0.187681479

direct correction ではなく selector confidence feature として使う方針は train-side で supported。v2 は OOM 対策で model fit 条件が exp157/158 と異なるため、inference port / submit 前に raw-test parity、worst-well / bucket / exp115 subgroup 詳細、必要なら高メモリまたは split train を確認する。
