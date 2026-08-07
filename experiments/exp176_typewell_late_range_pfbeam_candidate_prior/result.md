# exp176_typewell_late_range_pfbeam_candidate_prior 結果

## 状態

Kaggle train v3 完了。結論は positive だが、direct submit はしない。

best OOF は `lgb_candidate_error_ranker` で RMSE 10.641298。exp157 best OOF 10.795800 から -0.154502、exp158 best Viterbi 10.789163 から -0.147865、`likpf_mean_single` 11.594898 から -0.953600 改善した。

一方で row-wise selector の max path switch は 330.842 / 1000 rows とまだ高い。exp157 の 357.199 からは下がったが、exp158 continuity selector のような segment guard なしに inference port / submit する状態ではない。

## 実装

exp157 の supervised candidate ranker を親にし、candidate set と LightGBM 構成は維持した。追加した差分は typewell late-range prior feature のみ。

- raw train typewell から `typewell_min`、`typewell_max`、`typewell_span` を作成。
- horizontal well の visible prefix から `known_last_pct` を作成。
- 候補ごとに `candidate_pct = (candidate_tvt - typewell_min) / typewell_span` を作成。
- fixed lower bounds `0.50/0.60/0.70` と dynamic lower bounds `known_last_pct - 0.05/0.10` を feature 化。
- 候補は hard invalid / clip せず、LightGBM feature としてだけ渡した。
- v3 では candidate-long memory 対策として row-level `tlp_` feature を long-frame から除外し、candidate-long 用 `candidate_tlp_` feature は維持した。

## CV

| variant | mode | RMSE | MAE | within10 | oracle acc |
| --- | --- | ---: | ---: | ---: | ---: |
| lgb_candidate_error_ranker | oof | 10.641298 | 6.434563 | 0.791815 | 0.257439 |

比較:

- vs `likpf_mean_single`: RMSE -0.953600
- vs exp157 best OOF: RMSE -0.154502、MAE -0.042433、within10 -0.000689
- vs exp158 best Viterbi: RMSE -0.147865
- PF/ANCC selection rate: 0.380816
- max path switch / 1000 rows: 330.841857

## 実行履歴

v1 は papermill 起動時に `ValueError: No kernel name found in notebook and no override provided.` で失敗した。train / inference notebook に `python3` kernelspec を追加して v2 とした。

v2 は fold 0 の multiclass LightGBM が best iteration 100 まで進んだ後、candidate-long binary / error ranker の処理に入る前後で `DeadKernelError: Kernel died` となった。追加した 77 列の row-level `tlp_` feature を candidate-long frame に 8 candidates 分複製したことと、long frame 上の `DataFrame.replace()` による一時 copy が主なメモリ圧迫要因と判断した。

v3 では row-level `tlp_` feature を long-frame から除外し、`max_train_rows_per_fold` を 650000 から 300000 に下げ、`fit_impute()` を NumPy 配列上の処理に変更した。v3 は CPU で 9329.73 sec、Kaggle status `KernelWorkerStatus.COMPLETE`。

## 判断

`typewell_late_range_pfbeam_candidate_prior` は supported。backlog から外す。

次は exp176 の signal を exp158-style continuity selector または exp148/ML anchor への confidence feature として使う。row-wise selected path の direct replacement、hard invalid、clip、PF/Beam generation soft prior、submit には進めない。

Kaggle output archive は取得していない。CV、SHA、生成物パスは `kaggle kernels logs` の summary を根拠に記録した。
