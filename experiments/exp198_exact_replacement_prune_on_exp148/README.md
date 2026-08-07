# exp198_exact_replacement_prune_on_exp148

## 状態

- Route: `ml_model`
- Status: `submitted_public_lb_7_930_not_adopted`
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960

## 仮説

exp148 の active feature surface には、後追いで追加した learned-likelihood / U-projection / public replay 由来列のうち、既存特徴量と完全一致、符号反転、または定数になっている列が残っている。これら 17 列だけを削ると、不要な重複を減らしつつ exp148 anchor の性能を保てる、または改善できる可能性がある。

## 検証方針

exp148 と同じ exp072 base cache、exp092 U-projection、exp145 learned-likelihood cache、GroupKFold by well、LightGBM `lgb0/lgb1/lgb2` を使う。active variant は `drop_exact_replacements_17` のみで、control は再学習しない。Kaggle GPU train は 3 configs x 5 folds = 15 boosters。

削除対象は `corr_prune_sanity_readout_on_exp148` で固定した 17 列だけにする。feature count は exp148 の 294 から 17 減った 277 を期待値として fail-fast で確認する。

## 所見

Kaggle train v1 `kentookumura/exp198-exact-replacement-prune-exp148-train` version 1 は完了。`lgb_mean` pooled RMSE は 8.457923653 で、exp148 GPU train 8.501281182 から -0.043357529 改善した。feature count は 277、join coverage は 3,783,989 rows / 773 wells で pass、削除対象 17 列は schema に残っていない。

distance bucket は `000_050` / `050_100` / `1000_plus` が改善し、`100_250` / `250_500` / `500_1000` は小幅悪化。well 単位では 423 wells 改善、350 wells 悪化、最大悪化は `b37fd114` の +1.022149086 RMSE。

Kaggle inference v4 `kentookumura/exp198-exact-replacement-prune-exp148-inference` も完了。14,151 rows、fallback 0、submission SHA256 `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`。`submission.csv` は sample と header / row count / id order が一致し、重複 ID、欠損、NaN、Inf-like value はない。submit-check は PASS。

scoring ref `54354847` の Public LB は 7.930。exp148 GPU inference v7 7.960 と exp193 7.946 は上回ったが、現 ML route anchor の exp148 CPU runtime 7.921 には届かないため未採用。
