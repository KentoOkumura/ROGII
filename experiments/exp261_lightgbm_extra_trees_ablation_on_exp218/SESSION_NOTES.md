# exp261_lightgbm_extra_trees_ablation_on_exp218 セッションノート

## 目的

exp218の回帰LightGBMへ `extra_trees=True` だけを追加し、同一feature/fold/configの保存済み
controlと比較する。親/controlは再学習しない。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train v1完了・guard fail・回帰variant不採用
- CV: parent `8.475793751656624` -> `extra_trees=True` `8.755217123821655`（`+0.27942337216503077`悪化）
- LB: 未提出
- inference / submit: 全guard不通過のため実行しない

## Kaggle train前コスト

| plan | active variant | LightGBM configs | folds | boosters | parent/control再学習 |
| --- | ---: | ---: | ---: | ---: | --- |
| `lgb1_probe` | 1 | 1（historical best individual lgb1） | 5 | 5 | なし |
| `full_family` | 1 | 3（lgb0/lgb1/lgb2） | 5 | 15 | なし |

2026-07-16 18:46 JSTにユーザーが `full_family` を明示承認した。実行scopeは
`full_family_1_variant_3_configs_5_folds_15_boosters_no_control_retraining`。
単一のcanonical train notebook 1ファイル内でlgb0/lgb1/lgb2を順に実行し、親/controlは再学習しない。

## コマンドログ

- 2026-07-16: `make new-steering EXP=exp261_lightgbm_extra_trees_ablation_on_exp218`。
- 2026-07-16: `make new-exp EXP=exp261_lightgbm_extra_trees_ablation_on_exp218`。
- 2026-07-16: 親exp218 config / metrics / notebook、`docs/06_reproducibility.md`、最近のcompact notebookを確認。
- 2026-07-16: Jupytext percent形式train/inference source、config、実験記録を実装。Kaggle trainは未push。
- 2026-07-16: train/inferenceの`jupytext --to ipynb --test`、`py_compile`、`ruff --select F821`がPASS。
- 2026-07-16: `make validate-exp EXP=exp261_lightgbm_extra_trees_ablation_on_exp218` strictと`make validate-template`がPASS。
- 2026-07-16: canonical `kentookumura/exp261-lightgbm-extra-trees-ablation-on-exp218-train` packageを`run_on_push=false`でprepare。private GPU、internet off、kernel sources 3件、bootstrap 14 filesを確認した。pushはしていない。
- 2026-07-16 18:46 JST: ユーザーが `full_family`（active variant 1、LightGBM configs 3、folds 5、合計15 boosters、parent/control再学習なし）を明示承認。単一train notebookでの実行指定も確認した。
- 2026-07-16: strict validationとtemplate validationを再実行してPASS。canonical packageを`run_on_push=true`で再prepareした。code fileは単一の`exp261_lightgbm_extra_trees_ablation_on_exp218_train.ipynb`、private GPU、internet off、kernel sources 3件。notebook内bootstrap 14 filesを展開せずに監査し、埋め込みconfigが`full_family`、config indices `[0,1,2]`、5 folds、15 boosters、control再学習なし、承認scope一致であることを確認した。
- 2026-07-16: `make push-kaggle-train EXP=exp261_lightgbm_extra_trees_ablation_on_exp218`の初回pushは`SaveKernel`の詳細なし400で失敗した。別slugへ変更・再pushはせず、まずcanonical idの存在確認とmetadata/CLI制約の切り分けを行う。
- 2026-07-16: 52文字のinitial id `kentookumura/exp261-lightgbm-extra-trees-ablation-on-exp218-train`をpullしたが403で、kernel作成は確認できなかった。既存実験のlong-slug `SaveKernel 400`復旧例と一致するため、同じexp261・同じ単一notebook・同じ科学contractのまま、意味を残した短いcanonical id/title `kentookumura/exp261-lgb-extra-trees-exp218-train` / `exp261 lgb extra trees exp218 train`へ同時に揃える。initial slugへの再pushは行わない。
- 2026-07-16: 短縮canonical packageを再監査。slug/title 35文字で一致し、code fileは単一train notebook、private GPU、internet off、run-on-push true、kernel sources 3、bootstrap 14、承認scope・3 configs・5 folds・15 boosters・control再学習なしを再確認した。
- 2026-07-16: `make push-kaggle-train EXP=exp261_lightgbm_extra_trees_ablation_on_exp218`が成功し、`kentookumura/exp261-lgb-extra-trees-exp218-train` version 1を開始した。
- 2026-07-16: push後の`kaggle kernels pull -m`が成功。Kaggle kernel `id_no=127462163`、private、internet off、`machine_shape=Gpu`、kernel sources 3件を確認した。
- 2026-07-17: ユーザーから完了連絡を受領し、`kaggle kernels logs kentookumura/exp261-lgb-extra-trees-exp218-train`を取得。version 1は`train_completed_guard_failed`、elapsed `19931.364`秒で完走した。
- 2026-07-17: `kaggle kernels output ... --file-pattern '.*exp261_.*\\.(csv|json)$' --page-size 200`で評価用CSV/JSONだけを`kaggle/output/train_v1/`へ取得した。OOF gzip、15 model本体、plotはnegative結果の確認に不要なため取得していない。
- 2026-07-17: 取得したparameter audit、schema、metrics、fold/bucket/hidden/by-well/blend/importance、guard、manifest、summaryのSHAを再計算し、Kaggle summary記録と全件一致した。feature schemaはheader込み381行（380 features）、manifestは15 models、parameter差分は3 configすべて`extra_trees`だけだった。

## Train v1結果

| model | parent RMSE | `extra_trees=True` RMSE | delta |
| --- | ---: | ---: | ---: |
| lgb0 | 8.557165712 | 8.864913264 | +0.307747552 |
| lgb1 | 8.512227651 | 8.751623580 | +0.239395929 |
| lgb2 | 8.524447601 | 8.757903289 | +0.233455688 |
| 3-config mean | 8.475793752 | 8.755217124 | +0.279423372 |

- selected meanのfold deltaは`[-0.077098, +0.056801, +0.114268, +0.675549, +0.588158]`で、改善は1/5 folds。
- 1000+は`9.294447098 -> 9.610221816`（`+0.315774718`）。
- hidden-like spatialは`+0.243318548`、typewell-purgedは`+0.250390215`。
- worst well `389ae58f`は`21.213000 -> 32.537422`（`+11.324423`）。
- fixed blendは最小のextra weight `0.25`でもoverall `+0.031917897`悪化。near 0-100など一部bucketの小改善はあるが、1000+とhidden-likeは悪化した。
- 親3-config予測の再構築はmean/max absolute differenceとも0、RMSEも`8.475793751656624`で完全一致した。
- 全6 adoption guardがfalseで、`adoption_supported=false`。

## 実装内容

- exp072 base cache、known-prefix anchor、U-projection、exp145 learned-likelihood、GRWRをexp218と同じ順で再構築する。
- 保存済みexp218 OOF、feature schema、15-model manifestをSHAで固定する。
- 選択したconfigへ `extra_trees=True` を追加し、他parameter差分0をassertする。
- 保存済み対応config boosterを同一fold valid rowsへ推論し、historical pooled RMSEを再現してから新規variantを学習する。
- full family時は保存済み3-config平均とfrozen exp218 `lgb_mean` OOFのrow parityを確認する。
- OOF相関、fixed blend weight `0.25/0.50/0.75`、distance / 1000+ / exp115 hidden-like 2面 / fold / by-well / worst-wellを出力する。
- feature importance plot、model SHA付きmanifest、OOF decompressed SHA、feature content SHAを保存する。
- 親exp218にcompact self-contained train sourceはない。親の正規train sourceは183行/4章、exp261は1000行/8章で、親helperを重いfeature builderとして再利用しつつconfig、入力、feature surface、control推論、学習、stress、生成物の上位orchestrationをnotebookへ展開した。

## 再現性メモ

- seed policy: exp218 GroupKFold seed 42とconfig別seed `123/0/29`を固定。
- stochastic components: upstream exp072/145 cache、`extra_trees` random threshold、GPU LightGBM。
- CPU/GPU runtime: Kaggle GPU、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8。CPU controlは学習しない。
- parent OOF decompressed SHA: `5f3fc95182eea348f3545771e67778ce191e7ba468eee7b267f4993369422976`。
- parent feature schema SHA: `aaf5f13f1e7c5236cd332dcebfdbf98e9c08247465833232e79ce3ff56362b49`。
- parent model manifest SHA: `904570def0d6ad0140f3df95c8bb38f31823295fd191206290e3833b5b2cc237`。
- hidden-like assignment SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`。
- feature content SHA: `f6ff78f6a95e47b0ed8e76a22c31d3403d0a9e78471b7d64f37eef7a2a398e29`。
- new OOF decompressed SHA: `021fd47ac556b7ce98f7991b0c97aa6996b359f07c07a7e6a339c84d84100f00`。
- model manifest SHA: `d76ca02ad54184310eafb2758bd287a48cf20cbd95911db5eaef103c8e8de476`。15 model個別SHAはmanifestに記録。
- summary SHA: `34cb4c1d62076f0f62f402a52975b159ae1c966ddfff12fcbdf24384f6705ad4`。
- submission SHA / rerun check: 初回範囲外。

## 次のアクション

1. 回帰LightGBMの`extra_trees=True`は不採用とし、inference / submit / rescue gridへ進めない。
2. 一部near bucketの小さなblend改善を再訪する場合も、保存OOFだけの0-booster安定性readoutに限定する。
3. selector LightGBMのexp262は目的・学習行・損失が異なるため、この結果から成否を代用せず独立に判定する。
