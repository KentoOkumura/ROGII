# exp185_last50_first_prefix_feature_rebuild_on_exp148 結果

完了。不採用。inference port / submit はしない。

## 結果

- `lgb0`: CV 8.636150399
- `lgb1`: CV 8.583238238
- `lgb2`: CV 8.583791509
- 3-config `lgb_mean_split3`: CV 8.544817143
- 主 baseline exp148 `lgb_mean`: CV 8.501281182 / Public LB 7.960
- exp148 との差分: +0.043535961 悪化

exp172 last50 multiobs replacement-only best single 8.575126850 は上回ったが、親 exp148 anchor には届かない。last50-first prefix feature rebuild 仮説は global OOF で negative と判断する。

## 採否

- 採用: なし
- inference port: なし
- submit: なし
- backlog: `last50_first_prefix_feature_rebuild_on_exp148` は実装済み/不採用として閉じる

GPU split 実行は失敗。feature cache は Kaggle GPU notebook `kentookumura/exp185-l50rebuild-features` で完了確認済み。

2026-07-04 時点で split train は以下の状態。

- `train_lgb0`: `kentookumura/exp185-l50rebuild-lgb0` v1 は GPU 実行失敗。CPU 版へ切り替え。
- `train_lgb1`: `kentookumura/exp185-l50rebuild-lgb1` v1 は GPU 実行失敗。CPU 版へ切り替え。
- `train_lgb2`: GPU 版は Kaggle GPU 同時 session 上限 `Maximum batch GPU session count of 2 reached.` により push 実行未開始。CPU 版へ切り替え。

CPU 版 split train は push 済み。

- `train_lgb0`: `kentookumura/exp185-l50rebuild-lgb0` v2 は `prefix_crop_variant_join_start` 直後に `DeadKernelError`。メモリ対策後、v3 / CPU 実行開始。
- `train_lgb1`: `kentookumura/exp185-l50rebuild-lgb1` v2 は `prefix_crop_variant_join_start` 直後に `DeadKernelError`。メモリ対策後、v3 / CPU 実行開始。
- `train_lgb2`: original slug `kentookumura/exp185-l50rebuild-lgb2` は `Notebook not found` で作成できなかったため `kentookumura/exp185-l50rebuild-cpu-lgb2` に変更。v1 は `prefix_crop_variant_join_start` 直後に `DeadKernelError`。メモリ対策後、v2 / CPU 実行開始。

split train を使う場合は Kaggle metadata と LightGBM mode の両方を CPU にそろえる。

今回の失敗は3本とも同じで、prefix crop cache の 76列を 3,783,989 行の full frame に横結合する段階で kernel が死んでいる。対策として、prefix crop cache 読み込み時に numeric columns を `float32` 指定し、finite check を列単位に変更し、不要になった大きい中間 frame を prefix join 前に解放し、学習用 `variant_frame` は runtime columns と選択 feature だけに slim 化した。

## 比較基準

- 主 baseline: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- control 再学習: なし
- 参考: exp161 last50 add-only best single CV 8.56472499591314
- 参考: exp166 tail500 replacement-only best single CV 8.566426970340796
- 参考: exp172 last50 multiobs replacement-only best single CV 8.57512684958155

## 実行構成

- active variant: `last50_first_prefix_rebuild`
- active mode: `cpu_deterministic_threads8`
- LightGBM configs: `lgb0`, `lgb1`, `lgb2`
- folds: 5
- total boosters: 15
