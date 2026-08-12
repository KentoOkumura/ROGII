# 設計

## アプローチ

`exp092` の U-projection correction / disagreement surface を再構成し、`exp114` の fold-safe spatial neighbor prior OOF を row id / well で join する。追加特徴は prior の値そのものだけでなく、neighbor quality、prior uncertainty、PF / Beam / likelihood-PF との disagreement、exp118 best-gate 相当の target-free proxy、near / longtail interaction に限定する。

学習は保存済み exp092 baseline との比較で見る。control 再学習は GPU コストを増やすため disabled のままとし、Kaggle train では add-only variant だけを 3 LightGBM configs x 5 folds で学習する。

## 実験範囲

- 対象実験: `exp164_spatial_prior_confidence_features_on_exp092_kaggle`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: exp092 surface に追加する spatial prior confidence feature group
- 固定する変数: exp072 feature cache、exp092 U-projection settings、GroupKFold by well、LightGBM config family、seed 42、train target、baseline metrics
- 実行環境: Kaggle Notebook GPU、internet disabled、kernel sources は exp072 train cache と exp114 spatial prior audit

## 再現性設計

- seed policy: `seed=42`。新規 PF / Beam / likelihood-PF sampling は行わず、保存済み cache を読み込む。
- stochastic 処理の有無: 新規 feature merge は deterministic。LightGBM GPU 学習のみ stochastic risk が残る。
- PF/Beam / likelihood-PF / seed bagging の有無: 上流 exp072 cache を参照するだけで、この実験内では再生成しない。
- 並列処理と乱数の関係: LightGBM は `deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`、`gpu_use_dp=true`。
- CPU/GPU runtime と deterministic flags: primary mode は `gpu_repro_guard_dp_threads8`。CPU mode は config に残すが active にはしない。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を記録する。train summary に input / feature schema / feature content SHA を保存する。
- model manifest / prediction / submission SHA 記録方針: train では model manifest と OOF prediction SHA を記録する。submission SHA は inference disabled のため記録しない。
- Kaggle package bootstrap 確認方針: 正の編集対象を更新後、`prepare-kaggle-notebooks` を再実行し、generated `kernel-metadata.json` と bootstrap support files を確認する。

## リスク

- リークリスク: spatial prior は exp114 OOF diagnostics を使うため、fold-safe であることが前提。validation wells が同一 fold の train source に混ざらないこと、true validation TVT / absolute error / oracle choice / fold label を feature に使わないことを notebook と code で確認する。
- CV/LB 不一致リスク: train-side OOF で spatial neighbor prior が効いても hidden wells では崩れる可能性がある。inference は OOF、worst-well、bucket、raw-test parity を見てから判断する。
- ランタイム/メモリリスク: exp072 full replay rows に spatial feature を追加し、3 configs x 5 folds を GPU LightGBM で学習する。想定 booster は 15。control を含めないことでコストを抑える。
- 再現性リスク: 上流 cache は deterministic anchor として参照するが、この実験自体は train-side add-only audit であり、submission anchor ではない。Kaggle kernel version と SHA を実行後に記録する。
