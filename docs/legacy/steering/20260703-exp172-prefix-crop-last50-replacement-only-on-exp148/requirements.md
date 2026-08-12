# 要件

## 依頼

`prefix_crop_last50_replacement_only_on_exp148` backlog を実装する。CPU 実行とし、タイムアウト対策のため学習コードは `lgb0` / `lgb1` / `lgb2` に分割する。

## 制約

- Route: `ml_model`
- Runtime: CPU。`runtime.kaggle.enable_gpu=false`、LightGBM は `cpu_deterministic_threads8`。
- 2段階構成にする。prefix crop feature cache を別 notebook で作成し、学習 notebook はその cache を読み込む。
- 学習 notebook は `lgb0` / `lgb1` / `lgb2` の3本に分割する。
- control は再学習しない。exp148 の保存済み CV / Public LB を historical baseline とする。
- 学習・評価行は crop しない。crop は known prefix feature の集計窓だけに限定する。
- PF/Beam、U-projection、learned probability/error model は last50 版に再生成しない。
- `docs/06_reproducibility.md` に従い、Kaggle kernel version、feature content SHA、model manifest SHA、prediction SHA を記録できる構成にする。

## 受け入れ基準

- `last50` の prefix crop cache を Kaggle CPU notebook で生成できる。
- `lgb0` / `lgb1` / `lgb2` split train notebook が cache を必須入力として読み、LightGBM 学習中に crop feature を再生成しない。
- active variant は `prefix_crop_last50_multiobs_replacement` のみ。
- replacement 対象 learned multiobs 系列が model feature list から除外され、対応する `prefix_crop_last50_multiobs` group が追加される。exp072/exp092 full-prefix base 列は維持する。
- 実行前に variant 数、mode 数、fold 数、LightGBM config 数、合計 booster 数を `SESSION_NOTES.md` に記録している。
