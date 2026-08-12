# タスクリスト

## TODO

- exp232 Kaggle train を完了し、candidate / bucket / hidden-like / by-well /
  interval artifacts を exp233 train の input として固定する。
- Kaggle train package を生成し、bootstrap 内 config と kernel metadata を照合する。
- CPU-only Kaggle train を実行し、logs / notebook 出力から full-train metrics を記録する。
- RMSE、coverage、1000_plus、hidden-like、worst-well、temperature comparison guard を確認する。

## 進行中

- exp232 の target-free gate を保持した state-neutral Uniform mixture PF を実装中。
- mixture / gate diagnostic、exp232 artifact contract、train notebook を整備中。

## ブロック中

- exp232 artifacts が未生成のため、temperature comparison による exp233 の採用判定は不可。

## 完了

- mixture と temperature を別 experiment として扱い、同時に変更しない方針を固定した。
- User approval により Uniform support `[0,500]` と epsilon `0.02/0.05` を固定した。
- User approval により exp232 と exp233 の CPU Kaggle train を並行実行する。exp233 の
  initial output は comparison pending として扱い、温度比較前には採用しない。
- global mixture、target-derived gate、outlier Gaussian、control 再生成、inference、
  submissionを範囲外にした。
