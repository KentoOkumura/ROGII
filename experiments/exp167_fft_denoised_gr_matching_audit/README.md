# exp167_fft_denoised_gr_matching_audit

## 状態

- ルート: pf_beam
- 状態: completed
- CV: train-side diagnostic
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-02
- 親実験: `fft_denoised_gr_matching_audit` backlog

## 仮説

GR sensor rotation 由来の周期ノイズを target-free な FFT notch で落とすと、typewell GR matching の cost surface が raw GR より鋭くなり、top1 localization、top1-top2 gap、entropy、decoy gap が改善する。

## 変更点

- raw train の horizontal / typewell GR のみを入力にした train-side shift-scan audit を追加。
- raw、rolling median、Savitzky-Golay fallback、FFT notch を同じ prior / shift grid / eval rows で比較する。
- PF/Beam 候補生成、ML feature 化、prediction replacement、submission は行わない。

## 検証方針

- Fold: なし。train-side diagnostic。
- Group: well。
- Stratification: hidden tail sampled rows、prefix backtest、distance bucket、worst-well。
- Leakage Check: eval true TVT は metric 計算にだけ使い、denoise 周波数選択、normalization、shift center、threshold には使わない。

## 実行入口

- 学習 notebook: `exp167_fft_denoised_gr_matching_audit_train.ipynb`
- 推論 notebook: `exp167_fft_denoised_gr_matching_audit_inference.ipynb`
- Kaggle 準備: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp167_fft_denoised_gr_matching_audit --notebook train --strict`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- Kaggle train v2 で 773 wells の raw / rolling / savgol / FFT notch shift-scan audit が完了。
- rolling median / Savitzky-Golay fallback は raw より gap、entropy、decoy gap を改善した。

### 悪かった点

- FFT notch は raw に対する改善が小さく、hidden_tail の gap / decoy gap はほぼ改善しなかった。
- direct top1 RMSE は 100ft 級で、候補 path として直接使う水準ではない。

### リスク / 注意

- FFT notch は `denoised_gr_pfbeam_generation_audit` へ直接進めない。続けるなら rolling/savgol smoothing と heel calibration の組み合わせを別実験で見る。

## 次

- `heel_calibrated_shift_scan_pfbeam_audit` を優先する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
