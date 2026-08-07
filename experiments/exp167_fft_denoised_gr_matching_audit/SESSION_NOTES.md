# exp167_fft_denoised_gr_matching_audit セッションノート

## 目的

`fft_denoised_gr_matching_audit` バックログの実装。GR rotation FFT denoise 後に typewell GR matching / shift scan の localization quality が raw GR より改善するかを、PF/Beam 生成や ML feature 化の前に train-side diagnostic として確認する。

## 現在の状態

- Route: pf_beam
- 状態: completed
- CV: train-side diagnostic only
- LB: まだなし

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp167_fft_denoised_gr_matching_audit
uv run python scripts/new_experiment.py --name exp167_fft_denoised_gr_matching_audit --source templates/experiment
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_inference.py
.venv/bin/python -m py_compile experiments/exp167_fft_denoised_gr_matching_audit/fft_denoised_gr_matching_audit.py experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_train.py experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_inference.py experiments/exp167_fft_denoised_gr_matching_audit/settings.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_inference.py
.venv/bin/ruff check experiments/exp167_fft_denoised_gr_matching_audit/fft_denoised_gr_matching_audit.py experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_train.py experiments/exp167_fft_denoised_gr_matching_audit/exp167_fft_denoised_gr_matching_audit_inference.py experiments/exp167_fft_denoised_gr_matching_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp167_fft_denoised_gr_matching_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp167_fft_denoised_gr_matching_audit --notebook train --kernel-id kentookumura/exp167-fft-denoised-gr-matching-audit-train --title 'exp167 fft denoised gr matching audit train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp167_fft_denoised_gr_matching_audit --notebook inference --kernel-id kentookumura/exp167-fft-denoised-gr-matching-audit-inference --title 'exp167 fft denoised gr matching audit inference' --strict
.venv/bin/python -m py_compile experiments/exp167_fft_denoised_gr_matching_audit/kaggle/train/fft_denoised_gr_matching_audit.py experiments/exp167_fft_denoised_gr_matching_audit/kaggle/train/settings.py experiments/exp167_fft_denoised_gr_matching_audit/kaggle/inference/settings.py
uv run python scripts/record_experiment.py --experiment exp167_fft_denoised_gr_matching_audit --status scaffold_completed --metric diagnostic --key-idea 'FFT denoised horizontal GR shift-scan audit for typewell matching; no ML/PF generation/submission yet.' --notes 'Implemented train-side raw/rolling/savgol-fallback/fft-notch GR matching audit; Kaggle train execution pending.'
kaggle kernels push -p experiments/exp167_fft_denoised_gr_matching_audit/kaggle/train
kaggle kernels logs kentookumura/exp167-fft-denoised-gr-matching-audit-train
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp167_fft_denoised_gr_matching_audit --notebook train --kernel-id kentookumura/exp167-fft-denoised-gr-matching-audit-train --title 'exp167 fft denoised gr matching audit train' --run-on-push --strict
kaggle kernels push -p experiments/exp167_fft_denoised_gr_matching_audit/kaggle/train
kaggle kernels logs kentookumura/exp167-fft-denoised-gr-matching-audit-train
kaggle kernels output kentookumura/exp167-fft-denoised-gr-matching-audit-train -p experiments/exp167_fft_denoised_gr_matching_audit/kaggle/output/train_v2
```

## Kaggle 実行結果

- v1: push 成功後、Kaggle 実行は `ValueError: No kernel name found in notebook and no override provided.` で失敗。原因は Jupytext 変換 notebook の kernelspec metadata 欠落。
- 修正: train / inference notebook に `kernelspec` metadata を入れ、train package を再生成。
- v2: `kentookumura/exp167-fft-denoised-gr-matching-audit-train` version 2 完了。
- URL: https://www.kaggle.com/code/kentookumura/exp167-fft-denoised-gr-matching-audit-train
- Kaggle kernel id_no: `125653908`
- runtime: CPU、internet disabled、GPU disabled。
- logs: 773 train wells を処理し、242 sec 前後で metrics / 生成物保存まで完了。
- output: `experiments/exp167_fft_denoised_gr_matching_audit/kaggle/output/train_v2`
- output 取得: `row_context.csv.gz` が大きく、`kaggle kernels output` は途中で `IncompleteRead` になった。部分取得された row_context は削除し、取得済みの集計 CSV だけを根拠にする。

取得済み生成物 SHA:

- `filter_metrics.csv`: `1062b2ea50743e895b409426d8f51cdce920261f073b48aede520d34c8c8bc48`
- `filter_gain_vs_raw.csv`: `782c2463d8fea29a344d33b585904f141703acdd3e40d1beca8dbdbd2d00b125`
- `bucket_metrics.csv`: `6c645c89a15be49c6872744bb9ebf26239312759b36e8da1e3c29dfaef7fcec2`

主要 metrics:

| filter | region | RMSE | MAE | within10 | gap | entropy | decoy gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | all | 108.659395 | 69.872190 | 0.163022 | 1.166088 | 0.712493 | 5.846472 |
| fft_notch_top2 | all | 108.581079 | 69.807558 | 0.163499 | 1.165975 | 0.711578 | 5.848499 |
| rolling_median_11 | all | 108.532613 | 69.544342 | 0.165760 | 1.298408 | 0.697719 | 6.352187 |
| savgol_31_p2 | all | 108.522038 | 69.569115 | 0.165121 | 1.343931 | 0.693299 | 6.520488 |
| raw | hidden_tail | 125.711348 | 76.615849 | 0.151576 | 1.156021 | 0.711906 | 5.815072 |
| fft_notch_top2 | hidden_tail | 125.580817 | 76.453655 | 0.151722 | 1.152656 | 0.711042 | 5.814574 |
| rolling_median_11 | hidden_tail | 125.690496 | 76.406712 | 0.153122 | 1.289864 | 0.696782 | 6.321843 |
| savgol_31_p2 | prefix_backtest | 87.711709 | 62.465740 | 0.177393 | 1.352752 | 0.694301 | 6.552275 |

解釈:

- FFT notch は all / hidden_tail RMSE と MAE で raw をわずかに上回ったが、hidden_tail の gap と decoy gap は raw と同等または微悪化で、surface 改善の根拠は弱い。
- prefix_backtest では FFT notch の RMSE は raw よりわずかに良いが、MAE / within2 / within5 は悪化した。
- rolling median と Savitzky-Golay fallback は gap、entropy、decoy gap の改善が明確で、単純 smoothing のほうが matching surface の安定化には有望。
- `denoised_gr_pfbeam_generation_audit` へ FFT notch をそのまま進める根拠はない。進める場合は FFT ではなく rolling/savgol smoothing を別仮説として小さく切る。

## 変更点

- `config.yaml` を `pf_beam` route の matching audit 用に更新。
- raw train horizontal/typewell GR と known `TVT_input` prefix だけで、target-free linear TVT prior 周辺の shift scan を行う設計にした。
- `fft_denoised_gr_matching_audit.py` に raw / rolling median / Savitzky-Golay fallback / FFT notch の filter 比較と、row context / filter metrics / bucket metrics / well metrics / raw-vs-denoised gain / input summary / summary JSON の保存を実装。
- Jupytext percent 形式の train / inference `.py` を作り、正規 `.ipynb` に変換。
- Kaggle package:
  - train kernel id: `kentookumura/exp167-fft-denoised-gr-matching-audit-train`
  - train title: `exp167 fft denoised gr matching audit train`
  - inference kernel id: `kentookumura/exp167-fft-denoised-gr-matching-audit-inference`
  - inference title: `exp167 fft denoised gr matching audit inference`
- active variant 数: 0。
- LightGBM config 数: 0。
- fold 数: 0。
- 合計 booster 数: 0。
- 親実験 control 再学習: なし。

## 再現性メモ

- seed policy: no RNG。eval row sampling は deterministic `np.linspace`。
- stochastic components: なし。
- CPU/GPU runtime: CPU のみ。GPU 学習なし。
- PF/Beam / likelihood-PF: 生成なし。
- Kaggle kernel id / version: `kentookumura/exp167-fft-denoised-gr-matching-audit-train` v2。
- input / feature schema SHA: 取得済み集計 CSV の SHA を上記に記録。full row context は未取得。
- feature content SHA: 集計 CSV は raw file SHA を記録。row_context gzip は incomplete output のためローカル保存しない。
- model manifest / model SHA: 対象外。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun check: 未実行。

## 次のアクション

1. FFT notch を PF/Beam generation に直接進めない。
2. GR smoothing を続けるなら `rolling_median_11` / `savgol_31_p2` を対象に、heel calibration と組み合わせた shift-scan audit として別実験化する。
