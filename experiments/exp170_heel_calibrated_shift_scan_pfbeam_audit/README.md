# exp170_heel_calibrated_shift_scan_pfbeam_audit

## 状態

- ルート: `pf_beam`
- 状態: completed_train_side_rejected_no_submit
- CV: diagnostic_only
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-02
- 親/参照: `KAGGLE_DIRECTION.md` backlog、`exp167_fft_denoised_gr_matching_audit`、`exp072_exp063_full_replay_feature_cache`

## 仮説

known prefix の `TVT_input` を使って typewell GR の gain / offset を well-local に calibration すると、flat prior や raw typewell GR よりも GR shift-scan の localization surface が鋭くなり、固定 PF/Beam 候補の observation likelihood readout も改善する可能性がある。

これは `heel_calibrated_shift_scan_confidence_on_exp148` のような ML 特徴量化や、PF/Beam likelihood 本体の変更へ進む前の train-side audit である。

## 変更点

- exp167 の typewell GR shift-scan diagnostic を拡張した。
- raw / `rolling_median_11` / `savgol_31_p2` の horizontal GR filter を比較する。
- calibration mode は `raw`、`flat_calibrated`、`heel_calibrated`。
- `heel_calibrated` は known prefix の `TVT_input` で typewell GR を sample し、horizontal GR へ robust affine fit する。
- `flat_calibrated` は同じ known prefix で MD-linear TVT prior を使って affine fit する。
- exp072 の固定 PF/Beam/likelihood-PF candidate cache を読み、candidate RMSE と observation cost / rank / gap を出す。
- PF/Beam 再生成、ML 学習、candidate replacement、inference port、提出は行わない。

## 検証方針

- 検証面: train well hidden-tail と prefix-backtest の sampled rows
- 主指標: shift-scan top1 RMSE / MAE / within2 / within5 / within10
- 補助指標: top1-top2 gap、entropy、decoy gap、best cost、raw 比 gain
- PF/Beam readout: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`、`pf_z` の candidate RMSE と observation cost
- guard: near-row、`1000_plus`、well別 worst regression、common worst wells

## 実行入口

- 学習 notebook: `exp170_heel_calibrated_shift_scan_pfbeam_audit_train.ipynb`
- 推論 notebook: `exp170_heel_calibrated_shift_scan_pfbeam_audit_inference.ipynb`
- Kaggle 準備例:

```bash
make prepare-kaggle-notebooks EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp170-heel-calibrated-shift-scan-pfbeam-audit-train --title 'exp170 heel calibrated shift scan pfbeam audit train' --run-on-push --strict"
```

## 所見

Kaggle train v1 は完了した。heel calibration は hidden_tail / prefix_backtest の両方で raw より top1 error を悪化させ、PF/Beam observation rank も改善しなかった。PF/Beam likelihood 変更、exp148 add-only feature 化、inference port、submit は行わない。
