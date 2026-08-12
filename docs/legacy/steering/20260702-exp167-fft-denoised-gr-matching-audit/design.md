# 設計

## アプローチ

raw train の各 well について、known `TVT_input` prefix から MD to TVT の線形 continuation prior を作る。
hidden tail と prefix backtest rows を deterministic に subsample し、各 row の prior TVT 周辺で
typewell GR への shift scan を行う。

比較する horizontal GR filter:

- `raw`: interpolation のみ。
- `rolling_median_11`: 短周期 spike の robust smoothing。
- `savgol_31_p2`: SciPy があれば Savitzky-Golay、なければ rolling mean fallback。
- `fft_notch_top2`: full horizontal GR から target-free に dominant periodic peak を検出し、その周波数帯を notch する。

各 filter で同じ prior、同じ shift grid、同じ typewell GR、同じ eval rows を使い、raw より
localization error、gap、entropy、decoy gap が改善するかだけを見る。

## 実験範囲

- 対象実験: `exp167_fft_denoised_gr_matching_audit`
- Route: `pf_beam`
- 親実験: backlog `fft_denoised_gr_matching_audit`
- 変更する変数: horizontal GR denoise filter。
- 固定する変数: raw train input、known-prefix linear TVT prior、shift grid、typewell GR interpolation、eval row sampling、metric definitions。

## 再現性設計

- seed policy: no RNG。eval row sampling は `np.linspace` による deterministic subsample。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。PF/Beam route の前段 diagnostic だが候補生成は変更しない。
- 並列処理と乱数の関係: 並列処理なし、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: CPU のみ、GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: raw train files と生成 CSV/JSON の SHA を summary に記録する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は作らないため対象外。
- Kaggle package bootstrap 確認方針: 正の編集対象を更新後、`prepare-kaggle-notebooks --notebook train --strict` で package 化し、必要に応じて `kaggle/train/config.yaml` と helper を確認する。

## リスク

- リークリスク: eval true TVT を filter selection / shift center / normalization に使うと漏れる。実装では評価 metric 以外から排除する。
- CV/LB 不一致リスク: train-side raw matching audit であり hidden test submit には直結しない。positive でも PF/Beam generation か exp148 confidence feature へ別実験で進める。
- ランタイム/メモリリスク: 全 row x dense shift grid は重い。per-well deterministic subsample と vectorized chunk scan に限定する。
- 再現性リスク: FFT peak tie や SciPy availability で filter が変わり得る。filter metadata と fallback flag を well summary に保存する。
