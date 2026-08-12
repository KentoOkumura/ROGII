# 設計

## アプローチ

`exp167_fft_denoised_gr_matching_audit` の known-prefix linear TVT prior と typewell GR shift-scan を
軽量な可視化用途に再構成する。各 well で hidden tail と prefix backtest rows を deterministic に subsample し、
raw / smoothing / denoise filter ごとに同じ shift grid を評価する。

各評価 row では、水平井 GR の local row window と、best shift 後に typewell TVT 軸から補間した GR window を
「match pair」として扱う。選定した pair について以下を 1 枚の PNG にまとめる。

- 水平井全体の GR context と known-prefix 境界 / 評価 row。
- shift ごとの matching cost curve と best / true shift。
- local window の normalized GR waveform overlay。
- typewell GR context と prior / matched / true TVT marker。

## 実験範囲

- 対象実験: `exp168_gr_matching_pair_visualization`
- Route: `pf_beam`
- 親実験: `exp167_fft_denoised_gr_matching_audit`
- 変更する変数: 可視化対象 row / filter / region の選定と描画。
- 固定する変数: raw train input、known-prefix linear TVT prior、shift grid、local offsets、typewell GR interpolation、metric definitions。

## 再現性設計

- seed policy: no RNG。row sampling と example selection は `np.linspace` と stable sort のみ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。PF/Beam route の前段診断だが候補生成はしない。
- 並列処理と乱数の関係: 並列処理なし、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: CPU のみ、GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: raw train file SHA と、scored pair gzip の decompressed content SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は作らないため対象外。
- Kaggle package bootstrap 確認方針: Jupytext 変換後、`prepare-kaggle-notebooks --notebook train --strict` で package 化し、metadata と bootstrap 内 config の整合を確認する。

## リスク

- リークリスク: true TVT を match selection に使うと漏れる。実装では true TVT は評価・図示 marker のみに使う。
- CV/LB 不一致リスク: train-side 可視化であり submit には直結しない。positive / negative の解釈は follow-up 実験で行う。
- ランタイム/メモリリスク: 全 row 描画は重い。default は well 数、eval row 数、PNG 数を制限する。
- 再現性リスク: SciPy availability で Savitzky-Golay fallback が変わり得る。filter metadata と summary に記録する。
