# 要件

## 依頼

`exact_hmm_posterior_bimodality_audit` を `exp236` として実装する。exp221 の
固定 Gaussian-emission exact HMM posterior を well 単位で再生成し、二峰性、
mode persistence、および posterior mean が二峰間の谷に入る現象を train-side
OOF で監査する。

## 制約

- Route: `ensemble`。exp148 の保存済み LGB OOF を HMM emission center に使うが、
  LGB の再学習、prediction 変更、blend weight の探索はしない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 主対象は exp221 の `sigma=20.0` / `lambda=0.50` single variant とする。HMM
  grid、GR emission、transition、rate grid、initialization は exp221 と完全に固定する。
- 生成は CPU-only、LightGBM config / fold / booster は `0 / 0 / 0`。親 control は
  再学習しない。
- peak 検出、二峰判定、mode tracking は posterior と固定 config だけで行う。
  true TVT / error は評価・oracle readout・plot annotation にのみ使用する。
- full posterior tensor を artifact として保存しない。well 単位で要約を書き出して
  posterior を破棄し、代表的な well plot だけを保存する。
- inference、submission、mixture emission、mode-state追加、post-hoc correction は
  実装範囲外とする。

## 受け入れ基準

- exp221 OOF center が全 train pseudo-tail ID を一意に被覆し、各 well の HMM
  入力と row order が一致する。
- `posterior_mean`、`marginal_map`、`dominant_mode_conditional_mean` を同一posterior
  から生成し、overall / distance bucket / hidden-like / worst-well / step-delta で
  比較する。
- row / segment / well summary に peak count、top1/top2 mass・間隔、valley depth、
  entropy、bimodal rate、mode persistence、switch count を保存する。
- top2 oracle coverage は診断値としてのみ保存し、いかなる decoder / threshold の
  選択にも用いない。
- static validation、Jupytext round-trip test、構文チェック、F821、strict experiment
  validation が通る。
