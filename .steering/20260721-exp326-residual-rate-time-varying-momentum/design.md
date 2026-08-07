# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。後継資格はexp338の`successor_policy`を正とする。

## アプローチ

exp323の`mu_r,t`変化を親`sig_r`単位へ標準化し、固定0.998から得るcorrelation lengthを`1+s_t`で短縮する。これをresidual-rate AR(1)の`m_t`にだけ使う。

## 実験範囲

- 対象: `exp326_residual_rate_time_varying_momentum`
- Route: `pf_beam`
- 親: `exp323_time_varying_exp226_dip_rate_prior`
- 変更: residual-rate momentumだけ。
- 固定: prior mean、process/observation sigma、position、grid、posterior mean。

## 再現性設計

RNGなし。parent/momentum schedule、activation、prefix backtest、predictionのcontent SHAを保存する。CPU/internet off、0 booster、親control再実行0。

## リスク

- `sig_r`との非識別性: `sig_r`を親固定し同時変更しない。
- absolute rateを0へ引く危険: exp323 residual座標以外では実行禁止。
- rare tail: Stage 1 hard guardsを必須にする。
