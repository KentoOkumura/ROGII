# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。後継資格はexp338の`successor_policy`を正とする。

## アプローチ

全41 rate statesのposition transition meanについてgrid丸め残差を計算し、row medianをbase floorへ二乗加算する。known-prefix one-step transition NLLで、固定floorより離散化calibrationが良いか先に監査する。

## 実験範囲

- 対象: `exp327_time_varying_position_sigma_floor_audit`
- Route: `pf_beam`
- 親: `exp323_time_varying_exp226_dip_rate_prior`
- 変更: effective position sigmaだけ。
- 固定: grid step、rate mean/variance、GR、momentum、decoder。

## 再現性設計

RNGなし。transition mean、quantization residual、sigma schedule、prefix NLL、prediction content SHAを保存する。CPU/internet off、0 booster。

## リスク

- 既存kernelが既にquantization offsetを評価しており冗長な可能性が高い。
- sigma拡大でbranch jumpが増えるため、上限0.245とtail/worst gateを固定する。
- Late phaseでは上位案を優先し、本案を自動実装しない。
