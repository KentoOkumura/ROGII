# exp327_time_varying_position_sigma_floor_audit

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp323_time_varying_exp226_dip_rate_prior`

親exp323がterminal closeしたため、2026-07-22に本実験も閉鎖した。新exp327相当は、exp338 PASS後に作る新exp323相当がさらにPASSした場合だけ、新番号で設計する。

## 仮説

現行`sig_p=0.02`は`0.35 × step = 0.1225 ft`のfloorにより不活性である。各行のposition transition meanが0.35 ft gridの中間へ来るほどposition kernelを少し広げれば、丸め位相による過信を減らせる。

```text
q_t = median_rate_states |mu_t - 0.35 round(mu_t/0.35)|
sig_p,t = clip(sqrt(0.1225² + q_t²), 0.1225, 0.245)
```

posteriorや誤差は式に入れない。

## 段階

- Stage 0: known-prefix one-step transition NLLを固定floorと比較する。HMM 0。
- Stage 1: Stage 0全PASSと別承認後だけ1 variant / 773 HMM runs。

## 検証方針

- Stage 0: 固定floor比でposition transition NLLを1%以上改善し、4/5 folds、平均sigma inflation 0.01 ft以上、upper clip率25%以下、tail非悪化を要求する。
- Stage 1: 保存済みexp323親HMM比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst非悪化を要求する。
- 量子化残差とsigma scheduleをprefix情報だけで生成し、`0.1225 ft`未満の値がないことをhard assertする。

## 所見

現行`sig_p=0.02`が実効floorで不活性なため、これは物理ノイズ推定ではなくgrid量子化の数値監査である。上位のmean・rate sigma・momentum案より優先度を下げ、現在は結果なしとする。

位置sigmaは物理モデルより数値離散化の補正に近い。本実験の実装・実行は今後行わない。
