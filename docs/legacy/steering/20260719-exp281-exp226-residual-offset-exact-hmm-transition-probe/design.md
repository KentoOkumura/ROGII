# 設計

## アプローチ

exp226 geometry-only OOFをmoving coordinate center `g_t`とし、joint stateを
`(delta_t, offset_rate_t)`へ置き換える。

```text
TVT_t = g_t + delta_t
delta_t = delta_(t-1) + offset_rate_t * dMD_t + epsilon_position
```

absolute TVTのtransition centerは`g_t - g_(t-1)`となる。`offset_rate=0`ならexp226の
局所形状を一定の縦shiftだけ加えて追う。emissionは各row / offset stateについて
`typewell_GR(g_t + delta)`を評価し、exp209と同じknown-prefix sigmaとGaussian log-likelihoodを使う。
forward-backward後のposterior mean `E[delta_t]`を`g_t`へ加えて候補を作る。

## 実験範囲

- 対象実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- Route: `pf_beam`
- 親実験: shape parent `exp226`、decoder parent `exp209`、先行条件 `exp280`、失敗参照 `exp279`
- 変更する変数: absolute TVT position stateをexp226中心のoffset stateへ置き換える。
- 固定する変数: exp209 Gaussian emission / prefix calibration / missing-GR、step 0.35、41 rates、rate span 0.10、sig_r 0.002、sig_p 0.02、start sigma 0.75、r0 sigma 0.01、momentum 0.998、likelihood weight 1.0。
- 固定offset grid: `[-80, 80] ft`。exp280の13 shift bankの最大範囲をcontinuous exact gridへ写す。
- 初期状態: `delta=0`、offset-rate=0。救済gridや同一OOF tuningは行わない。
- 生成量: 1 fixed variant / 773 wells / LightGBM 0 / trained fold 0 / booster 0。

## 再現性設計

- seed policy: RNGなし。well文字列昇順と保存済みexp226 foldを固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072保存予測はexp263 control再構成だけに使う。
- 並列処理と乱数の関係: outer worker 1、Numba threads 4、RNGなし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/TPU/internet off。exp279実測から約5～6時間を想定する。
- train cache / test feature regeneration の SHA 記録方針: exp226 / exp209 / exp072のdecompressed SHAをhard guardし、OOF raw gzip / decompressed / logical prediction SHAを分ける。
- model manifest / prediction / submission SHA 記録方針: fitted modelなしのためdecoder scientific manifest SHAで代替し、OOF prediction SHAを保存する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: prepare後にloose / bootstrap configとtrain sourceのSHAを照合する。

## リスク

- リークリスク: unknown-suffix TVTやexp226 errorでgrid/state/pathを選ばない。truthは候補凍結後だけ読む。
- CV/LB 不一致リスク: exp226自体のCV/LB差とexp279の全fold回帰があるため、exp263 fixedとhidden-like/worst-wellを厳格guardする。
- ランタイム/メモリリスク: `T x delta-grid x 41 rates`のforward-backwardが重い。posteriorはwell単位で解放し、全posteriorを保存しない。
- 再現性リスク: Numba floating reductionの環境差があり得るため、初回成功だけではdeterministic anchorと呼ばない。
- 科学的リスク: exp280 top1は18.95%に留まる。hard shift correctionではなくslow grammarでのみ統合し、負結果後のparameter救済を禁止する。
