# exp236_exact_hmm_posterior_bimodality_audit 結果

## 結論

固定 exp221 exact HMM の posterior は一部で二峰化するが、採用中の posterior mean を
MAP または dominant-mode conditional mean に直接置換する根拠は得られなかった。Kaggle
CPU train-side audit v1 は完了し、推論・提出は行わない。

## 実行

- Kernel: `kentookumura/exp236-exact-hmm-posterior-bimodality-audit-train` v1（COMPLETE）
- CPU-only、internet disabled、HMM variant 1、LightGBM config / fold / booster = `0 / 0 / 0`
- 3,783,989 rows / 773 wells、elapsed 27,168.333 sec（約7時間33分）
- 親 exp221 の `exp148 lgb_mean` center、sigma 20.0、lambda 0.50、HMM grid / transition /
  emission を固定。親/controlの再学習はない。

## decoder readout

| Decoder | RMSE | MAE | RMSE差（posterior mean比） |
| --- | ---: | ---: | ---: |
| posterior mean | 8.327728486 | 4.811963870 | 0 |
| marginal MAP | 8.365160435 | 4.843304464 | +0.037431949 |
| dominant-mode conditional mean | 8.331754352 | 4.812731548 | +0.004025866 |

posterior mean は親 exp221 の記録値 8.327736951 と 0.000008465 差で再現できた。distance
bucket は全域で posterior mean が最良だった。hidden-like では dominant mode に約 -0.0024
の局所差があったが、全体悪化を覆さない。

## posterior 形状

- 二峰 row: 35,399 / 3,783,989（0.9355%）、138 wells、317 segments
- mean-in-valley row: 6,781（0.1792%）
- mode mass switch / track break: 17 / 17。頻繁な mode slip は確認されなかった。
- 二峰 subset の RMSE は posterior mean 11.053503351、dominant mode 11.373182708、MAP
  11.438155606。mean-in-valley subsetも posterior mean 9.259919533 が最良。
- oracle top2（診断のみ）は二峰 row で MAE 4.400838542、within10 0.878329896。これは
  target を使わない選択規則が未確立のまま利用できる改善ではない。

MAP は `|step delta| > 0.2` が 3.2659% となり、posterior mean の 0.0219% より大きく
spike-prone だった。dominant mode も 0.0394% で posterior mean を上回った。

## 判断

二峰性は高誤差域と重なる診断信号ではあるが、稀で mode switch も少ない。したがって
midpoint 補正、mode state、mixture emission、直接 decoder 置換、raw-test HMM 再生成、
inference、submission はすべて不採用とする。将来扱う場合も、raw-test parity と fold-safe
な生成を先に満たした add-only confidence 特徴量に限定する。
