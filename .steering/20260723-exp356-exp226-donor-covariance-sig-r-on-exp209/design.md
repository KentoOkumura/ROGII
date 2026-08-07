# 設計

## アプローチ

exp209 exact-HMMのrate meanをconstantのまま固定し、exp226 K16 local-linear donorの
projected U-rate dispersionからsegment別`sig_r,t`だけを構成する。旧exp324と異なり
exp323 rate scheduleを必要としないため、transition meanとvarianceの同時変更を避ける。

Stage 0ではHMMを回さず、fold-safe donor ledgerから生成したscheduleをfreezeした後に、
constant`sig_r=0.002`とtransition NLL / coverageを比較する。Stage 0 PASSでも
Stage 1 exact-HMMは別承認とする。

## 実験範囲

- 対象実験: `exp356_exp226_donor_covariance_sig_r_on_exp209`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: rate-transition diffusion `sig_r,t`だけ
- 固定する変数: exp209 rate mean、observation、`sig_p=0.02`、grid、momentum、
  start/rate prior、posterior mean
- donor selection: exp226 local-linear K50 / bandwidth 500 / outer-fold-safe
- donor quantity: projected `Δ(TVT+Z)/ΔMD`
- robust scale: center周りの`1.4826 × weighted MAD`
- effective support: `(Σw)^2 / Σw^2`
- shrinkage: `alpha=n_eff/(n_eff+50)`のlog-space、parent scale 0.002
- clip/fallback: `[0.001,0.004]` / 0.002
- Stage 0: diagnostic 1 / 5 folds / HMM・model・booster 0
- Stage 1予約: 1 variant / 773 HMM runs / control再実行0

## 再現性設計

- seed policy: RNGなし、fold/well/segment/donor sort順を固定
- stochastic処理: なし
- PF/Beam/likelihood-PF: なし
- CPU/GPU: Kaggle CPU、internet/GPU/TPU off
- input SHA: exp226 OOF decompressed SHA、exp209 control SHA、raw well identity
- feature SHA: donor ledger、support、raw/shrunk scale、clip/fallback、row schedule
- model SHA: fitted modelなし。Stage 1時はdecoder contract SHAを保存
- prediction SHA: Stage 1時だけ保存
- submission SHA: 非該当
- bootstrap: package時にloose/package/bootstrap内configの一致を確認

## リスク

- リークリスク: validation wellと同fold validation wellsをdonorから完全除外する。
- 識別リスク: donor covarianceがactual rate uncertaintyを表さずclipへ崩壊し得る。
- 既知negative: exp338はwell-adaptive`sig_r`が全well upper clipとなった。
- CV/LBリスク: transition NLL改善がTVT RMSE改善へ移らない可能性がある。
- runtime: Stage 0は軽量、Stage 1は773 HMM runs。
