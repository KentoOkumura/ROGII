# exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm 結果

## 状態

- `stage_0_completed_guard_failed_closed`
- Kaggle private CPU Stage 0 version 1完了、固定7条件中4 PASS / 3 FAIL
- decision: `stage_0_failed_close_without_rescue`
- Stage 1、HMM、inference、submissionは未実行
- CV / LBなし

## 仮説

known prefixのhorizontal GRとtypewell GRのPearson一致度が低いwellだけ、exp209の
well-level `sigma_gr` を `1.3` 倍すると、良好wellを変えずにGaussian exact-HMMの誤mode固定を
弱められる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- primary selector: raw finite known-prefix Pearson `rho_gr`
- coefficient: `rho_gr >= 0.50 -> 1.0`、`rho_gr < 0.50 -> 1.3`
- support fallback: pair 64未満、低分散、nonfinite相関は `1.0`
- scale application: exp209 `[10,60]` clip後に1回だけ掛け、再clipなし
- Stage 0: truth-free agreement audit、HMM 0
- Stage 1: 全Stage 0 gate PASSと別承認時だけ1 variant / 5 folds / 最大773 HMM runs
- メトリック: Stage 0 identifiability / stability、Stage 1 unknown-suffix RMSE
- シード: RNGなし

## 結果

| メトリック | 値 |
| --- | --- |
| 専用test | 11 passed |
| Jupytext conversion test | PASS |
| ruff / py_compile | PASS |
| strict experiment validation | PASS |
| Kaggle kernel | version 1 / id_no `128540665` |
| Stage 0 runtime | `39.35975061899995 sec` |
| wells / full evaluable | `773 / 773` |
| fallback | `0 / 773 = 0.0` |
| poor multiplier | `8 / 773 = 0.01034928848641656`（FAIL、下限`0.10`） |
| tail evaluable | `773 / 773 = 1.0` |
| full/tail multiplier agreement | `0.666235446313066`（FAIL、下限`0.80`） |
| full/tail Spearman | `0.16746641700676126`（FAIL、下限`0.70`） |
| minimum fold primary coverage | `1.0` |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: no RNG、sorted well / raw row / fold order
- kernel:
  `kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train` version 1 /
  id_no `128540665`
- scientific contract SHA:
  `d2af5925416871f393aaffd3e638c0fb01777d7f74abd592479e7ea9890c3053`
- agreement logical SHA:
  `20a69c425a4f85b288b091f65b1bfd6cfb4990548c43a4b1e50108b1f1357d51`
- coefficient logical SHA:
  `859512d721cd7e543efb2596ae25cead136181e361dd4b64a6fcbc50af14e8bc`
- truth rows before freeze: 0
- model SHA / manifest SHA: modelなし
- prediction SHA: -
- submission SHA: -
- rerun result: rescue / version 2なしでclose

## 解釈

full-prefixでは`rho_gr < 0.50`が8 wellsしかなく、係数surfaceは事前固定した
non-degeneracy下限を大きく下回った。last-512との係数一致`0.6662`とSpearman`0.1675`も
固定下限を下回り、prefix Pearsonをwell-level reliability selectorに使う根拠は得られなかった。
coverageとleakage guardは正常であり、失敗は入力欠損や実装異常ではなく科学的gateの不成立である。

## 次

threshold、multiplier、support、window、相関種を調整せずcloseする。Stage 1、
inference、submission、version 2、同familyのrescue backlogは作らない。
