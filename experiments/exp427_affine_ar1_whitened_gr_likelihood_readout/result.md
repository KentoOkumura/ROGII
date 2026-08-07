# exp427_affine_ar1_whitened_gr_likelihood_readout 結果

## 状態

Kaggle private CPU version 2でStage 0を完走した。technical / scientific gateは
ともにFAILし、`stage_0_failed_close_without_rescue`として閉じた。
prediction、inference、submissionは生成していない。

## 仮説

known prefixから得たaffine係数のposterior uncertaintyと、outer-train foldから
固定したAR(1) residual covarianceを含むblock Gaussian predictive likelihoodは、
exp280の行別raw-Gaussian scoreよりtruth-nearest shiftの順位識別力が高い。

## 設定

- 親: `exp280_exp226_shift_likelihood_separability_readout`
- Route: `pf_beam`
- 検証: exp280固定13 shifts × 非重複512-row blocks
- 要因分解: `identity/affine × iid/AR1`
- primary: `affine_ar1`
- metric: MRR / top3、fold / 1000+ / hidden-like / top1-regret p90
- seed: real score RNGなし、negative controlのみstable SHA256 local RNG
- Stage 0実行量:
  score 4 + saved control 1、5 reporting folds、
  HMM / PF / Beam / model / booster / GPU各0

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 | technical FAIL / scientific FAIL |
| evaluated wells / blocks | `697 / 5,615` |
| eligible block率 | `0.721073584`（`>=0.75` FAIL） |
| `identity_iid_matched` MRR / top3 | `0.388002620 / 0.450400712` |
| `affine_iid` MRR / top3 | `0.385603633 / 0.437577916` |
| `identity_ar1` MRR / top3 | `0.386476559 / 0.451291184` |
| `affine_ar1` MRR / top3 | `0.386090045 / 0.439180766` |
| saved exp280 MRR / top3 | `0.388146378 / 0.449866429` |
| shuffled MRR / top3 | `0.235872482 / 0.223152271` |
| runtime / peak RSS | `4,358.768411秒 / 1.264053 GB` |
| CV | なし |
| Public LB | なし |
| Private LB | なし |

primary差分:

| 比較 | MRR差 | top3差 | 改善fold |
| --- | ---: | ---: | --- |
| matched identity-iid | `-0.001912575` | `-0.011219947` | MRR `2/5`、top3 `1/5` |
| saved exp280 | `-0.002056333` | `-0.010685663` | MRR `2/5`、top3 `1/5` |
| affine-iid | `+0.000486412` | `+0.001602850` | MRR `2/5` |
| identity-AR1 | `-0.000386514` | `-0.012110419` | MRR `2/5` |

stress scopeでは、long-tail 1000+のprimary MRR / top3は
`0.370157071 / 0.418129305`でmatched `0.373882285 / 0.432126194`とsaved
`0.373567032 / 0.431459676`の双方を下回った。hidden-like spatial / typewell-purged
はprimary MRRが両controlを上回った一方、top3は両controlを下回った。
top1-regret p90もprimary `39.852949`、saved `38.499431`で悪化した。

fold rhoは`0.749083`から`0.754092`で、outer-valid source overlapは全fold 0。
real scoreはshuffleをMRR / top3とも5/5 foldsで上回った。

## 再現性

- deterministic anchor: いいえ
- seed policy: real score RNGなし、negative controlのみstable local RNG
- kernel version: `kentookumura/exp427-affine-ar1-whitened-gr-readout-train` v2
- kernel id_no: `128931242`
- scientific contract SHA:
  `75241052d0bdeba3dcbad6548167bb1193f4375b1035e8de625591d4fdb24773`
- target-free bundle SHA:
  `3cae530e8c2629eea16468383ae06edc3e971d1ed77fb3a4d8d71d4043ba8a4d`
- target-free score content SHA:
  `62f8d44475666552ad046e3c093f10f66ed7f62f00376266d117aebc93d87050`
- eligibility content SHA:
  `6bc27561921873603e45293ebaff82a242e7cbbb9e20a67c6bc5cca7ed5cfa65`
- prefix posterior content SHA:
  `6f751896ac059877ad7455cd898aa8572e16e346672b448e545a17e4ebaeb855`
- fold rho content SHA:
  `e9f042b05cc12240b2fd08cd02cf9a5f54b1c5d10ab53779f51d1a7c2cf77aeb`
- model SHA / manifest SHA: 対象外
- prediction SHA: 対象外
- submission SHA: 対象外
- truth pre-freeze read: `0`
- hidden-role pre-freeze read: `0`
- rerun result: version 1は空table schema実装エラー、同一科学契約のversion 2で完走

## 解釈

fold-safe AR1係数は安定して推定でき、negative controlに対する順位信号も残ったが、
prefix affine uncertaintyとAR1 whiteningの組合せはmatched / saved controlを上回らない。
affine main effectはtop1をわずかに増やす一方でtop3とtailを悪化させ、AR1 main effectも
top3だけをわずかに改善してMRRを改善しない。両者の複合で相殺される追加価値はない。

technical FAILはraw-finite supportの事前coverage gateだけであり、実装やleakageによる
偽の科学FAILとは考えにくい。さらにeligibleな5,615 blocks上でもprimaryが両controlを
明確に下回るため、coverage gateを緩和しても仮説は支持されない。

## 実装した検証

- current-well known-prefix finite pairだけを使う解析的Bayesian affine posterior
- 元missing位置をまたがないlag-1 pairによるper-well Yule-Walker rho
- outer-valid wellを除外したfold共通Fisher-z median rho
- stationary AR(1) whiteningとrank-2 Woodbury / determinant lemmaによる
  Gaussian posterior-predictive log density
- `identity_iid_matched` / `affine_iid` / `identity_ar1` / `affine_ar1`の固定2×2
- stable SHA256 local RNGによるblock単位negative control
- score / eligibility / posterior / rho / control / manifestのcontent SHA freeze後だけ
  truthとhidden-like roleを読むledger
- pooled / fold / 1000+ / hidden-like / top1-regret p90の固定AND gate

version 1のscore対象外wellで発見した空table schema欠陥を修正し、回帰testを追加した。
最終専用pytestは15件。小さいfactorial / fold / scope / rho / gate / summaryだけを
ローカルへ取得し、291,980-row score archive全体は取得していない。

## 次

exp427内のprior、rho、support、block、shift、score family、gateをsame-OOFで救済せず、
rerun、decoder、prediction、inference、submissionへ進まない。exp427完全PASSを
前提にした条件付きexp431も閉じる。

次候補は低優先度P4のsaved-artifact-only失敗原因分解に限定する。目的はaffine / AR1の
top3・tail悪化がprefix calibration、rho、finite supportのどの事前固定regimeで
一貫するかを説明することであり、exp427を再開・昇格するためのparameter探索には使わない。
