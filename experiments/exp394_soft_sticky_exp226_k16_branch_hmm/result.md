# exp394_soft_sticky_exp226_k16_branch_hmm 結果

## 仮説

exp226 geometry path と K16-relative full-grid HMMをsoft-stickyに周辺化すると、
両者の異なる誤差区間をGR尤度で選び、固定blendより良い物理TVTを復元できる。

## 設定

- 親: `exp355_exp226_dip_rate_prior_on_exp209`
- branch親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- decoder親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- promotion baseline: exp263 OOF `8.238331546`
- metric: unknown-suffix TVT RMSEとwell-tail / hidden-like guard
- seed: RNGなし

## 結果

実装完了後、Kaggle private CPU canonical version 1（id_no `128536142`）で
fixed16 technical preflightを実行した。

- runtime: `3703.079064 sec`
- scope: `140,721 rows / 16 wells`
- finite prediction / H full-grid coverage: `1.0 / 1.0`
- posterior normalization max error: `4.529710e-14`
- transition row-sum max error: `8.881784e-16`
- projected peak RSS: `1.515934 GB`（上限25 GB、PASS）
- projected full runtime: `112,736.889439 sec`（上限30,600 sec、FAIL）
- truth/error/hidden-like pre-freeze read: `0`
- RMSE / CV / LB: 未計算

technical gateはruntime projectionだけがFAILし、decisionは
`technical_blocker_not_scientific_negative_result`となった。

## 生成物

- compact self-contained train source / 別名Notebook候補
- 固定16-well technical preflightとfull OOFのfail-closed orchestration
- prediction / branch posterior / schedule content SHA契約
- exp263主比較、fold/stress/well-tail/occupancy/switch/recoveryのpromotion gate

Kaggle version 1のpreflight gate、summary、selection、runtime、prediction、
branch posterior、scheduleを生成し、
`/tmp/kaggle-output/exp394_soft_sticky_exp226_k16_branch_hmm/train_v1`へ取得した。

## 再現性

- deterministic anchor: source contractとversion 1実行証拠あり
- summary raw SHA:
  `7f497fe5a44f1bf58d3dab758b6398eeedae986250c6a17b9e6b6c3be3c6321d`
- prediction / posterior / schedule content SHA:
  `b71bf254...710a` / `40558f9e...82a1` / `c05a0764...843b`
- kernel version / id_no: `1 / 128536142`
- submission SHA: submission無効

## 解釈

数値安定性とmemoryは成立したが、fixed implementationのfull projectionは
`31.3158 h`でKaggle上限`8.5 h`の`3.684212x`だった。科学scoreを読んでいないため、
仮説そのもののnegative resultではない。一方、現実装のままexp263超えや
Public LB 6.5を検証することはできない。

## 次

fixed runtime gateに従い、full 773-well OOF、inference、submissionを実行せず閉じる。
再訪は同じfixed16 prediction/posteriorを保つ独立した計算最適化auditに限定する。
