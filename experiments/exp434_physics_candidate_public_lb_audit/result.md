# exp434_physics_candidate_public_lb_audit 結果

## 状態

compact self-contained inferenceを正規Notebookへ採用し、通常9候補と
LikPF同一性FAILによる条件付き1候補の計10 Kaggle versionを完了した。
10/10でoutput取得とsubmit-checkまでPASSし、凍結順序のversion 1–10を
competition submissionした。全10件のPublic LBが確定し、既存LBを
同一性確認後に再利用した2候補と合わせて、全12候補のLB censusが完了した。

## 仮説

exp263の物理候補bankを固定したままPublic LBを測ると、OOFで確認した
primitive間の精度差とblend補完性がLBでも維持されるかを記述できる。

## 設定

- 親:
  `exp263_last_anchor_better_candidate_confidence_pair_cache`
- route:
  `pf_beam`
- OOF:
  3,783,989 rows / 773 wells
- candidate:
  6 primitive + 5 fixed 50:50 pair + 1 fixed 50/25/25 blend
- training:
  なし

## 候補表

| 候補 | 種別 | OOF RMSE | Public LB | LB - OOF | OOF順位 | LB順位 | 状態 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `exp226_w500_50_50` | fixed | 8.238331 | 7.800 | -0.438331 | 1 | 2 | 既存exact parity |
| `exp226_k16__selfgr_hmm_a070` | pair | 8.532715 | 7.913 | -0.619715 | 2 | 3 | ref `55083262` COMPLETE |
| `exp226_k16__exact_hmm` | pair | 8.635074 | 7.678 | -0.957074 | 3 | 1 | ref `55083266` COMPLETE |
| `exp226_k16__likpf_mean` | pair | 8.813822 | 8.365 | -0.448822 | 4 | 4 | ref `55083270` COMPLETE |
| `exp226_k16` | primitive | 9.427110 | 9.837 | +0.409890 | 5 | 10 | 既存、同一性gate PASS |
| `selfgr_hmm_a070__likpf_mean` | pair | 10.123457 | 8.812 | -1.311457 | 6 | 6 | ref `55105249` COMPLETE |
| `likpf_mean__exact_hmm` | pair | 10.269697 | 8.642 | -1.627697 | 7 | 5 | ref `55105256` COMPLETE |
| `selfgr_hmm_a070` | primitive | 11.349943 | 9.318 | -2.031943 | 8 | 8 | ref `55105261` COMPLETE |
| `likpf_mean` | primitive | 11.594898 | 9.807 | -1.787898 | 9 | 9 | ref `55133074` COMPLETE、SHA256 seed版 |
| `exact_hmm` | primitive | 11.938287 | 9.063 | -2.875287 | 10 | 7 | ref `55105266` COMPLETE |
| `pf_ancc` | primitive | 14.493051 | 12.061 | -2.432051 | 11 | 11 | ref `55133068` COMPLETE |
| `beam_mean` | primitive | 15.774327 | 15.563 | -0.211327 | 12 | 12 | ref `55133072` COMPLETE |

## 再現性

- deterministic anchor:
  false、PF familyを含むため。全versionでcandidate bank SHA一致
- seed policy:
  exp073 stable SHA256 per-wellを継承
- kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- kernel version:
  1–10
- feature / prediction / submission SHA:
  `kaggle_run_ledger.json`へversion別に記録
- model SHA:
  非該当
- 実装済みguard:
  4 generator source SHA、exp226 config SHA、Stage 0 / Stage 1 parent SHA、
  exposed reference decompressed SHA、row/ID/finite、formula parity、
  existing submission SHA、prediction/submission/candidate-version SHA

## 実装検証

- compact self-contained Jupytext source / candidate Notebook: 作成済み
- 正規Notebook: compact self-contained版を採用済み
- exp434専用test: 8件PASS
- exp263 + exp434関連test: 22件PASS
- Jupytext test / py_compile / Ruff F821: PASS
- 保存済みexp263 v3 formula bankとの全12候補parity: 最大差`0.0 ft`
- 既存submission read-only gate:
  fixed 3-way `0.000484375 ft`、K16 `0.000488265 ft`で`0.001 ft`以内
- existing submission gate:
  K16 `0.000488265 ft`でPASS、fixed `0.000484375 ft`でPASS、
  LikPF `4.7783203125 ft`でFAILし既存LB流用を禁止
- model config / trained fold / booster / parent retraining:
  `0 / 0 / 0 / 0`
- Kaggle run / output / submit-check / competition submission:
  `10 / 10 / 10 PASS / 10 submitted（10 COMPLETE）`
- 共通実行監査:
  14,151 rows / 3 wells / fallback 0、親式parity最大差`0.0 ft`、
  candidate bank SHA
  `870f0795649ee21852679176f313efb668bf7aec0c3262681a17c04b33eca03d`

Repo全`make test`は、exp297 / 301 / 333 / 336 / 349の既存config locatorに
起因する5 collection errorで停止した。exp434由来のtest failureはない。

## 解釈

全12候補のLB最良は`exp226_k16__exact_hmm`の`7.678`で、固定3-way
`7.800`、K16 + self-GR HMM `7.913`が続いた。全12候補のOOF/LB Spearman
順位相関は`0.846154`で、OOF順位は全体として有用だが完全には一致しない。

最大の順位逆転はK16で、OOF 5位からLB 10位へ下がった。一方、OOF 10位の
`exact_hmm`はLB 7位へ上がり、primitiveでは`exact_hmm 9.063 <
selfgr_hmm_a070 9.318 < likpf_mean 9.807 < exp226_k16 9.837 <
pf_ancc 12.061 < beam_mean 15.563`となった。LikPF pairでもOOFではself-GR
pairが上だったが、LBは`likpf_mean__exact_hmm` `8.642`が
`selfgr_hmm_a070__likpf_mean` `8.812`を上回った。

SHA256 seed版LikPFのLBは`9.807`である。exp069 v3のBLAKE2b per-well seed版
`9.721`より`0.086`悪いが、両者は別Monte Carlo realizationであり、既存値を
exp434候補へ流用しない。同一性監査ではcurrent-test差分RMSE`1.723127 ft`、
最大差`4.7783203125 ft`だった。Public LBは記述的な監査結果として扱い、
この結果からweight tuningや候補の自動採用は行わない。

## 次

12候補のPublic LB censusは完了した。この表を後続のdirect physical LB候補との
比較根拠に使うが、weight tuningやtrain-side採用には使わない。exp452のscale5
LikPFは別候補であり、未提出・Public LB未評価のまま扱う。
