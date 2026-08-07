# exp423_same_typewell_gr_dtw_truth_warp_transfer_readout 結果

## 状態

Kaggle CPU audit 完了。technical / scientific gate は FAIL し、branch を閉じた。

## 仮説

同じ typewell group 内で GR 波形が似た outer-train donor の真の TVT warp は、
query well の最終既知 TVT に再アンカーして移送できる。

## 固定設定

- 親: `exp109_typewell_neighbor_prior_features`
- Route: `pf_beam`
- 検証: `exp099` / `exp109` と同じ 5-fold pseudo-tail OOF
- primary: `analog_top5_median`
- metric: score-row RMSE
- matching: 256-point robust-normalized GR、Sakoe-Chiba band 32、
  axis-run 上限 4、top-K 5
- donor: same `native_overlap=1` group の outer-train well のみ
- seed: 42（matching は乱数なし、negative control だけ stable SHA256）
- 実行量: 1 audit variant、5 reporting folds、0 model、0 booster、
  0 PF/HMM/Beam run、0 GPU run

## Kaggle 実行

- kernel:
  `kentookumura/exp423-gr-dtw-truth-warp-readout-train`
- version 1: 旧 exp099 slug が無効で入力解決前に fail-closed
- version 2: 初回有効 run、`COMPLETE`、772.01 秒
- version 3: 独立 rerun、`COMPLETE`、786.57 秒
- test inference / submission: 対象外・未実行

## 主要結果

| Candidate | Overall RMSE | exp109 比 |
| --- | ---: | ---: |
| `analog_top5_median` | 14.103812714 | +2.960445945 ft（悪化） |
| `analog_top1` | 16.023753292 | +4.880386523 ft（悪化） |
| `stable_random_same_group` | 17.256757094 | +6.113390325 ft（悪化） |
| `analog_top5_oracle_well` | 12.285086482 | +1.141719713 ft（悪化） |
| `exp109_best_fixed` | 11.143366769 | 0 |
| `exp099_likpf_mean` | 11.594901164 | +0.451534395 ft（悪化） |

primary は全 5 fold で exp109 より悪化した。top-5 内の donor を真値で
well 単位に oracle 選択しても全 5 fold で exp109 より悪く、transferability
headroom はなかった。

診断上、`analog_top1` は stable random donor より overall で
`1.233003803 ft` 良く、5 / 5 folds で non-worse だった。一方、DTW cost と
donor path RMSE の pooled Spearman は `0.102226493 < 0.15` で、primary や
oracle の悪化を覆す選択根拠にはならない。

## Scope / tail

| Scope | primary RMSE | exp109 RMSE | 差 |
| --- | ---: | ---: | ---: |
| 1000+ | 15.371422869 | 12.203005921 | +3.168416947 ft |
| hidden-like spatial | 15.897058961 | 13.384878355 | +2.512180606 ft |
| hidden-like typewell-purged | 15.867470897 | 13.241573709 | +2.625897188 ft |

- by-well primary minus exp109 p95: `+14.895650101 ft`
- by-well worst: `+52.735848591 ft`
- supported rows: `1,394,464 / 3,783,989 = 0.368516928`
- supported wells: `286 / 773 = 0.369987063`
- supported path finite fraction: `1.0`

## Technical gate

PASS:

- 全 input SHA 一致
- query truth pre-freeze read 0
- donor/query intersection 0
- row identity unique
- supported path finite fraction 1.0
- independent rerun logical content SHA 一致

FAIL:

- supported score-row fraction `0.368517 < 0.90`
- supported well fraction `0.369987 < 0.90`

technical gate 全体は FAIL。

## Scientific gate

PASS は top-1 対 random の overall / fold consistency と、Spearman の正符号
5 / 5 folds のみ。oracle gain、primary gain、oracle / primary fold consistency、
pooled Spearman、1000+、両 hidden-like、by-well p95 / worst は FAIL した。
scientific gate 全体は FAIL。

## 再現性

- scientific contract SHA:
  `0429bae5f1cb16cc209a4a9e50fdccef1a62ce505591a265115e65bd33c148ca`
- target-free schema SHA:
  `15d9401450f6f8ec4b21971fb7ede0e2874c90e9528a537660fe87ded493f8c5`
- version 2 logical content SHA:
  `6b5b54521ba6612665436f95d4ab3d42c711e8eb18a29bb2ad1916862849d3b3`
- version 3 logical content SHA:
  `6b5b54521ba6612665436f95d4ab3d42c711e8eb18a29bb2ad1916862849d3b3`
- determinism status: `matched_independent_reference`
- scope / fold / by-well / donor-rank / Spearman / fold-separation /
  scientific-gate artifact SHA は version 2 / 3 で一致
- model SHA / manifest SHA: 対象外（0-model）
- submission SHA: 対象外

## 解釈と判断

GR-DTW は同 group 内で random donor より相対的に良い donor を選ぶ弱い信号を持つが、
選択可能な donor pool の coverage が低く、選ばれた top-5 内に exp109 の group prior
を上回る truth-warp path が存在しない。したがって問題は selector だけではなく、
個別 donor の TVT warp を query へ移す仮説自体にある。

事前固定した oracle 不合格の分岐規則に従い、same-typewell donor truth-warp
transfer を PF/Beam candidate、inference、submissionへ昇格しない。top-K、band、
support、group、median/top-1、selector の post-hoc rescue grid は行わず、
exp423 を閉じる。
