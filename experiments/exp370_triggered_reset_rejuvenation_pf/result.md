# exp370_triggered_reset_rejuvenation_pf 結果

## 仮説

target-free triggerとfold-safe atlas coverageが成立する時だけ10%再注入すれば、
PF mode slipを回復できる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- Route: `pf_beam`
- Stage 0: 500 particles × 1 seed × 773 wells = 773 diagnostic seed-well runs
- reporting folds: 5
- scientific variant / full parent PF replay / LightGBM / booster: 0 / 0 / 0 / 0
- Kaggle: private CPU、internet off、version 2、id_no `128591535`

## 結果

Kaggle version 2は`COMPLETE`。technical gateは全項目PASSしたが、
scientific gateは全項目FAILし、
`stage0_failed_close_without_rejuvenation_pf`となった。Stage 1は不適格。

| 指標 | 結果 | gate |
| --- | ---: | ---: |
| accepted triggers / eligible rows | 13 / 3,685,818 | - |
| trigger row fraction | 0.000003527 | 0.001–0.10 |
| trigger bad-event AUC | 0.499998 | >= 0.60 |
| AUC gain vs circular | -3.76e-12 | >= 0.05 |
| atlas top-3 within 10 ft coverage | 0.076923 | >= 0.60 |
| saved likPF within 10 ft coverage | 0.846154 | 比較対象 |
| coverage gain vs saved likPF | -0.769231 | >= 0.10 |
| passing folds | 0 / 5 | >= 4 / 5 |

hidden-like spatial / typewell-purgedはいずれもevent 1件、atlas coverage 0、
saved likPF比coverage gain -1.0で、両方向正のgateもFAILした。

version 1はcompetition input mount resolverが`test`を選び得る欠陥により、
科学計算前のwell identity guardで停止した。PF runは0。resolverを
`/kaggle/input/competitions/<slug>/train`優先、`train`限定、paired 773-well
guardへ修正し、version 2で全件実行した。

## 再現性

- scientific contract SHA:
  `4546b84c6ca6c3fa71fee3378d46b38101ece3bac1da94f817ea87712abcf875`
- summary SHA:
  `3ce97d6b11d8bec962b67df600b0e67f17f8cb82d2a465b48ee343517d652075`
- gate report SHA:
  `941564dfb9b596111982a47f7b33d1d1c9308c93a3995ae2c7bd61b02fe00821`
- trigger ledger decompressed SHA:
  `7abd280aca86ae4727a893bcf848d7e236afaaf1ca8234ea09930cb06227a666`
- proposal ledger decompressed SHA:
  `58511f5376fe893912b3bf5f70b6a1fcfc977cff0b2e9205e4681687a3216a8f`
- atlas prototype decompressed SHA:
  `ad2635c9f74412fe4b82b20bad88c81a53d2ec4b293796775cc4bdfdd2c1633d`
- saved exp072 cache decompressed SHA:
  `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- runtime: 671.342秒、Python 3.12.13、NumPy 2.0.2、pandas 2.3.3
- target truth / hidden role before freeze: 0 / 0
- donor-fold leakage violations: 0

## 解釈

失敗の主因は二重である。

1. q99.5 GR changeとESS/N `<=0.20`のANDは13行しか残さず、
   event率が下限の約283分の1まで退化した。bad-event識別もrandom相当だった。
2. 稀に発火した行でもatlas proposalの平均best absolute errorは263.11 ftで、
   saved likPFの5.73 ftを大幅に下回った。top-3 coverageも7.69%に留まった。

したがって、閾値緩和やatlas top-k増加で同じ仮説を救済せず、
10% particle rejuvenation PF branchは閉じる。

## 次

Stage 1、inference、submissionは実装・実行しない。将来reset系PFを再検討する場合も、
まず0-PF / 0-atlasの独立readoutでGR-changeとESSのjoint supportが非退化かを
確認し、exp370のgateや設定は再利用しない。
