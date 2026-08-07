# exp348_prefix_gr_unary_window_path_ranking_ssm 結果

## 状態

Kaggle T4 Stage 0 version 2を完了し、固定AND gate不通過でbranchを閉じた。technicalとmemoryはPASSしたが、early-holdout learningと保守的runtimeがFAILした。Stage A/B/C、推論、提出へは進まない。

## 仮説

256-row windowのpositive pathと固定negative bankをrankingすれば、exact partition functionを計算せずにtransition-awareなpath-level supervisionを与えられる。

## 設定

- 親: `exp332_prefix_gr_unary_fixed_window_structured_ssm`
- Route: `ensemble`
- 唯一の科学変更: exact structured NLLからmargin`0.05`のpath-ranking lossへ置換
- path bank: positive 1、negative最大16、fit前SHA freeze、最低12 unique
- Stage 0: 1 variant、固定16 windows（optimizer 12 / early holdout 4）、一時model 1
- 永続model、trained fold、LightGBM、booster、PF/Beam、親/control再学習: すべて0
- Kaggle: private T4、internet無効、version 2、id_no `128524049`

## 結果

| Gate / 指標 | 結果 | 基準 |
| --- | ---: | ---: |
| Technical | PASS | 全technical check |
| path bank | 16 banks / positive 16 / negative 256 | unique negative `>=12` |
| unique negatives | 全16 windowsで16 | `>=12` |
| outer-valid truth access | 0 | 0 |
| training exact partition sweep | 0 | 0 |
| Early-holdout positive top-1 | `0.0` | `>=0.80` |
| Positive − max-negative margin | `-0.388485` | `>=0.02` |
| 保守的fold runtime外挿 | `75.356700 h` | `<=8.5 h` |
| p50 fold runtime外挿 | `74.228681 h` | 参考 |
| peak GPU memory | `1.193590 GB` | `<=14 GB` |
| Stage 0 AND gate | FAIL | 全条件PASS |
| Stage A model | 0 | 0 |

Kaggle notebook全体のelapsedは約`1566.692 sec`。Stage 0 report SHA256は`2ba5d21934ca1ce49b2e384dd1ea7414f618e4926b4167ac0991d2787fe34c9b`。

## 解釈

経路ランキングはtraining中のexact partition sweepを0にでき、memoryも十分小さかった。一方、fit前に全windowのpositive/negative legal path bankを生成するコストが支配的で、14,816 windowへの外挿だけで保守的`256636.743 sec`を要する。fold全体は約`75.36 h`となり、exp332のruntime問題を解消できない。

learning面でも4 early-holdout windowsすべてでpositive top-1が0、平均marginが`-0.388485`だった。固定negative bankに対する短いoptimizer benchmarkでは、未見windowのpositive pathを上位化できていない。runtimeとlearningが独立に大幅FAILしているため、negative family/count、margin、loss、window、architecture、decoder、epochの救済は行わない。

version 1はraw horizontal CSVに存在しない`id`列を仮定したtechnical bugで学習前に停止した。version 2ではexp209の正規ID契約`{well}_{row_index}`へ修正し、同じ科学条件で完走したため、最終判断はversion 2を根拠とする。

## 再現性・生成物監査

- report記載のselection、teacher boundary、measurement、path-bank summary SHAは取得した実ファイルと一致。
- gzip path-bank manifestのdecompressed SHAは`93a49579cae7b01b79df4349a874ade334d89a4e20e708cead9ca608ee5c3985`で一致。
- exp209 baselineのdecompressed SHAは固定値`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`と一致。
- 大きなKaggle output archiveは取得せず、Stage 0判定に必要な6生成物だけを`/tmp`へ取得した。

## 次

exp348 branchは`close_without_negative_bank_margin_or_science_rescue`で終了する。Stage A/B/C、推論、提出は実施しない。同じwindow-path-bank方式を救済せず、構造学習を再検討する場合は、per-window Viterbi path bankを持たない局所transition-consistency surrogateを独立設計・独立Stage 0で検証する。
