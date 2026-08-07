# exp348_prefix_gr_unary_window_path_ranking_ssm

## 状態

- Route: `ensemble`
- 状態: `stage0_failed_branch_closed`
- 優先度: P3高リスク、完了・非昇格
- 親: terminal closedの`exp332_prefix_gr_unary_fixed_window_structured_ssm`
- Kaggle: private T4 version 2、id_no `128524049`

## 仮説

256-row window内で全状態のpartition functionを計算せず、truthに近いlegal pathが固定16 negative pathsより高得点になるよう学習すれば、pointwise CEより経路構造を保ちつつexp332の4-sweep DPを学習loopから除去できる。

## 単一変更

- structured NLL `1.0`を、margin`0.05`のpath-ranking loss `1.0`へ置き換えた。
- local CE`0.25`、window、boundary、architecture、fixed exp209 grammar、full-well exact decodeは維持した。
- positive/negative path bankはfit前にfreezeし、model/error依存のhard-negative miningを行っていない。

## 検証方針

固定16-window T4 Stage 0でpath bankのlegality/unique count/SHA、early-holdout ranking、runtime、memoryを固定AND gateで評価した。Stage Aへ進むにはStage 0全条件のPASSと別承認を必須とした。

## Stage 0結果

固定16 windowsを12 optimizer / 4 early holdoutに分け、1 temporary modelだけで評価した。永続model、trained fold、LightGBM config、booster、PF/Beam、親/control再学習はすべて0。

| Gate / 指標 | 結果 | 基準 |
| --- | ---: | ---: |
| Technical | PASS | 全check |
| unique negatives | 全windowで16 | `>=12` |
| Early-holdout positive top-1 | `0.0` | `>=0.80` |
| Positive − max-negative margin | `-0.388485` | `>=0.02` |
| 保守的fold runtime外挿 | `75.356700 h` | `<=8.5 h` |
| peak GPU memory | `1.193590 GB` | `<=14 GB` |
| Stage 0 AND gate | FAIL | 全条件PASS |

version 1はraw CSVの存在しない`id`列を仮定して学習前に停止した。version 2で正規ID契約`{well}_{row_index}`へ修正し、科学条件を変えずに完走した。

## 所見

exact partition sweepは0にでき、memoryも十分小さかった。一方、fit前path-bank生成が計算量を支配し、短いoptimizer benchmarkでは未見windowのpositive pathを上位化できなかった。runtimeとlearningの両面で仮説を支持しない。

## 判断

technicalとmemoryはPASSしたが、learningとruntimeが独立に大幅FAILした。`close_without_negative_bank_margin_or_science_rescue`としてbranchを閉じ、negative family/count、margin、loss、window、architecture、decoder、epochの救済は行わない。Stage A/B/C、推論、提出も実施しない。

再検討する場合は、per-window Viterbi path bankを持たない局所transition-consistency surrogateを別実験・別Stage 0として事前設計する。

## 次のアクション

exp348としての次工程はない。上記の独立案を検討する場合も、新しいsteeringと実装・実行承認を必要とする。

## 文書

- Steering: `../../.steering/20260722-exp348-prefix-gr-unary-window-path-ranking-ssm/`
- 設定: `config.yaml`
- 詳細結果: `result.md`
- 実行記録: `SESSION_NOTES.md`
