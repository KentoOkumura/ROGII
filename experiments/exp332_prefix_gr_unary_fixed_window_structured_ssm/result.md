# exp332_prefix_gr_unary_fixed_window_structured_ssm 結果

## 状態

Kaggle T4 Stage 0 version 1は技術的に完走したが、固定runtime gateをFAILしたためbranchを閉じた。Stage A以降、推論、提出は未実施。

## 仮説

Gaussian soft-label structured trainingを256-row・最大3 non-overlap windows/well/epochへ限定すれば、window内のtransition-aware learningを残しつつexp295の計算量を実行可能範囲へ削減できる。

## 設定

- 親: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`
- window: 256 rows、3 scheduled slots / 最大3 active per well per epoch
- objective: structured NLL`1.0` + local CE`0.25`
- 評価: official suffixのfull-well exact SSM posterior mean
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |
| Stage 0 p50 fold外挿 | `12.744535682 h` |
| Stage 0 保守的fold外挿 | `13.151137275 h`（FAIL、上限`8.5 h`） |
| peak GPU memory | `1.203262806 GB`（PASS、上限`14 GB`） |
| gate decision | `close_without_window_or_loss_rescue` |

## 実装検証

- Jupytext train/inference変換と`--test`: PASS
- `py_compile`: PASS
- Ruff `E,F,I,UP,B`: PASS
- `make validate-exp EXP=exp332_prefix_gr_unary_fixed_window_structured_ssm`: strict PASS
- `make validate-template`: PASS
- 専用pytest: `14 passed, 1 skipped`
- skip: ローカル環境にPyTorchがない場合のexact structured gradient test
- 全体pytest: `548 passed, 3 skipped, 2 failed`。2件はいずれも既存`exp296`の完了後status/run flagと旧test期待値の不一致で、exp332専用testは全件PASSした。
- parent compact比較: exp295 `2,099`行、exp331 `2,575`行、exp332 `3,045`行。exp332は13章を維持し、window/boundary/Stage 0をNotebook上で追える。
- `__file__`参照: 0
- canonical train Notebook採用: 1

## 再現性

- deterministic anchor: false
- Stage 0 report SHA: `acdadad623784fe8a79bf3fa5d8ae4214b60eb997917d7212adbfe90f2a7ba8e`
- selection / boundary / measurement SHA: report内記録と実ファイルで一致
- model/prediction/submission SHA: Stage A未実行のため未生成

## 解釈

256-row windowでstructured学習rowを絞っても、4-sweep exact DPを8 epochs回すfit部分だけで保守的`9.214264 h`、full-well 3-control decodeが`2.937457 h`を占め、合計`13.151137 h`となった。memoryには十分余裕があるため、ボトルネックは容量ではなくexact DPの計算量である。事前契約ではwindow長/数、boundary、loss、architecture、decoderの救済を禁止しており、Stage Aを実行する根拠はない。

## 次

branch close。Stage A/B/C、推論、提出、同一exp内のcompute rescueは行わない。
