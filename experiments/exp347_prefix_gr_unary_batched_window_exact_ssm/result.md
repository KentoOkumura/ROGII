# exp347_prefix_gr_unary_batched_window_exact_ssm 結果

## 状態

Kaggle T4固定16-window Stage 0 version 1を完了した。計算gateはすべてPASSしたが、scalar/batched posterior parityが固定閾値を超えたため、AND gateはFAIL。事前契約どおりbranchを閉じ、Stage A/B/C、推論、提出、同一exp内の救済は行わない。

## 仮説

exp332の4-window実効batchを、4本同時のpadded/masked exact DPとしてGPU並列化すれば、structured objectiveを変えずに保守的fold runtimeを`8.5 h`以下へ短縮できる。

## 設定

- 親: `exp332_prefix_gr_unary_fixed_window_structured_ssm`
- Route: `ensemble`
- 唯一の変更: `batch 1 × accumulation 4`から`batch 4 × accumulation 1`
- 実行量: benchmark variant 1、固定16 windows、一時neural model 1
- 永続model、trained fold、LightGBM、booster、PF/Beam、control/親再学習: すべて0
- Kernel: `kentookumura/exp347-prefix-gr-batched-window-ssm-stage0` version 1、id_no `128239400`、T4、private、internet off

## 結果

| メトリック | 値 | Gate |
| --- | ---: | --- |
| scalar/batch loss max abs error | `0.0` | PASS (`<=1e-6`) |
| partition max abs error | `0.0` | PASS (`<=1e-6`) |
| posterior max abs error | `1.4662743e-5` | **FAIL** (`<=1e-6`) |
| unary gradient max abs error | `1.4319085e-8` | PASS (`<=1e-5`) |
| AdamW 1-step max abs error | `0.0` | PASS (`<=1e-5`) |
| invalid posterior / gradient max abs | `0.0 / 0.0` | PASS |
| finite rate | `1.0` | PASS |
| p50 fold外挿 | `4.741982 h` | - |
| 保守的fold外挿 | `5.108737 h` | PASS (`<=8.5 h`) |
| exp332比speedup | `2.574244x` | PASS (`>=1.55x`) |
| peak GPU memory | `5.928168 GB` | PASS (`<=14 GB`) |
| outer-valid truth access | `0` | PASS |
| Stage A model | `0` | 予定どおり |
| 総合Stage 0 | - | **FAIL / branch closed** |

実測は52行で、structured train 4 batches、forward-only 4 batches、full-well unary 32 well-control runs、batched exact decode 12 batch-control runs。window/boundary manifestは各16行、batch padding manifestは68行で、report内SHAと取得ファイルのSHAはすべて一致した。

## 解釈

batch化は計算面では成功した。exp332の保守的`13.151137 h`を`5.108737 h`へ短縮し、必要なspeedupとmemory余裕を確保した。ただしscalar/batch posteriorは`1e-6`契約に対して約`14.66x`の差となり、loss、partition、gradient、optimizer updateが一致していてもexact posteriorの等価性を主張できない。

reportの`finite_pass=false`は実測時間やposteriorが非有限だった意味ではない。実測時間は全行で正、parityのfinite checkもPASSしており、実装上`finite_pass = technical_pass AND measurement_finite`のためposterior parity FAILに連動してfalseとなった。

## 再現性

- benchmark report SHA256: `e8a706ba9a75dff54b30b97f289255b002333cb76d2b2dfcac000cfdf56fe454`
- scalar parity SHA256: `3822eddc5dc9f1939f6c22302076e48b97fc1a22be2181f38c427e20fd99051e`
- selection / boundary / padding SHA256: `b78ed92d...1e89` / `664b3fc7...1d1` / `28f30e4e...49b1`
- measurement / log SHA256: `5c3f89eb...e4a8` / `53310887...ed6`
- output: `kaggle/output/stage0_v1`

## 次

exp347はterminal closeとする。batch size、padding、compile/fused kernel、閾値、科学契約の救済や再実行は追加しない。独立仮説のexp348はexp347のterminal decisionという先行条件を満たしたが、高リスクP3の別実験であり、実装・実行はユーザーの別判断とする。
