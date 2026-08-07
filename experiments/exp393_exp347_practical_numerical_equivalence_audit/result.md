# exp393_exp347_practical_numerical_equivalence_audit 結果

## 状態

Kaggle private T4 Stage 0 version 2は13 gate中10 PASS / 3 FAIL。ユーザーoverrideで進めたStage A fold 0もversion 4で8/11 checks PASS、3 FAILとなった。decision=`close_stage_b_without_exp347_rescue_grid`。Stage B、推論、提出は実行せず、exp347とStage 0のFAILも変更しない。

## 仮説

exp347のposterior cell差はfloat32 GPU reduction/layout差であり、posterior mean TVTとMAPへの影響は実用上無視できる。

## 設定

- 親: `exp347_prefix_gr_unary_batched_window_exact_ssm`
- Route: `ensemble`
- 実行量: fixed16 / audit 1 / FP64診断4 / temporary model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・親control再学習各0
- 比較: scalar FP32、batched FP32 batch 1/4、先頭4 scalar FP64診断
- 主gate: TVT差RMSE`<=0.001 ft`、p99`<=0.005 ft`、max`<=0.02 ft`、MAP一致`>=0.9999`
- Stage A: fold 0 / seed 42 / neural model 1 / persisted model 1、control再学習0

## Stage 0結果

| メトリック | 値 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| posterior mean TVT RMSE | `0.007435774 ft` | `<=0.001 ft` | FAIL |
| posterior mean TVT p99 abs | `0.0 ft` | `<=0.005 ft` | PASS |
| posterior mean TVT max abs | `0.191623403 ft` | `<=0.02 ft` | FAIL |
| marginal MAP一致率 | `1.0` | `>=0.9999` | PASS |
| posterior row-sum max error | `2.958618e-05` | `<=1e-05` | FAIL |
| loss / partition max error | `2.384186e-07` | `<=1e-06` | PASS |
| gradient / AdamW update max error | `1.484295e-08` | `<=1e-05` | PASS |
| finite率 | `1.0` | `1.0` | PASS |
| outer-valid truth access | `0` | `0` | PASS |
| Stage A model | `0` | `0` | PASS |
| audit runtime | `0.044587 h` | `<=1 h` | PASS |
| peak GPU memory | `0.241697 GB` | `<=14 GB` | PASS |

legacy posterior cell max差は`1.519918e-05`でexp347の`1e-6`診断には不一致だが、exp393のpromotion gateには含めていない。batch 1はscalarとTVT RMSE `0`、MAP一致`1.0`だった。

## 実行履歴

- 元slugはGPU session上限拒否後にghost recordとなったため、意味付き短縮slug `kentookumura/exp393-exp347-practical-eq-audit-train`を使用した。
- version 1は親config SHAをmutable repo側へ固定していたため、model/window実行前のidentity guardでERROR。実行済みexp347 Stage 0 config SHA `376c03da...51f`へ訂正した。
- version 2は`COMPLETE`。report SHA256は`14f646a9d835bf0d724dc1efcd59c9dbaa7fdaa28a56417819a45b85877794db`。
- version 3は旧Stage 0専用GPU guardが残っていたため、model生成前にERROR。stage dispatchだけを修正し、科学契約・実行量を変えずversion 4へtechnical retryした。
- version 4は`COMPLETE`。Stage A runtime `3.830431 h`、peak GPU memory `7.495397 GB`。

## 解釈

MAPは完全一致し、差の大部分は0だったが、少数rowのposterior mean TVT差が事前固定した実用等価性上限を超えた。したがってproduction batch 4をpractical equivalenceとして昇格させる根拠にはならない。exp347のFAILを維持する。

## Stage A override

ユーザーは数値差を受容した上でアイデア自体を採用し、Stage Aへ進むことを明示した。これはStage 0 PASSへの変更ではない。fold 0 / seed 42 / neural model 1だけを、exp347と同じ学習条件で実行した。LightGBM、booster、PF/Beam、親・control再学習は0。

## Stage A結果

outer-valid 155 wells / 780,457 rowsを全well予測freeze後に評価した。

| 判定項目 | 値 | 比較・閾値 | 判定 |
| --- | ---: | ---: | --- |
| real GR RMSE | `22.866144 ft` | exp209 `12.671087 ft`より`>=0.25 ft`改善 | FAIL |
| well RMSE p95 | `43.017463 ft` | exp209 `26.301518 ft`以下 | FAIL |
| worst-well regression | `75.227871 ft` | `<=10 ft` | FAIL |
| real vs geometry RMSE | `22.866144 vs 32.465005 ft` | `>=0.25 ft`改善 | PASS |
| real vs shuffle NLL | `14.321158 vs 23.796372` | gain `>=0.05` | PASS |
| real vs shuffle within10 mass | `0.517560 vs 0.181426` | gain `>=0.03` | PASS |
| target-in-grid / finite / prefix clamp | `1.0 / 1.0 / 0 ft` | 各固定条件 | PASS |
| runtime / peak memory | `3.830431 h / 7.495397 GB` | `<=8.5 h / <=14 GB` | PASS |

worst well `44441e54`はreal `76.693843 ft`、exp209 `1.465971 ft`。hidden-like spatial、hidden-like typewell-purged、distance 1000+でもrealはexp209より約10–11 ft悪く、悪化は一部wellだけに限定されない。

一方、real GRはshuffleとgeometryを明確に上回ったため、GR信号を学習できていないわけではない。しかし既存exp209 exact HMM baselineを大幅に下回り、このneural unaryをStage Bへ昇格させる根拠はない。

### 再現性

- freeze前outer-valid truth access 0、forbidden neighbor source 0。
- model SHA: `3c71deec787ea236a562d3e0aa9add68e792a062b5428c6b3921592cbd3ce598`
- frozen prediction decompressed SHA: `6c38315da08e0b0c2c14f62e6f824b95465aa13ce064e29ff10e36f880d01c1d`
- Stage A metrics SHA: `7ec9077952b37a9ac87048ca1531c795bae3b31274580c426d20db7063e2cd45`
- summary SHA: `f55f24931de94ba994e0d2c1ad09e66b4ade39db54a133019240e25aa3ae23e3`

## 次

exp393を`stage_a_failed_branch_closed`として終了する。Stage B、推論、提出、exp347 rescue gridへ進まない。次は既存の高優先度backlogから別仮説を選ぶ。
