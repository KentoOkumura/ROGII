# 要件

## 依頼

exp347は事前固定したposterior cell max abs error `<=1e-6`をFAILしたためterminal closeの履歴を維持する。一方、loss・partition・AdamW update差0、gradient差`1.4319085e-8`、padding/finite PASS、保守的fold外挿`5.108737 h`、exp332比`2.574244x`という証拠から、差がGPUのfloat32 reduction順による実用上無視できる誤差かを別実験exp393で再検討する。

当初の2026-07-25指示ではbacklog、steering、design-only実験scaffoldまでとした。続くユーザー指示`exp393を実装してください`により、Stage 0コード、正規Notebook、fail-closed inference、専用test、静的検証までを追加承認範囲とする。Kaggle package/push/run、Stage A以降、推論、提出は引き続き行わない。

## 仮説

exp347のscalar/batched posterior cell最大差`1.4662743e-5`は、float32 GPU上でtensor layout、padding、`logsumexp` reduction順が変わったことによる局所的数値差であり、最終posterior mean TVT、MAP state、loss、gradient、optimizer updateには実用上無視できる影響しか与えない。

## 制約

- Route: `ensemble`。neural unaryとfixed exp209 exact SSM posterior readoutの組合せを監査する。
- 親: terminal closedの`exp347_prefix_gr_unary_batched_window_exact_ssm`。exp347のconfig、report、判定、Notebook、生成物を変更・再分類しない。
- Stage 0はexp347と同じ固定16 windows、同じseed 42のtemporary neural unary model 1個を使い、同一run内でunaryを1回生成してfreezeしてから全比較modeへ渡す。
- 比較modeはscalar FP32、batched FP32 batch size 1、production batched FP32 batch size 4。FP64参照は固定先頭4 parity windowsのforward/posteriorだけに限定する。
- Stage 0はnumerical audit 1件。一時model 1、永続model 0、trained fold 0、LightGBM config 0、booster 0、PF/Beam 0、parent/control再学習0。
- 学習科学契約はexp347/exp332を固定する。256 rows、41 rates、objective、Gaussian sigma`0.35 ft`、local CE`0.25`、teacher boundary、architecture、optimizer、full-well decoderを変更しない。
- 推奨されたpractical gateを事前固定し、実行後に閾値を変更しない。posterior cell max abs error `1e-6`は診断値として保存するが、exp393のpromotion gateには使わない。
- Stage 0 PASSかつ別承認がある場合だけ、同じexp393内のStage A fold 0 / architecture 1 / seed 42 / neural model 1を検討する。Stage B/C、推論、提出はさらに別gate・別承認とする。
- Kaggle T4を正とする。直近のKaggle週次GPU quota制約があるため、実行時期またはColab等の別環境を独断で選ばない。
- 再現性は`docs/06_reproducibility.md`に従い、input/config/source/window/boundary/padding/unary/comparison/report/package/kernel SHAを記録する。

## Stage 0 practical gate

固定16 windowsの全valid rowsについて、scalar FP32を比較基準、production batch-4 FP32を候補として次をすべて要求する。

- posterior mean TVT差RMSE `<=0.001 ft`。
- posterior mean TVT差p99 `<=0.005 ft`。
- posterior mean TVT差max abs `<=0.02 ft`。
- marginal MAP state一致率 `>=0.9999`。
- loss / partition max abs error `<=1e-6`。
- unary gradient / AdamW 1-step update max abs error `<=1e-5`。
- invalid row/position/rate posteriorとgradient max abs `=0`。
- posterior、loss、gradient、update、全runtime measurementがfinite、posterior row sum誤差`<=1e-5`。
- outer-valid truth access 0、Stage A model 0。
- peak GPU memory `<=14 GB`、audit runtime `<=1 h`。

batch-1 FP32、posterior total variation、FP64参照との差、MAP disagreement rowのtop-2 margin/state距離は原因帰属用の診断として保存するが、上記gateを置換しない。

## 受け入れ基準

- `docs/legacy/steering/20260725-exp393-exp347-practical-numerical-equivalence-audit/`と`experiments/exp393_exp347_practical_numerical_equivalence_audit/`が存在する。
- `config.yaml`にRoute、親、単一仮説、固定比較mode、Stage 0 gate、実行量、禁止事項、再現性が記録されている。
- `experiment.status=implementation_complete_pending_stage0_execution_approval`で、implementation承認だけがtrue、Kaggle/Stage A/inference/submission承認はfalseである。
- 正規train Notebookはcompact self-contained Stage 0実装、正規inference Notebookはfail-closed実装である。Kaggle package/output/model/prediction/submissionは存在しない。
- `KAGGLE_DIRECTION.md`のbacklogに既存候補との優先順位を含めて追加される。
- deterministic anchorとは扱わない。将来rerun差とSHAを確認した場合だけ再分類する。
- gzip生成物は予定しない。将来追加する場合はdecompressed content SHAを主証拠とする。

## 2026-07-25 Stage Aユーザーoverride

Stage 0 version 2は13 gate中10 PASS / 3 FAILだった。FAILはposterior mean TVT RMSE
`0.007435774 > 0.001 ft`、max差`0.191623403 > 0.02 ft`、posterior row-sum
`2.958618e-05 > 1e-05`。このFAILとexp347のterminal FAILは変更しない。

ユーザー指示`ずれていようがアイデアを採用してStage Aに進みたいです`を、上記3 FAILを
承知した例外的なStage A fold 0の実装・Kaggle T4実行承認として記録する。実行量は
active variant 1 / architecture 1 / fold 0 / seed 42 / neural model 1 /
persisted model 1 / LightGBM config 0 / booster 0 / PF・Beam 0 /
parent・control再学習0。exp347のStage A science gate、256-row window、
batch 4、objective、optimizer、AMP、full-well decoder、real/shuffle/geometry controlを
変更しない。Stage B、推論、提出は未承認。

## Stage A完了判定

Kaggle T4 version 4でfold 0を完了した。real GR RMSE`22.866144493 ft`は
exp209`12.671086935 ft`より`10.195057557 ft`悪化し、well RMSE p95も
`43.017462701 vs 26.301518476 ft`、maximum well regressionも
`75.227871352 > 10 ft`でFAILした。Stage A checksは8/11 PASS・3 FAIL。
decision=`close_stage_b_without_exp347_rescue_grid`とし、Stage B、推論、提出、
同family rescue gridへ進まない。
