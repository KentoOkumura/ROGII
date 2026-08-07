# exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264

## 状態

- ルート: MLモデル
- 状態: Stage 0完了・scientific gate FAIL・branch閉鎖
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-26
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 補助入力: `exp368_marginalized_reliability_pf`

compact self-contained trainを正規Notebookへ採用し、Kaggle private CPU
version 4でStage 0を完了した。technical gateは全PASSしたがscientific
all-AND gateはFAIL。model、booster、PF、prediction、submissionは0である。
fail-closed inference候補は実行していない。

## 仮説

exp368はPFへ進むgateにはFAILしたが、保存suffix blockのbad10 AUCは
`0.636675`、circular差は`+0.058264`、5/5 foldsとhidden-like 2面でも
識別力が残った。このtarget-freeなweak posteriorが、exp264の既存scoreが
別候補を正しく指名できる「回復可能なlikpf失敗区間」を識別できれば、
後段selectorの連続context featureとして役立つ。

## 変更点

検証した候補は1列だけ。

```text
ctx__exp368_weak_risk(row)
  = mean(weak_posterior_mean(block)
         for every exp368 block covering row)
```

- block 512、stride 256、tail keep、overlap blockは等重み。
- threshold、hard router、変換、平滑化、clip、gridは使わない。
- exp368 Stage 1 PFを再開せず、保存済みtarget-free artifactをload-onlyする。

## 検証方針

### Stage 0

0 model / 0 booster / 0 PF / 0 predictionのreadoutである。

1. weak riskとstable circular controlをrowへ集約し、truth前にSHA freezeする。
2. exp264 corrected Stage C v6のstrict-nested scoreを読み、
   `primitive_pair_bank` 11候補と`primitive_fixed_bank` 7候補を分離する。
3. `likpf_mean`が10 ft以上外したrowだけで、既存`pred_abs_error`が指名した
   other candidateが10 ft未満へ戻すかを`nominated_recovery10`とする。
4. AUC、circular差、fold、hidden-like、既存selector margin条件付きAUC、
   weak-risk Q4-Q1のrealized advantageを全AND判定する。

primary gateはpooled AUC `>=0.60`、circular差`>=0.02`、4/5 folds、
hidden-like両面`>=0.55`、margin-conditional AUC `>=0.55`、
Q4-Q1 advantage `>=0.50 ft`である。詳細はsteeringと`config.yaml`を正とする。

### 条件付きStage 1

Stage 0全gate PASSかつ別承認後だけ、exp264 Stage Cの88列へweak risk 1列を
add-onlyする。

- 1 variant
- LightGBM config 1
- objectives 2
- outer 5 × inner 4
- 合計40 CPU selector boosters
- parent/control再学習0
- PF/HMM/Beam replay 0
- downstream TVT / GPU / inference / submission 0

学習前にraw train replayのlogical-content parityと、
raw current test 14,151 rows / 3 wellsのfinite・100% coverageを必須にする。

## 所見

### この実験で分かること

「GRを弱める閾値をどこに置くか」ではなく、exp368 riskが既存selector scoreを
補完して、切替可能な誤差だけを見つけられるかが分かる。Stage 0 PASSだけでは
RMSE改善を主張せず、Stage 1 selector scoreまでPASSして初めてdownstream検証を
別途検討する。

### Stage 0結果

- Kaggle: version 4 / id_no `128626512` / COMPLETE
- runtime: 129.300秒
- technical: 15/15 PASS
- scientific: 4/12 PASS、総合FAIL
- primary pooled AUC: `0.520214`
- primary circular差: `-0.003253`
- primary margin-conditional AUC: `0.458846`
- primary hidden-like: `0.527468 / 0.513626`
- primary Q4-Q1 realized advantage: `+3.879372 ft`
- decision: `stage_0_failed_close_without_rescue`

## 禁止事項

- exp368 FAILの再分類、transition / sigma / block / threshold救済
- 12候補の単一hard domain化、candidate subset選択
- truth/errorを見たfeature、bucket、gate調整
- AUCを見た後のscore反転
- Stage 1、Stage D、inference、submission

## 実行入口

実装は次のJupytext source / fail-closed inference候補にある。

- `exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_train.py`
- `exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_compact_selfcontained_inference.py`

train候補は11章構成で、入力SHA、overlap block平均、exp264 Parquet
row-group scan、truth-late join、AUC / margin-conditional AUC /
cross-fit quartile、全AND gate、生成物SHAまでNotebook上で追える。
45,407,868 candidate-long rowsはrow group単位で読み、12候補TVTだけを
一時float32 memmapへ置く。inference候補は常にfail closedであり、
submissionを生成しない。

実装検証は専用test 9件、Jupytext round-trip、py_compile、Ruff
`F821/F401/F841/E9`、strict experiment validationをPASSした。
Kaggle outputは`kaggle/output/train_v4`へ取得し、feature、scope metrics、
nomination distribution、summaryのSHAを実ファイルで照合した。Stage 0 FAILに
よりStage 1へは進まない。
