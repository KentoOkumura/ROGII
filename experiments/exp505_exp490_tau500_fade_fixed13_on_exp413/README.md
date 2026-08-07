# exp505_exp490_tau500_fade_fixed13_on_exp413

## 状態

- ルート: `ensemble`
- 状態: Stage C完了・scientific gate FAIL・終端閉鎖
- CV: hard OOF `8.243315437`
- Public / Private LB: 未提出 / 未提出
- Kaggle: private CPU version 1、id_no `129519165`、`COMPLETE`
- selector親: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- downstream親: `exp413_scale5_likpf_full_replacement_on_exp335`

## 仮説と変更

exp501 fixed13のraw exp490 slotだけを、次の固定fade候補へ1対1置換した。

```text
w = 1 - exp(-md_since / 500)
p_fade = p_exp357 + w * (p_exp490 - p_exp357)
```

候補数13、fixed12、fixed7 fallback、outer 5 × inner 4、2 objectives、sampling、
LightGBM設定、固定scopeはexp501から変更していない。raw exp490とfadeを併存させず、standalone
predictionやfinal TVTへの直接fadeも行っていない。

## 検証方針

saved raw exp501をcontrolとし、strict-nested hard OOFのpooled / fold / 固定7 scopeを比較する。
technical checks、4/5 folds、fade利用に加え、fixed12比by-well p95を`0.10 ft`、worstを
`1.0 ft`以上縮小する全AND gateを事前固定した。Stage C PASS時だけStage Dを別承認対象とし、
FAIL時はsame-OOF rescueなしで閉じる。

## 結果

- exp505 hard OOF: `8.243315437`
- raw exp501 hard OOF: `8.264890209`
- gain: `0.021574771 ft`
- nonworse folds: `4 / 5`
- 固定7 scope: `7 / 7`改善
- fade top1利用率: `55.2414%`、positive folds `5 / 5`
- technical / leakage checks: 全PASS

しかし、fixed12比tailの縮小はp95 `0.000036536 ft`、worst `0.173168079 ft`で、必要な
`0.10 / 1.0 ft`を大きく下回った。Stage C gateは
`FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`。

## 実行量

- 1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boosters
- control selector再学習: 0
- HMM / PF / Beam再実行: 0
- Stage D GPU booster: 0
- inference / submission: 0 / 0

## 成果物

- 正規train Notebook: compact self-contained Stage Cを採用済み
- candidate / feature contract、5件のcontract test
- Kaggle output: feature/model/compact manifest、40 model、25 compact partitions、
  outer-valid score、fold/scope/by-well/usage/gate readout
- 正規inference Notebook: placeholderのまま

主要SHAとfold/scope値は`metrics.json`と`result.md`に記録した。output archive全体は取得せず、
logsと記録に必要な小artifactだけを確認した。

## 所見

tau=500 fadeはraw exp501より平均をわずかに改善したが、well-tailをmaterialに安全化しなかった。
事前契約どおりStage Dを実装・実行せず、同一OOFでtau / alpha / threshold / feature / gateを
救済しない。inferenceとsubmissionも行わない。
