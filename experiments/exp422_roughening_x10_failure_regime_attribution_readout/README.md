# exp422_roughening_x10_failure_regime_attribution_readout

## 状態

- ルート: PF/Beam
- 状態: Kaggle CPU audit完了・scientific FAIL・branch close
- CV / Public LB / Private LB: - / - / -
- Submit ID: -
- 作成日: 2026-07-28
- 親実験: `exp416_roughening_x10_likpf_full_oof_ablation`
- Kaggle kernel:
  `kentookumura/exp422-rough-x10-regime-attribution-train` version 2

## 仮説

exp416のroughening x10は一律適用では失敗したが、PFの回復圧力が高く、
GR欠損と長いsuffixによる損傷露出が低いwell regimeでは改善する。これが
persistent-offsetの局所回復と全体悪化の分岐を説明する。

## 変更点

- exp416 / exp072 / exp226の保存生成物だけを読む0-PF readout。
- outcome読込前に4診断の`recovery_pressure_score`、2診断の
  `damage_exposure_score`、fold-safe中央値、1つのtarget cellをfreezeする。
- suffix進捗4分割、raw-GR observed / missing、1000 ft以遠をsecondaryに読む。
- scoreごとに4096回のfold内置換を行う。
- adaptive roughening、model、prediction、inference、submissionは実装しない。

正規train notebookはcompact self-contained Jupytext sourceから採用済みである。
canonical inference notebookはplaceholderのまま、inferenceは無効である。

## 検証方針

- exp226 reporting folds 0--4、groupは`well_id`。
- 773 wells / 3,783,989 rowsとexp416 / exp072のpooled・by-well parityを必須とする。
- recovery-pressureとdamage-exposureの方向、4/5 fold再現性、固定target cellの
  row / equal-well gain、persistent episode supportをAND評価する。
- regime assignmentとrow scopeのschema / logical SHAをtruth attachment前に保存する。

## 実行量

- saved-output readout: 1
- reporting folds: 5
- new prediction / PF / LightGBM / booster / HMM / Beam / GPU: すべて0
- rows / wells: 3,783,989 / 773
- runtime / peak RSS: 362.877 sec / 3.298 GiB

## 結果

technical gateはPASSしたが、scientific gateはFAILした。

| メトリック | 値 |
| --- | ---: |
| candidate - control RMSE | +2.022823162 ft |
| recovery-pressure rho / p | -0.166697697 / 1.000000000 |
| recovery-pressure positive folds | 0 / 5 |
| damage-exposure rho / p | -0.041484753 / 0.111300952 |
| target-cell row RMSE gain | -1.852449584 ft |
| target-cell improved folds | 1 / 5 |
| target-cell equal-well gain vs rest | -0.518965568 ft |
| target-cell improved-well fraction | 0.314049587 |
| target-cell episode SSE gain | 45.801967% |
| target-cell share of positive episode gain | 39.400617% |

recovery-pressureは仮説と逆方向に5/5 foldsで関連し、fixed target cellもglobalには
悪化した。episode局所gainは残ったが、事前固定したregimeでは十分に集約できなかった。

## 再現性

- technical gate: PASS
- scientific contract SHA:
  `20d2644085334ed0028ff8ca0caa38d6379073980f3547c6d05b1f7eee410426`
- artifact manifest SHA:
  `c2fe9339994e8785bf33dc0585f985d5a819e1ff6bf653262bad46e108c04f16`
- feature / assignment / row-scope logical SHAは`metrics.json`に記録した。
- version 1の親logical SHA列契約不一致は、科学設定を変えずversion 2で修正した。

## 所見

- 良かった点: truth-late freeze、入力SHA、parity、0-PF / 0-model実行量を含む
  technical contractはすべてPASSした。
- 悪かった点: recovery-pressureは事前仮説と逆方向で、固定target cellもglobalには
  悪化した。
- 注意: persistent episode内の改善だけを使ってrow-level triggerやadaptive policyを
  正当化できない。

## 結論

exp416のterminal FAILを維持し、exp422のtarget-free attribution branchも終了する。
同一OOF上のscore、threshold、cell、roughening parameter救済は行わず、inference /
submissionにも進まない。
