# exp362_segment_local_donor_slope_exact_hmm

## 状態

- ルート: pf_beam
- 状態: completed_postrun_support_audit_failed_closed
- CV: 11.161677（prefix-rate-onlyへ完全退化した参考値。意図したlocal donor-slope評価としては無効）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp226 の予測値ではなく、「坑跡を区間に分け、空間的に近い donor 区間の傾きを補間する」という
考えを exact HMM の rate transition mean に直接入れる。近傍井から得た K16 の局所地層勾配を
target 坑跡方向へ射影し、残差 rate だけを exp209 と同じ HMM で追跡すれば、constant rate prior
より長い未知 suffix の drift を減らせる可能性がある。

## 変更点

- outer-train の raw truth から K16 donor segment の `d(TVT+Z)/dMD` と坑跡方向を作る。
- target の未知 suffix も K16 に分け、各区間で donor well ごとに最近傍 segment を 1 個選び、
  近傍 50 wells、Gaussian bandwidth 500 ft、固定 weighted ridge で局所 2D gradient を推定する。
- target 方向への射影を区間 rate mean とし、MD 上で線形補間して exp209 HMM の時変遷移平均にする。
- support 不足、遠距離、方向情報不足、異常 rate は target 既知 prefix 末尾 30 step の median rate
  へ fail closed する。
- exp226 OOF、`tvt_geop`、`tvt_pred`、GR 補正、kappa、near-strike ANCC、U projection は読まない。
- Stage 0、parameter grid、blend、transition variance 変更、control 再実行は行わない。

## 検証方針

- Fold: raw well id を SHA256 seed 42 で安定整列し、round-robin で 5 fold。
- Group: well id。outer-valid fold の全 wells を donor から除外する。
- Stratification: なし。pooled、fold、距離帯、1000+、hidden-like 2 面、by-well、support/fallback を報告する。
- Leakage Check: donor ledger、rowwise prior schedule、prediction と SHA を freeze してから unknown-suffix
  truth を join する。exp226 artifact resolve 数 0 と truth-before-freeze 0 を hard gate にする。
- Primary: candidate HMM direct RMSE と保存済み exp209 exact HMM direct RMSE `11.938287417 ft` の比較。
- 成功条件: pooled `>=0.05 ft` 改善、4/5 folds 改善、1000+ / hidden-like 2 面 / by-well p95 非悪化、
  worst-well `<=+0.25 ft`。全条件 AND。
- 計算契約: 1 scientific variant、5 reporting folds、773 HMM well-runs、0 model config、
  0 trained fold、0 booster、CPU、control 再実行 0。

## 実行入口

- 実装済み学習候補:
  `exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py` /
  `.ipynb`
- fail-closed 推論候補:
  `exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py` /
  `.ipynb`
- 正規学習 notebook: `exp362_segment_local_donor_slope_exact_hmm_train.ipynb`
- 正規推論 notebook: `exp362_segment_local_donor_slope_exact_hmm_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp362_segment_local_donor_slope_exact_hmm`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
- compact self-contained train候補は正規notebookへ採用済み。
- Kaggle private CPU version 1（id_no `128368310`）はCOMPLETE。
- branch close後は`execution.run_hmm=false`で、同じpackageの再実行を無効化している。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 11.161677 |
| Public LB | - |
| Private LB | - |
| exp209との差 | +0.776610 ft改善 |
| 改善fold | 3/5（gateは4/5以上） |
| worst-well差 | +52.741426 ft（gateは+0.25以下） |
| local-gradient採用 | 0/12,368 segments |

## 所見

### 良かった点

- exp226 の予測を HMM の unary や prior として再利用せず、区間局所傾きという仮説だけを独立に検証できる設計になった。
- 事前診断を通過条件にせず、最初の科学実行を direct HMM 1 variant に固定した。
- fold-safe donor ledger、K16 local-gradient schedule、residual-rate exact HMM、prediction freeze、
  late truth/control join、全 success gate と SHA manifest を 1 本の self-contained notebook 候補に実装した。
- 専用テスト 10 件、exp209 kernel bitwise parity、Jupytext、`py_compile`、Ruff F821、
  strict experiment validation を通過した。
- 3,783,989 rows / 773 wells、outer-valid donor除外、truth-before-freeze 0、
  exp226 resolve 0、parent SHA、posterior正規化、runtimeのnotebook technical gateは通過した。
- pooled、1000+、hidden-like 2面は保存済みexp209より改善した。

### 悪かった点

- 改善foldは3/5で、fold 1/4は悪化した。
- worst well `86454a6f`が`+52.741426 ft`悪化し、固定tail guardを大幅に超えた。
- target prior実ファイルでは全12,368 segmentsがprefix rateへfallbackし、
  finite local gradientは0件だった。したがってCVはlocal donor-slope介入の結果ではない。
- 保存fallback列はlocal-gradient側とprefix-rate側の同名field衝突で全行Falseとなり、
  notebookのfallback countを誤表示した。`fallback_reason`と`mu_rate`の監査で完全退化を確認した。

### リスク / 注意

- 坑跡方向が局所的に揃うと 2D gradient の直交成分は識別しにくい。target 方向情報 gate で fail closed する。
- train/test の空間 support 差と exact HMM の実行時間が主リスク。
- 同じ OOF を見て K、近傍数、bandwidth、ridge、fallback、HMM parameter を調整しない。

## 次

- exp362は完了済み・fail closed。support threshold、bandwidth、fallback、HMM parameterの救済や
  version 2再実行を行わない。
- inference、blend、selector、submissionへ進めない。
- 同じK16 donor supportに依存するexp356は、非退化supportの独立証拠が得られるまで保留する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
