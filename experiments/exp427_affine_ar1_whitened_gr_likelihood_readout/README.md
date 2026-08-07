# exp427_affine_ar1_whitened_gr_likelihood_readout

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical / scientific FAIL・terminal close
- 優先度: 低-中 P3
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-28
- 親実験: `exp280_exp226_shift_likelihood_separability_readout`

## 仮説

行別GR差を独立Gaussianとして合算するより、known prefixで得たaffine係数の
posterior uncertaintyと、outer-train foldから固定したGR残差AR(1)共分散を含む
block posterior-predictive likelihoodの方が、exp226 path周囲のtruth-nearest
vertical shiftを一貫して順位付けできる。

Pearson / ZNCCのようにoffsetとscaleを完全に捨てず、affine uncertainty、
residual scale、系列相関、log determinantをproper Gaussian predictive densityへ残す。

## 固定した変更

exp280の13 shift、非重複512行block、exp226 OOF path、fold、Type Well policy、
known-prefix sigma clip、tie order、truth-late境界を固定する。

同一raw-finite supportで次の2×2要因分解だけを評価する。

| variant | affine | residual covariance | 役割 |
| --- | --- | --- | --- |
| `identity_iid_matched` | identity | iid | matched control |
| `affine_iid` | prefix posterior | iid | affine単独 |
| `identity_ar1` | identity | fold-safe AR1 | AR1単独 |
| `affine_ar1` | prefix posterior | fold-safe AR1 | primary |

保存exp280 raw-Gaussian scoreもstrong referenceとして別に比較する。

## 検証方針

- validation: exp226 group-safe 5-fold reporting strata
- unit: unknown suffixの非重複512-row block
- primary metrics: truth-nearest shiftのMRR / top3
- stress: `MD since 1000+`、hidden-like spatial、hidden-like typewell-purged、
  top1-regret p90
- leakage:
  affine posterior、fold rho、eligibility、4 score、negative control、manifest、
  content SHAをfreezeするまでsuffix truth / error / formation / roleを読まない
- negative control:
  immutable block key由来のstable shift-label permutation

Stage 0は0-HMM / 0-PF / 0-Beam / 0-model / 0-boosterである。
全technical / scientific gateをPASSしても、decoderは別実験で設計する。

## 実行入口

- 正規train Notebookへ採用した実装:
  `exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_train.py`
  をJupytext sourceとし、同内容を正規`*_train.ipynb`へ反映する。
- inference実装候補:
  `exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_inference.py`
  / 同名`.ipynb`。predictionを生成しないfail-closed実装。
- 正規`*_inference.ipynb`はplaceholderのまま保持する。
- 正規train Notebookを採用し、Kaggle private CPU version 2でStage 0を完了した。
- terminal FAIL後はpackage / push / rerunを再び無効化した。正規inference
  Notebook、prediction、submissionも無効のまま。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 | `stage_0_failed_close_without_rescue` |
| eligible blocks | `5,615 / 7,787 = 0.721074`（gate `>=0.75` FAIL） |
| `affine_ar1` MRR / top3 | `0.386090 / 0.439181` |
| matched MRR / top3 | `0.388003 / 0.450401` |
| saved exp280 MRR / top3 | `0.388146 / 0.449866` |
| runtime / peak RSS | `4,358.768411秒 / 1.264053 GB` |
| CV | なし |
| Public LB | なし |
| Private LB | なし |

## 所見

- `affine_ar1`はshuffleを5/5 foldsで上回り信号自体はあるが、matched / saved
  controlに対するMRR改善は2/5 folds、top3改善は1/5 foldsに留まった。
- affine単独比MRR差は`+0.000486`で固定`+0.005`に届かず、AR1単独比は
  `-0.000387`。複合効果の追加価値は確認できない。
- long-tailではMRR / top3が両controlを下回り、hidden-like 2面もMRRは上回ったが
  top3を下回った。primary top1-regret p90 `39.852949`はsaved
  `38.499431`より悪い。
- technical gateもraw-finite block coverageだけがFAILした。eligible well率、
  finite score、row alignment、truth-late、rho fold safety、Woodbury parity、
  runtime / RSSはPASSした。

## リスク / 注意

- exp343ではper-well ACF scheduleが不安定だったため、rhoはouter-train fold共通にする。
- exp345のaffine HMMは平均改善とworst-well悪化を併発したため、順位平均だけで昇格しない。
- exp359のheuristic window scoreとexp360のZNCCは閉鎖済みで、救済しない。
- Student-t / Huber row emissionは再利用せず、Gaussian covariance familyだけを評価する。
- blockごとのaffine再fitは禁止する。known prefix posteriorを全candidateへ固定する。

## 次

exp427内でsupport、prior、rho、block、shift、score family、gateを変更せず閉じる。
条件付き後続exp431もprerequisite FAILで閉じる。次候補は低優先度P4の
saved-artifact-only失敗原因分解に限定し、prediction / HMM / PF / modelへ進めない。
