# exp284_prefix_masked_wrong_mode_branch_recovery_backtest

## 状態

- ルート: PF/Beam
- 状態: Kaggle CPU version 2完了、scientific guard FAIL、branch closed
- CV / LB / Submit ID: controlled backtest / 対象外 / なし
- 作成日: 2026-07-19
- 親実験: `exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`

## 仮説

known prefix末尾を隠してGR-supported wrong modeを注入し、safe baseを残したmultiple-hypothesis状態に
すれば、post-event累積evidenceがwrong branchを棄却できるかをoracle triggerなしで検証できる。

## 固定設計

- last known prefixの末尾640行をmask、cut以前512行以上を要求。
- cut後128行wrong-modeを保持し、event後256行をprimary verifier、128/512行をdiagnosticにする。
- cut直前128行、exp280 fixed shift bank、`|shift|>=10 ft`からwrong modeを決定する。
- safe base、wrong active、exp283 self-GR top-3を比較し、safe baseは常時保持する。
- real/shuffled self-GR、wrong-only、safe+wrong、full、no-injectionの5 policiesを固定比較する。
- post-cut true TVTは全branch/evidence freeze後だけ評価に使う。

## 検証方針

- Group: well id、5 folds。1 well 1 fixed pseudo cut。
- primary: H256 safe-vs-wrong AUC、wrong-only / pair-onlyに対するfull branch RMSE gain。
- safety: no-injection false switch、H512 persistence、real-vs-shuffled self-GR。
- Leakage check: cut後TVT_inputをloader直後にmaskし、全target-free tableのfreeze後だけtruthを結合する。

## 成功条件

- eligible 100 wells以上、5 folds、finite/identity coverage 1.0。
- safe-vs-wrong H256 AUC 0.60以上を5/5 folds。
- full H256 RMSEがwrong-onlyより0.10ft以上、pair-onlyより0.02ft以上改善。
- no-injection false switch 5%以下、real self-GRがshuffledより安定。

全条件PASSだけがtriggered fixed-horizon decoder backlogの実装検討を許可する。compact self-contained
trainはユーザー承認後に正規notebookへ採用した。

## 所見

Kaggle version 2は766 eligible / 7 ineligible wells、5 foldsを11,717.244秒で完走した。全technical
guardはPASSしたが、scientific / safety guardはFAILした。

- pair AUC: pooled `0.675153`だがfold 3 / 4は`0.509459 / 0.555936`でguard未達
- pair choice accuracy: `0.590078 < 0.60`
- H256 full RMSE: `26.072230`
- wrong-only比gain: `+11.484854 ft`、5/5 folds改善
- safe+wrong pair比gain: `-2.438300 ft`、改善0/5 folds
- shuffled self-GR RMSE: `25.520057`でreal fullより良い
- no-injection false switch: `30.1724% > 5%`

safe base保持によるwrong-onlyからの回復は確認できたが、self-GR top-3はsafe+wrong pairへincremental valueを
追加できず、安全性も満たさない。判定は`close_without_parameter_rescue`。

実装はexp226保存済みfold/kappaとother-fold donor geometry fieldからpseudo cutのgeometry増分を再生する。
held-out target readerは`TVT`を読まず、wrong shiftはvisible `TVT_input`、proposalはeventでtruncateした
causal GRだけを使う。target-free 6 tableのcontent SHA固定後にだけheld-out truthをattachする。

## 生成物

mask manifest、wrong-mode injection、proposal、branch path、future-evidence、policy selection、
overall/fold/pairwise/by-well metrics、summaryをKaggle outputへ保存した。小規模metrics/manifestだけを
`/tmp/kaggle-output/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/train_v2_metrics`へ取得し、
summary記録SHAと全件一致した。

## 次

parameter rescue、decoder接続、current-test生成、inference、submissionへ進まずbranchを閉じる。

## 参照

- steering: `.steering/20260719-exp284-prefix-masked-wrong-mode-branch-recovery-backtest/`
- 設定: `config.yaml`
- 実行記録: `SESSION_NOTES.md`
