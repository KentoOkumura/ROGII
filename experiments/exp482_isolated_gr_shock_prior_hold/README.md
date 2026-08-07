# exp482_isolated_gr_shock_prior_hold

## 状態

- ルート: `pf_beam`
- 状態: Stage A0 eligibility FAIL・terminal close
- 優先度: 低・P3
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-30
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

過去側predictive messageと、現在観測を除いたfuture messageが同じTVT近傍を
支持しており、現在のraw GRだけが前後から孤立している場合に限れば、
current emissionを使わず前の予測を維持する方が安全である。

## exp440との違い

exp440はposteriorの曖昧さだけで広く発火し、predictive priorがすでにwrong basinに
ある行でもholdを後続状態へ伝播させた。本実験は次の3条件をすべて要求する。

1. raw GR現在点が`±5`行の前後から孤立したrobust z `>=4.5`の単発shock。
2. predictive meanとcurrent-observation leave-one-out meanが`1.05 ft`以内。
3. current emissionがpredictive meanを`1.05 ft`以上動かす。

trigger行だけparent outputをleave-one-out meanへ置換し、親HMM stateと
次行以降の予測は変更しない。exp440のambiguity flagやactive rowは使わない。

## 検証方針

- Stage A0: 全773 wellsのraw GRだけでshock censusを作り、
  shock-support 32 + zero-shock matched control 32のfixed64をtarget-freeに凍結。
- Stage A1: unchanged exp209 message replay 64 wells、candidate state変更run 0。
- Stage 1: Stage A0/A1全PASSと別承認時だけfull 773-well OOF。
- Group: `well_id`、reporting folds 5。
- truth/fold/role/errorはmanifest、message、trigger、predictionのSHA freeze後だけ結合。
- candidate 1本、LightGBM/model/booster/PF/Beam/GPUは全段階0。
- Stage 0はshock-enriched mechanism sampleであり、CVではない。

## 実装状態

- `config.yaml`とsteeringで科学設計を固定済み。
- compact self-contained train実装を正規train Notebookへ採用済み。
- fail-closed inference候補と専用testを実装済み。正規inference Notebookは
  generic scaffoldのままで、inferenceは無効。
- Kaggle private CPU version 1でStage A0 raw-only censusを実行済み。
- zero-shock controlが`10 < 32 wells`でeligibility FAILとなり、
  Stage A1前にfail-closedした。
- Stage 1、inference、submissionは未実行。
- 専用pytest`14 passed`、Jupytext、構文、Ruff F821/E9、
  strict experiment validationはPASS。

## 実行入口

- 学習 notebook: `exp482_isolated_gr_shock_prior_hold_train.ipynb`
- 推論 notebook: `exp482_isolated_gr_shock_prior_hold_inference.ipynb`
- trainはKaggle private CPU version 1（id_no `129168015`）で完了した。
  raw census 773 wells、isolated shock 17,047 rows、support 763 wells、
  zero-shock control 10 wells。HMM replay、candidate prediction、truth joinは0。
- ローカルNotebook実行は行っていない。

## 再現性

- RNGなし。well、row、state、message、manifest matching順を固定する。
- raw census、fixed64、message、trigger、prediction、metricsのSHAを保存する。
- gzipはdecompressed content SHAを主証拠にする。
- 初回成功runをdeterministic anchorにしない。

## リスク

- current emissionの大きな即時反転は既存監査で非常に少なく、support不足になり得る。
- 薄い実地層をsensor shockと誤認する可能性がある。
- past/futureが同じwrong basinで一致している可能性は残る。
- full OOF PASSまで現行ML Public-LB基準exp413 `7.201`やroute anchorを更新しない。

## 所見

設計を確定しただけで、改善を示す実行結果はない。exp440の失敗を
threshold変更で救済するのではなく、独立したraw観測品質仮説と
row-local readoutに問題を限定した。Late phaseのため、まずraw-only censusと
fixed64 mechanism gateで低コストに反証する。

## 次

事前固定したfixed64 control設計が成立しなかったためexp482を閉じる。
threshold/window/control定義の救済、再run、Stage A1、full Stage 1、
inference、submissionへ進まない。
