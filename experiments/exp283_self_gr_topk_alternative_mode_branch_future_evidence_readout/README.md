# exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout

## 状態

- ルート: PF/Beam
- 状態: Kaggle CPU version 2完了、technical PASS / scientific FAILで不採用
- CV / LB / Submit ID: 対象外（診断readoutのみ、推論・提出なし）
- 作成日: 2026-07-19
- 親実験: `exp282_longtail_prediction_zone_self_gr_loop_closure_readout`

## 仮説

exp282の局所matchをTVT donorとして直接使うのではなく、曖昧eventでbaseを保持したまま代替modeを
top-3提案し、提案後の未来256行typewell尤度で検証すれば、弱いself-GR signalをbranch proposalとして
利用できる可能性がある。

## 固定設計

- target-free event: exp236二峰segment、exact-HMM / likPF / exp226の128行persistent disagreement、
  exp280互換low-margin block。
- proposal: causal trailing GR 17/31/51行、known prefix + 256行以上前のprediction zone、
  forward/reverse、global top-3。
- branch: donor anchorからexp226 geometry incrementで未来へ延長。exp263 fixed baseは常に保持。
- verifier: proposal後256行のexp209 raw-GR/typewell累積尤度。geometryはdiagnostic/vetoのみ。
- freeze: event、proposal、future evidenceのcontent SHA確定後にだけtruthを結合。
- 0 booster、HMM/PF再生成なし、inference/submissionなし。

## 検証方針

- Group: well id、5 folds。foldはouter-train q20とpost-freeze集計にだけ使う。
- proposal primary: top-3 within10 lift vs same-well shuffled donor。
- verifier primary: future 256-row typewell score-margin AUCとselected branch RMSE。
- Leakage check: event/proposal/evidenceをtruthなしでfreezeし、content SHA確定後だけtrue TVTを結合する。

## 成功条件

- top-3 within10 lift vs shuffled `>= +0.02`、5/5 foldsで正。
- H256 branch-choice AUC `>= 0.60`を5/5 folds。
- selected H256 RMSEがbaseより`>= 0.10 ft`改善、5/5 folds非悪化。
- base unique-best時false switch率`<= 5%`。

正規notebookはユーザーの実行承認後にcompact sourceから採用した。Kaggle private CPU version 2は
1,331.408秒で完走し、3,783,989 rows / 773 wells / 4,397 eventsを処理した。

## 所見

technical guardは全PASS。top-3 within10は`0.755288`、shuffled `0.722083`、lift `+0.033204`で
5/5 folds正、branch-choice AUCもpooled `0.622168`かつ5/5 foldsで0.60以上だった。一方、selected
H256 RMSEはbase `8.221613`から`14.606586`へ悪化し、gain `-6.384973 ft`、nonregressing fold 0/5、
false-switch `55.5647%`。hidden-like 2面も約`-7.13～-7.17 ft`悪化した。proposalに弱いsignalはあるが、
固定future-evidence verifierをcommitへ使う安全性はない。

## 生成物

target-free event / proposal / future-evidence table、post-freeze event/branch readout、proposal/verifier/
fold/scope/by-well metrics、input manifest、summaryをKaggle outputへ保存した。fold・scope・by-wellとSHA
監査のため`/tmp/exp283-v2-output`へ一時取得し、metric CSV 8件と5 gzipのraw/decompressed SHAを照合した。
summary SHAは`8de2db0b...fe99`。

## 次

negative diagnosticとして閉じる。同一仮説のK/window/horizon/veto/margin/threshold救済、decoder接続、
inference、submissionへ進まない。exp284は別の明示overrideでstandalone実行されたが、exp283からの
scientific promotionは付与せず、triggered multibranch decoderの先行条件も満たさない。

## 参照

- steering: `docs/legacy/steering/20260719-exp283-self-gr-topk-alternative-mode-branch-future-evidence-readout/`
- 設定: `config.yaml`
- 実行記録: `SESSION_NOTES.md`
