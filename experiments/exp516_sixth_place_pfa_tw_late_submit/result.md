# exp516 6位 `pfA × twGR` LATE SUBMIT再現監査 結果

## 状態

PF単体の忠実再現実装、契約テスト、Jupytext pairing、strict experiment validation、Kaggle package生成まで完了した。Active Sessions確認gateはユーザー指示により廃止した。Kaggle full runとlate submissionは未実行で、数値結果はまだない。

`2026-08-07 13:17 UTC`時点のGPU週次残量は`1.07h`で、15 GRU再学習とfull smootherの完走時間を保証できないため、`2026-08-08 00:00 UTC`のquota refreshまでpushを延期した。

ユーザーの明示承認後にKaggle version 1をpushしたが、CRLF raw-file SHAとLF embedded-text SHAを混同したidentity guardが予測開始前にfail-closeした。数値設定を変えず2種類のSHAを分離し、version 2候補は契約テスト`7 passed`と全静的検証を通過した。version 1はPF予測未実行のため科学的negative resultではない。

静的検証はcontract test `6 passed`、生成source SHA `7aca22b...ccd2a6`。Notebookとsubmission messageには`LATE SUBMIT`を明示した。

## 評価契約

公開sourceの単体`pfA × twGR`を固定再現し、technical gate通過後に1回だけlate submitする。作者報告の単体component CV 7.8 / Public 7.88 / Private 7.78は外部参照値であり、exp516の実績ではない。

6位最終systemのCV 5.4577 / Public 5.626 / Private 5.984は、91候補、candidate-curve NN、TCN、GBM、de-shrinkを含む別契約であり、この実験は再現を主張しない。

この提出はコンペ終了後の`LATE SUBMIT`であり、正式順位や競技中のmodel selectionとは分けて記録する。

## Negative result scope

未評価。失敗した場合も`(GR/typewell + GR-free anchor + learned emission, particle TVT path, standalone direct decode, no fusion, late hidden test, 600×32 T4x2)`の範囲だけを閉じ、6位の91候補family全体を閉じない。
