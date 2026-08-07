# タスクリスト

## TODO

- なし。外部competition submitは別の明示承認が必要。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-08-01: backlog候補をexp497として切り出した。
- 2026-08-01: Public-LB特化処理の除外契約を固定した。
- 2026-08-01: outer 5 / inner 4、public-core内部weight、exp413 meta-fold blend、固定gateを設計した。
- 2026-08-01: 再現性、実行量、no-rescue、推論/提出停止条件を固定した。
- 2026-08-01: ユーザーの実装依頼をStage 0 compact preflight実装承認として記録した。
- 2026-08-01: 参照sourceをread-only pullし、Jupytext変換SHA `88c7b99e...5454`を確認した。
- 2026-08-01: source/symbol/decontamination監査、exp413 SHA契約、nested fold、spatial pool、truth-late freeze、meta5 weightの実装と専用testを追加した。
- 2026-08-01: SP45 195列、learned 205列、LikPF 2 bank、Beam 14+7本の静的inventoryを固定した。
- 2026-08-01: Stage P/M1/M2/Eと正規train runbookを実装した。
- 2026-08-01: 1 variant、LGB 120、Cat 80、total 200、Ridge 10、exp413再学習0でKaggle実行承認を得た。
- 2026-08-01: ユーザー指示によりColabを除外し、Stage MはKaggle GPUのみとした。
- 2026-08-01: Stage P単一kernel version 1は12時間上限で停止。fold別5 kernelへ分割した。
- 2026-08-03: Stage P outer0..4とStage M outer0..4を完了し、入力・予測・model manifest・成果物SHAを検証した。
- 2026-08-03: Stage E version 1をKaggle CPUで完了。candidate CV 7.87448814999802、exp413比0.010314644 ft改善だった。
- 2026-08-03: primary AND gateはfold 0/4、固定scope、by-well tailを含む5条件でFAIL。exp413を選択してinference/submissionなしでterminal closeした。
- 2026-08-03: ユーザーがgate FAIL後のexp497 prediction-only current-test inferenceを明示override。exp413再推論0、外部submit 0を固定した。
- 2026-08-04: Stage I version 3を完了し、14,151行のprediction契約をPASSした。40 booster本体が未保存であることを監査で確認した。
- 2026-08-04: ユーザーが40 boosterのmodel serialization追加と同一Kaggle GPU再実行を明示承認した。
- 2026-08-04: LightGBM txt 24、CatBoost cbm 16、Ridge JSON 2、保存直後reload parity、SHA/bytes/一意path契約を実装し、focused tests 25件をPASSした。
- 2026-08-04: Stage I version 4をKaggle GPUで完了。40 booster / 335,918,672 bytes、Ridge 2、model-set SHA、reload parity最大差0.0、14,151行のprediction契約を検証した。exp413再学習・再推論、submission、Colabは0。
- 2026-08-04: ユーザーが保存済みmodelを読むhidden-safe推論専用候補の作成を承認。新規学習0、dynamic exp413、固定blend、Kaggle outputのsubmission.csv生成、外部submit 0を固定した。
- 2026-08-04: 343行/7章のcompact inference候補とsaved-model coreを実装。Jupytext、構文、Ruff、29 tests、実v4 artifact 40件/335,918,672 bytesの読込契約をPASSした。Kaggle実行は未実施。
- 2026-08-04: 候補を正規inference Notebookへ採用し、exp413 hidden-safe依存を含む76 support filesをpackage化。全SHA readback、T4、internet off、13 kernel sourcesを検証した。
- 2026-08-04: `kentookumura/exp497-strict-public-core-saved-inference` version 1（id_no `129666751`）をpush。fit 0、外部submit 0、初期status `QUEUED`を1回確認して監視を停止した。
- 2026-08-04: version 1は約661.6秒で`ERROR`。exp413動的生成、strict特徴生成、保存モデル推論後にvisible parityがstrict `0.001281` / blend `0.014195`で共通tolerance `0.001`を超えた。OOM、入力欠落、model SHA不一致、学習、外部submitは0。
- 2026-08-04: ユーザーがversion 2再実行を承認。strict `0.002` / blend `0.020 ft`のcomponent別guard、中間exp413 submissionのID/order/finite/parity検証と別名隔離を実装し、focused tests `30 passed`。
- 2026-08-04: version 2 package 76 filesのSHA readback、remote T4 / internet off / 13 kernel sources、新しいcore/threshold/isolation markerを検証してpush成功。
- 2026-08-04: version 2はKaggle T4で`COMPLETE`。exp497 40 + Ridge 2、exp413 75をload、fit 0、strict/blend parity `0.001281/0.014195 <= 0.002/0.020 ft`をPASSした。
- 2026-08-04: final `submission.csv`は14,151行、sample ID順一致、重複・欠損・非有限値0、serialized blend差0.0、SHA一致、submit-check FAIL/WARN 0。外部competition submitは0。
