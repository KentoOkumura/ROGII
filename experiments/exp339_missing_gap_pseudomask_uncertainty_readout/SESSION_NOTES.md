# exp339 セッションノート

## 2026-07-22 設計確定

- 目的: 補間値を実観測と同じ確信度で扱う問題に対し、まず補間誤差だけをpseudo-gapで測る。
- 実行規模: scientific readout 1、control 2、HMM well run 0、booster 0。
- 再現性: RNGを使わず、fold・well・gap長・開始位置のstable SHAでpseudo-gapを選ぶ。
- 依存: foldはexp226、補間処理はexp209に固定する。
- leakage guard: outer-valid誤差による表推定と、未知suffix TVTの参照を禁止する。
- long-tail guard: coverage、fold別校正、長さとの単調性、circular placement controlをすべて通す。

## 未実施

- notebook実行、Kaggle package/push、成果物生成は行っていない。

## 2026-07-22 Stage 0 実装

- ユーザーの `exp339を実装してください` を実装承認として、0-HMM Stage 0だけを実装した。
- trainはJupytext percent形式のcompact self-contained候補を起点に、正規train Notebookへ採用した。親exp209にcompact self-contained版は存在しないため、同じ0-HMM診断のexp337（9章・1,104行）と比較し、exp339は10章・約1,490行でpreflight、pseudo-gap生成、table fit、gate、生成物保存までNotebook上で追える構成にした。
- outer foldごとにouter-trainのknown-prefix自然欠損run histogramを作り、5 length bin内のexact lengthをstable SHA CDFで最大4件/well/binへ割り当てる。
- real gapは両側finite anchor、hidden row非再利用、anchor保護を必須にした。circular controlはgap identity・長さ・件数を保持し、同一wellのeligible start列をstable non-zero offsetで循環移動する。
- hidden raw GRを含まないinterpolation predictionのcontent SHAを先に固定し、その後だけhidden GRをone-to-one joinして誤差を計算する。
- outer-train誤差だけで `cell -> length -> global` のsupport 200縮約表をfitし、outer-validのGaussian NLL、variance/MSE比、length-sigma Spearman、circular control差をAND gate評価する。
- 生成予定はfold manifest、natural missing inventory/histogram、gap plan、interpolation prediction、uncertainty table、late-join audit、fold summary、scientific contract、summary。gzipはdecompressed content SHAを主証拠にする。
- 実行規模はscientific readout 1、control 2、HMM/model/config/fold booster各0、親control再実行0。
- fail-closed inference候補はHMM integration、raw-test inference、submissionを明示停止する。
- 実装時点では`execution.kaggle_push_approved=false`、`run_stage_0=false`として、Notebookを誤実行してもStage 0開始前に停止する状態で固定した。

## 静的検証

- `tests/test_exp339_missing_gap_pseudomask_uncertainty_readout.py`: 8 tests PASS。
- `py_compile`: train/inference PASS。
- `ruff`: train/inference/test PASS。
- Jupytext変換と`--test`: train/inference PASS。
- `task` executableは環境になかったため、規定のfallback `make validate-exp EXP=exp339_missing_gap_pseudomask_uncertainty_readout`を実行しstrict PASS。
- repo全体pytestは`577 passed / 3 skipped / 2 failed`。2 failureはいずれも既存exp296の完了済みstatus/run flagと古いtest期待値の不一致で、exp339固有testは全件PASSした。
- 静的検証時点ではKaggle実行は別承認待ちだった。

## 2026-07-22 Kaggle CPU実行承認・push前確認

- ユーザーの`実行してください`を、exp339 Stage 0のcanonical Kaggle CPU package / push / run承認として記録した。
- 実行するvariantはscientific readout 1件と、global constant variance / matched circular placementのcontrol 2件。
- outer GroupKFoldは5 foldだが、各foldはtable fit/readoutの分離単位であり学習済みモデルは作らない。
- LightGBM config 0、model config 0、trained fold 0、合計booster 0、HMM well run 0、親control再学習0。
- GPU off、internet off。inferenceとsubmissionは引き続き無効で、exp341も全Stage 0 gate通過まではblockedとする。
- 初回canonical候補`kentookumura/exp339-missing-gap-pseudomask-uncertainty-readout-train` / `exp339 missing gap pseudomask uncertainty readout train`は、id/title slug自体は一致していたが、Kaggle `SaveKernel`が詳細なしの400を返してversionを作成しなかった。
- 元slugは`kernels list --mine --search`でnot found、`kernels pull -m`でも取得できず403だった。Kaggle上の既存kernel群ではslug/title本体が最大50文字なのに対し、初回titleは55文字だったため長さ制約が原因と判断した。
- 実験番号と主要suffixを保持し、重複的な末尾語`readout`だけを省いた47文字のcanonical id/title、`kentookumura/exp339-missing-gap-pseudomask-uncertainty-train` / `exp339 missing gap pseudomask uncertainty train`へ揃えて再packageする。別実験番号は作らない。
- 短縮canonical packageはKaggle version 1としてpush成功。URLは`https://www.kaggle.com/code/kentookumura/exp339-missing-gap-pseudomask-uncertainty-train`、Kaggle `id_no=128226213`。
- push後に同一slugを`kernels pull -m`し、private、GPU off、TPU off、internet off、competition source、exp226 kernel sourceがpackageと一致することを確認した。

## 2026-07-22 Kaggle CPU Stage 0 version 1結果

- kernel `kentookumura/exp339-missing-gap-pseudomask-uncertainty-train` version 1、id_no `128226213`を完了。readout runtimeは`320.79614 sec`。
- 116,458 outer-valid pseudo-gap rows / 773 distinct wells。coverageは全fold 1.0、各fold154--155 wells。
- primary/global/circular pooled NLLは`4.0413109881 / 4.0758737582 / 4.0445842809`。primaryはglobalに5/5 foldsで勝った。
- variance/MSE比はpooled `0.9755816370`かつ5/5 folds校正範囲内。length-sigma Spearmanはpooled `0.5184073076`かつ5/5 folds正。
- real placementはpooledでcircularより良かったが、fold別では0・1だけの2/5勝利で固定4/5 gateをFAIL。他の10 checksはPASSした。
- statusは`stage0_gate_failed_exp341_blocked`。事前規約に従い、bin/support/pseudo-gap数/補間法の救済、再実行、HMM、inference、submissionへ進まない。
- table content SHAは`6a9bd955a64ab60fc442e6675c2676828f5426a74fdae3e6fbbd68e34d0eb4e5`だが、gate FAILのためexp341入力として承認しない。
- logsとKaggle file listを確認し、記録に必要なmetrics、fold summary、summary、scientific contractだけを対象取得した。大きいoutput archiveは取得していない。
