# 要件

## 依頼

最終提出候補として、exp413の保存済みLightGBMを固定アンカーにし、
同じStage D入力面を使うCatBoost / XGBoostと、固定済み物理候補を
cross-fit bounded stackingするパイプラインを設計する。

2026-07-30の追加依頼により、今回の承認範囲へStage 0--5のtrain-side
Jupytext実装候補、変換Notebook、unit test、static validationを加える。
既存の正規Notebookは上書きせず、Kaggle package、学習実行、hidden inference、
提出は行わない。

2026-07-31のユーザー実行依頼により、承認範囲を正規train Notebook採用、
Kaggle train package / push / Stage 0--5実行まで拡張する。
実行量は2 variants / 2 configs / 5 folds / 10 GPU models、
control・selector・PF/HMM/Beam再学習0のまま変更しない。
hidden inference実装・実行と提出は今回の承認範囲に含めない。

train version 1は同名候補の元exp263値とexp413 overlay値の不一致をStage 0で
検出して学習0本で停止した。2026-07-31のユーザー確認により、親exp413との
train/inference parityを優先し、exp413 scale5-overlay版の実測OOF
`8.070218793924594`へ契約を一意化してversion 2を実行する。

version 2は明示Python例外なしのDeadKernelErrorとなり、fold完了logと再利用可能
outputは0だった。version 3では設計・モデル数・parameterを変えず、full matrix
SHAのzero-copy化、CatBoost Pool後の生行列解放、CatBoost/XGBoost行列の直列化だけを
行う。同じcanonical kernelの技術再実行とし、hidden inference / submissionは含めない。

version 3はStage 0 matrix preflight 5/5完了後、family train開始前に
DeadKernelErrorとなった。version 4は同じ科学contractのまま、allocator trim、
250,000-row chunk物理OOF Parquet、列先行fold matrix assembly、chunk finite検証を
追加する。同じcanonical kernelの技術再実行とし、control再学習は含めない。

version 4はStage 0を完了し、outer fold 0のCatBoost Pool生成後、fit開始時に
DeadKernelErrorとなった。version 5はclean273の273特徴を一時float32 NPY
memmapへ退避してDataFrameを学習前に解放し、CatBoost train / valid Poolを
raw matrix直列解放で構築する。fold別matrix content SHAの完全一致を必須とし、
科学contract、parameter、model数、control再学習0を変更しない。

## 制約

- Routeは`ensemble`。ML familyと物理候補の両方を最終予測候補として評価する。
- 親は`exp413_scale5_likpf_full_replacement_on_exp335`とする。
- exp413の3,783,989行、773 wells、outer 5 folds、final 370特徴、
  `last_known_tvt`残差target、Stage C/S出力、Stage D LightGBM OOF / 15 modelsを固定する。
- exp413のLightGBM、40 selector、20 signed selectorは再学習しない。
- 新規学習はCatBoost 1 config x 5 foldsとXGBoost 1 config x 5 foldsの
  2 variants / 2 configs / 10 GPU modelsだけとする。
- CatBoostはexp274で監査済みのPixiux `cb0`、XGBoostはexp275で監査済みの
  Cdeotte version 3設定を変更せず使用する。parameter gridは行わない。
- 物理候補は`exp226_w500_50_50`だけに固定する。実体は
  `0.50 * exp226_k16 + 0.25 * likpf_mean + 0.25 * exact_hmm`である。
- `likpf_mean` semantic slotは親exp413と同じ
  `likpf_scale_5_x1p0` overlayを使う。overlay前のexp263同名候補
  OOF `8.238331` / Public LB `7.800`はscale5版の根拠へ転用しない。
- Public LB 7.678の`exp226_k16__exact_hmm`を含む他の物理候補を
  同一OOF上で選択しない。
- stackingは非負・和1・interceptなしの固定boundで行い、
  leave-one-outer-fold-outのOOF-level cross-fit readoutを使う。
- stackingは追加boosterを学習しない。solverの重みだけを保存する。
- confidence gateは定数bounded stackが全採用条件をPASSした場合だけ評価する。
- hidden inferenceはsample submission由来のID、行数、well数を正とし、
  公開14,151行 / 3 wellsをassertしない。
- train / inferenceともKaggle Notebookを正とし、9時間制限に対して
  27,000秒のsoft budgetを置く。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- Stage 0からStage 6までの入力、出力、停止条件、次段条件が固定されている。
- `frozen_input_contract.yaml`にexp413のfold、Stage C/S、Stage D OOF、
  model manifest、物理候補のIDと既知SHAを記録している。
- `ensemble_contract.yaml`に2 model configs、10 models、stack bounds、
  solver、採用gate、conditional confidence gate、hidden runtime契約を記録している。
- Stage 0でfinal 370列の名前・順序、fold別float32 matrix content SHA、
  row key、target、anchorを保存する設計になっている。
- CatBoost / XGBoostは同じfold別matrixと同じ残差targetを読み、
  family以外の変数を変更しない。
- family別にpooled / fold / distance / hidden-like / by-well、
  prediction correlation、residual correlation、error covarianceを保存する。
- 物理候補は`exp226_w500_50_50` 1本だけで、PF/HMM/Beam再学習・再選択がない。
- stackingのboundsはLGB >= 0.60、Cat <= 0.25、XGB <= 0.20、
  Physics <= 0.20で固定されている。
- 定数stackの採用条件はexp413比pooled RMSE -0.03 ft以上、
  4/5 folds非悪化、固定scope悪化各+0.02 ft以内、
  by-well p95非悪化、worst well悪化+0.25 ft以内である。
- confidence gateはtarget-free disagreementだけを使う決定的な1候補で、
  定数stackからの行別変更を絶対値0.25 ft以内に制限する。
- train-side実行コードは`*_compact_selfcontained_train.py`と同名`.ipynb`に置き、
  採用承認済みの正規train notebookへ同期する。正規inference notebookは
  train gate PASSと別承認までplaceholderのまま維持する。
- Kaggle train run承認と全Stage 0--5 run flagが揃うまで、大きな入力を読む前に
  fail closedする。
- hidden inference実装はtrain gate PASSと別承認後に同じexp494へ追加する。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`に
  train-implementation-complete / not-runとして記録されている。
- deterministic anchorと呼ぶのは独立rerun一致後だけとし、feature content SHA、
  model SHA、prediction SHA、submission SHA、Kaggle kernel versionを記録する。
- gzip生成物はraw gzip SHAではなくdecompressed content SHAを主証拠にする。

## 2026-07-31 参考提出override

ユーザーの「今の実装で提出まで進みたい」を、scientific gate FAILを撤回せず、
version 5のconstant bounded stackをそのままhidden inference・参考提出する
明示overrideとして扱う。元の採用判断はexp413維持のままであり、exp494を
train-side anchorへ昇格させない。

- deployment weightsはLGB `0.681702678534061`、CatBoost
  `0.10372958993775055`、XGBoost `0.01456773152818835`、Physics `0.2`。
- conditional gate、well-level routing、trajectory後処理を追加しない。
- weight / candidate / parameter / bound / thresholdを再推定しない。
- 保存済み40 + 20 selector、15 LGB、5 Cat、5 XGBだけをloadし、学習0本とする。
- sample submissionの行数・ID・非空well集合をdynamic contractにする。
- root `submission.csv`をsubmit-checkしてから外部参考提出する。
- Public LBは参考結果として記録し、scientific FAILと分離する。
- 参考提出ref `55134873`はPublic LB `7.228`でCOMPLETEした。exp413
  `7.201`比`+0.027`悪化のためexp494は不採用とし、scientific / overall
  anchorはexp413を維持する。route別LB referenceだけをexp494へ更新する。
