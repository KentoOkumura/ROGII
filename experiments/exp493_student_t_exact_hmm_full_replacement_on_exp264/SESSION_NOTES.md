# exp493 セッションノート

## 目的

exp374 Student-t exact HMMをexp264の元の`exact_hmm`と全面置換し、
総数12候補のままcorrected strict nested dual selectorを評価する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage A/C完了、scientific gate FAIL、branch close
- CV: `8.616237400142841`
- LB: なし
- 実行コード:
  `exp493_student_t_exact_hmm_full_replacement_on_exp264_compact_selfcontained_train.py`
- executable notebook: compact候補1、canonical train 1
- canonical train: compact候補を採用
- inference notebook: markdown-only placeholderを維持
- Kaggle package: 作成済み
- Kaggle push: version 1--3成功
- Kaggle run: version 3 COMPLETE

## 凍結した実行量

- active variant: 1
- LightGBM objectives: 2
- outer folds: 5
- inner folds: 4
- planned CPU selector boosters: 40
- trained boosters: version 3で40、version 2との累計80
- 親/control再学習: 0
- GPU boosters: 0
- downstream TVT / inference / submission: 0 / 0 / 0

2026-07-31のユーザー依頼`実行してください`を、canonical train採用と
Kaggle package/push/runの承認として記録した。保存exp264 controlをbaselineとして
参照し、control variantは再学習対象に含めない。

## 変更契約

- 12 candidate ID/order/domainを維持する。
- changed 4:
  `exact_hmm`、`exp226_k16__exact_hmm`、`likpf_mean__exact_hmm`、
  `exp226_w500_50_50`
- unchanged 8はparent値と完全parityを要求する。
- exp264 corrected Stage A 88列schemaを同じ名前・順序で使う。
- exp374をglobal key join後にexp263 selector foldへ再分割する。

## 設計根拠

- exp374: `11.938287235 -> 11.720478702`、gain `0.217808533 ft`、4/5 folds。
- tail: by-well p95 `+0.982661344 ft`、worst `+35.015963236 ft`で不合格。
- exp388 fixed13: `8.652531956 -> 8.736104109`、`+0.083572154 ft`、
  2/5 folds。Student-t top1 692,647 rows / 18.304678%でもtailが悪化した。

## 再現性メモ

- seed: 42
- sampling: stable SHA256 keys
- HMM/PF/Beam再生成: 0
- exp374 decompressed SHA:
  `668fe87da902955acee742c72d30724abb53f32050bb5d0a5c1b3dee0cbd626e`
- parent feature schema logical SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- model manifest SHA:
  `2560711a4a2f333a31cc2b3b423ace9d7b3d31062c0513e0505376e4ff80d0ca`
- outer-valid score SHA:
  `da4f6af616c3a87495d643cca3ef6e3e531cb2828dca769ef32fe7f4c6f9db89`
- deterministic anchor: false。独立再実行一致前は昇格しない。

## version 3結果

- status: `COMPLETE`
- observed complete: `2026-07-31 05:28:52 UTC`
- notebook elapsed: `5896.184329532 sec`
- trained selector: `40/40`
- hard primary: `8.652531955610227 -> 8.616237400142841`
  （`-0.03629455546738569 ft`）
- fold RMSE:
  `8.827610034 / 8.547425036 / 8.698265159 / 8.566946237 / 8.435310875`
- parent比fold delta:
  `-0.164109991 / +0.120790111 / -0.202238182 / +0.140473234 / -0.064247797`
- 改善fold: `3/5`。固定条件`4/5`をFAIL。
- near 0--250: `-0.003840791 ft`
- distance 1000+: `-0.039663273 ft`
- hidden-like spatial / typewell-purged:
  `-0.104928471 / -0.067629142 ft`
- by-well p95: `+0.540095855 ft`。上限`+0.25 ft`をFAIL。
- worst: `f6d009f4 +10.472288433 ft`。上限`+0.25 ft`をFAIL。
- Student-t依存4候補top1:
  `1,372,891 / 3,783,989 = 36.281580%`
- fixed fallback report-only:
  `8.238331546 -> 8.160447731`（`-0.077883816 ft`）
- technical、leakage、selector score guard: PASS
- selector score 3指標: prior比5/5 folds改善
- decision:
  `FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR`

候補数を12に戻すとexp388 fixed13のpooled悪化は解消して平均改善になったが、
fold一貫性とwell-tailは改善しなかった。candidate count増加だけが失敗原因ではなく、
Student-t置換を強く利用するhard selectorのwell単位安全性が不足している。
same-OOF weight / threshold / domain / gate救済、downstream、inference、
submissionへ進まない。

## 実装内容

- `src/exact_hmm_full_replacement.py`をexp492 Huber / exp493 Student-tで
  共用できるよう拡張した。
- exp374はallowlist 6列だけを読み、raw gzip SHA、decompressed SHA、
  post-read content SHAを検証する。
- `(well_id,row_idx)`でglobal joinした後にexp263 outer foldへ再分割する。
- `exact_hmm`と依存formula 3本だけをfloat32式で再計算し、
  unchanged 8本の値・availability完全parityをfail-closedで検証する。
- exp264 corrected Stage A 88列 / compact 74列を自然生成できることを
  truth読込前にprobeし、列の追加・削除・refreezeを禁止する。
- Stage Cはstrict nested 2 objectives x outer 5 x inner 4の40 CPU boosterだけ。
- saved exp264 scoreとのprimary、fold、near、1000+、hidden-like、by-well
  scientific gateとfeature importance保存を実装した。
- PASS/FAIL後もdownstream TVT、inference、submissionへ進まない固定stopを入れた。

## sibling構成比較

exp492 compact候補は631行、exp493は633行。両方ともContentsと9章を持ち、
入力/SHA、Stage A、Stage C、科学readout、feature importance、再現性summaryを
同じ粒度で展開している。exp493側に`__file__`依存はない。

## コマンドログ

- 2026-07-30: `make new-steering`と`make new-exp`でscaffoldを作成した。
- 2026-07-31: ユーザーの`exp493を実装してください`を実装承認として記録した。
- 2026-07-31:
  `.venv/bin/pytest -q experiments/exp492_huber_exact_hmm_full_replacement_on_exp264/tests/test_exp492_huber_exact_hmm_full_replacement.py experiments/exp493_student_t_exact_hmm_full_replacement_on_exp264/tests/test_exp493_student_t_exact_hmm_full_replacement.py`
  は`17 passed`。
- 2026-07-31: exp374 upstream testも含む最終回帰確認は`26 passed`。
- 2026-07-31: `py_compile`、Jupytext `--to ipynb --test`、
  `PYTHONPATH=. EXP493_IMPORT_ONLY=1` import-only contract確認を通した。
- 2026-07-31: 実行直前契約を1 variant / 2 objectives / outer 5 /
  inner 4 / 40 CPU booster、control再学習0、GPU 0、downstream 0、
  inference 0、submission 0として再確認した。
- 2026-07-31: canonical notebook採用、package、runの承認を記録した。
- 2026-07-30 23:18:46 UTC: canonical kernel id
  `kentookumura/exp493-studentt-fixed12-replacement-selector-train`、
  title `exp493 studentt fixed12 replacement selector train`、
  private CPU / internet off / run-on-pushでpackageを作成した。
- package config SHA:
  `ab840a59fbeadd9a47c9df34ca7cb481f4cee596ad5f074fa00974ea06233792`
- package notebook SHA:
  `2d991fb00fdebf096df0a45921e8b46ccbaecc25b80d68d5fcf33de1dc346056`
- bootstrap ZIP内configとloose/local configのbyte parityを確認した。
- push前の同一kernel pullは403で、既存versionを確認できなかった。
- 2026-07-30 23:20:43 UTC: version 1をpushし、run-on-pushを開始した。
- push後pullで`id_no=129218034`、private、CPU、internet off、
  competition source 1 / kernel source 3を確認した。
- kernel:
  `kentookumura/exp493-studentt-fixed12-replacement-selector-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp493-studentt-fixed12-replacement-selector-train`
- version 1はelapsed `18.985 sec`で`exp264 parent config did not resolve`
  により停止した。Stage A前であり、学習boosterは0。科学的な試行には数えない。
- Kaggle側のexp264 kernelにはscore/schema/cache関連成果物が存在することを
  `kaggle kernels files`で確認した。source configだけはkernel inputから
  一意なpathで参照できなかったため、local SHA
  `55a878f80c5c316efdf3291a076b9e29054cc235a802b5ac7ca93b444ed9c185`
  の親configを`inputs/exp264_parent_config.yaml`としてbootstrapへ同梱する。
- version 2 push前に、bootstrap 33 files、親config 1件、親config SHA一致、
  loose/embedded config SHA一致を確認した。
- version 2 package notebook SHA:
  `6fdad48d5193ae5bb823fb7b827e7c3a8d460c862c4dc576c0066af68d2fe6d8`
- version 2 package config SHA:
  `6347f5c5165027793e583b0cef168e68fc6df987fa4652e3679c5328dac4e201`
- version 2も1 variant / 2 objectives / outer 5 / inner 4 /
  40 CPU booster、control再学習0、GPU/downstream/inference/submission 0。
- 2026-07-30 23:30:36 UTC: 同じcanonical kernelへversion 2をpushした。
  version 2はelapsed `5846.077 sec`でERROR。
- log上はouter 5 × inner 4 × 2 objectivesの全40 boosterを完了し、
  scientific readout cellも完了した後、feature-importance cellで
  `KeyError: Column not found: gain`となった。
- Stage Cのimportance schemaは
  `objective, feature, importance_type, importance`であるため、
  `importance_type == "gain"`を選び`importance`を平均するよう修正した。
- ERROR runのoutput file一覧は空で、pullしたsource notebookにもcell outputが
  ないため、CV / gate / SHAは回収不能。version 3で追加40 CPU boosterを
  学習すると累計80となる。
- 2026-07-31: ユーザーの`再実行してください`をversion 3再実行承認として記録。
  追加40・累計80 CPU booster、1 variant、2 objectives、outer 5、
  inner 4、保存control再学習0、GPU/downstream/inference/submission 0を維持する。
- version 3 push前に同一canonical kernelをpullし、`id_no=129218034`、
  private、CPU、internet offを再確認した。
- version 3 packageはbootstrap 33 files、親config 1件、親config SHA
  `55a878f80c5c316efdf3291a076b9e29054cc235a802b5ac7ca93b444ed9c185`
  一致、loose/embedded config一致、v3承認guardとimportance schema修正あり。
- version 3 package notebook SHA:
  `40def465079a0fb0b94f4bb408d563446344496a4628f3f71ec240f0ee32034b`
- version 3 package config SHA:
  `bacdf7d712d479493eefde1300a57dc84e06d2000baeca8ce3cb4a301f5de25a`
- version 3 package helper SHA:
  `6cfd9b51dee10db48d59eb3d818328487af214a1c4a43ffb83a179bdcdbe885b`
- 2026-07-31 03:49:34 UTC: 同じcanonical kernelへversion 3をpushした。
  2026-07-31 05:28:52 UTCに`COMPLETE`を確認した。
- 通常logの`FINAL_SUMMARY`からCV、gate、SHA、runtimeを記録した。
- full output archiveは取得せず、次だけを
  `kaggle/output/train_v3_selected/`へ取得した:
  metrics、scientific gate、fold metrics、scope/by-well/usage、
  model/reproducibility/replacement manifests、feature importance。
- selected outputのscientific gate、summary、model manifest、
  reproducibility manifest SHAをlog値と照合した。
- local notebook実行は行っていない。

## 次のアクション

branchをfail-closedで終了する。既存P4の
`student_t_gaussian_disagreement_continuous_risk_feature_on_exp264`は、
別の必要性と承認がある場合だけ0-booster readoutとして検討する。
