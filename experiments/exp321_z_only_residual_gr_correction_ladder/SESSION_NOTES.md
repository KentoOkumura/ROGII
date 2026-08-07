# exp321_z_only_residual_gr_correction_ladder セッションノート

## 目的

Z-only残差構造→GR shift識別力→exp226 window GR補正を1つの実験内の段階gateとして設計し、後続のexact HMM / selector候補追加を混ぜずに反証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Run AB version 1完了、Stage A PASS / Stage B FAIL、branch closed
- Stage A/B/C: A/B実装・実行済み / C未実装・gateで閉鎖
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- window decoder実行: 0。Stage C gate FAILにより773 well-runs候補は閉鎖
- parent/control再実行: 0
- CV / LB / submission: なし / なし / なし

## 2026-07-21 Kaggle CPU Run AB実行承認

ユーザーの追加依頼「実行してください」を、Stage A/BのみのKaggle CPU package / push / run承認として記録した。Stage C実装/run、inference、submissionへの承認には拡張しない。

- 実行量: active variant 1、diagnostic contract 1、fold strata 5。
- model config / trained fold / booster / HMM / window decoder: `0 / 0 / 0 / 0 / 0`。
- parent/control再実行: 0。保存exp226 OOFとexp115 assignmentだけを参照する。
- runtime: Kaggle CPU、GPU/TPU/internet off。
- canonical kernel id: `kentookumura/exp321-z-only-residual-gr-ladder-train`
- canonical title: `exp321 z only residual gr ladder train`
- directory全体由来slugは49文字で、exp305で確認済みの48文字制約を1文字超えるため、科学的意味を保った`gr-ladder`略記で38文字へ短縮した。id/title slugは一致させる。
- kernel sources: `kentookumura/exp226-k16-kappa-repro-train`、`kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`。
- credential preflight: Kaggle CLI 2.2.3、OAuth credentialとlegacy API keyは利用可能。API tokenは未設定だがCLI OAuthで操作可能。
- 先行exp305はpreflight時点では`KernelWorkerStatus.RUNNING`だったが、2026-07-21 21:06 JSTに`KernelWorkerStatus.COMPLETE`と終了ログを確認した。
- 2026-07-21 21:06 JSTにcanonical kernel `kentookumura/exp321-z-only-residual-gr-ladder-train`へversion 1を一度だけpushした。Kaggle URLは `https://www.kaggle.com/code/kentookumura/exp321-z-only-residual-gr-ladder-train`。重複pushせず同slugを監視する。

### Run AB package preflight

```bash
make prepare-kaggle-notebooks EXP=exp321_z_only_residual_gr_correction_ladder EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp321-z-only-residual-gr-ladder-train --title 'exp321 z only residual gr ladder train' --run-on-push --strict"
```

- strict experiment validation、専用10 tests、Jupytext、構文、ruff F821を再確認してPASS。
- package metadataはid/title slug一致、private、CPU、GPU/TPU/internet off、run-on-push true。
- competition source 1件、kernel source 2件を確認した。
- loose configとbootstrap内configの承認フラグ、1 variant、5 fold strata、0 model config / trained fold / booster、Stage C disabledを確認した。
- bootstrap 17ファイルのbyte/SHA mismatchは0。生成Notebookは23 cells。

## 2026-07-21 Kaggle CPU Run AB version 1結果

- canonical kernel `kentookumura/exp321-z-only-residual-gr-ladder-train`は`KernelWorkerStatus.COMPLETE`。
- runtime `611.963350 sec`、3,783,989 rows / 773 wells / 5 folds。
- 実行量は1 diagnostic contract、5 fold strata、model config / trained fold / booster / HMM / window decoder `0 / 0 / 0 / 0 / 0`、parent/control再実行0で事前契約どおり。
- Stage AはPASS。H512のZ-only / exp226 affine-quotient RMSE比`0.910543`、5/5 folds、affine SSE説明率`0.999968`、cap4 oracle gain`3.205124 ft`。
- Stage Bのtop1/top3/MRR/signは`0.332991 / 0.587903 / 0.503399 / 0.685887`。shuffleを全5 folds、1000+、hidden-like 2面で上回り、exp280 pooled保存値も4/4 strict改善した。
- 一方、固定`±80 ft` bank range coverageは`0.494029`、quantization coverageは`0.604212`、最大量子化誤差は`384.734576 ft`で、bank/quantizationの2固定checkをFAILした。Stage B総合はFAIL。
- decision manifestは`stage_a_and_b_pass=false`、`stage_c_status=blocked_by_stage_ab_gate`、`no_parameter_rescue=true`。Stage C、inference、submissionを実装・実行しない。
- 予約案4/5も開始条件不成立として閉じた。shift bank、sigma、threshold、decoderの事後救済を行わない。
- target-free contract SHAは`8ab762faff47b5d402064c88cfd0b6cdbc271c92b18e4ff1de1342b6c37186c4`、decision manifest SHAは`00eb8e81d8b82ff5ae5774b5cced0f5655b14c935f7c8be5a9c53eb6827b309c`。
- 大容量outputは取得せず、ログと小さいmetrics/manifestだけを取得して一致を確認した。path/score/block/readoutのraw/decompressed/schema/logical SHAは`metrics.json`と`result.md`へ記録した。

## 2026-07-21 Stage A/B実装

ユーザーの依頼「exp321を実装してください」を、凍結済み設計のStage A/B実装開始承認として扱った。Kaggle CPU Run AB、Stage C実装/run、inference、submissionの承認には拡張していない。

- compact self-contained train Jupytext source / Notebookと正規train Notebookを実装した。
- 最後の有限`TVT_input`行だけをanchorにし、`tvt_z=anchor_tvt-(Z-anchor_z)`を係数fitなしで生成する。known prefix非連続、raw suffix row identity不一致、anchor/suffix Z欠損はtechnical FAILにした。
- exp226 OOFはpre-freezeで`well_id/fold/row_idx/suffix_offset/tvt_geop`だけを読み、Z-only path、H128/H256/H512 block、Stage B scoreをgzip raw/decompressed/schema/logical-content SHA付きで凍結する。`tvt_true`はfreeze contract SHA確定後に別readerで初めて読む。
- Stage AはZ-onlyと保存exp226 `tvt_geop`を同一blockで比較し、direct / offset-only / affine quotient、singleton除外、lag-1、block mean/slope、H512 cap4 oracle diagnosticと固定gateを実装した。oracle値をrow predictionやStage B入力にはしない。
- Stage Bはexp280の固定13 shift、非重複512行block、known-prefix sigma clip `[10,60]`、raw-GR/typewell Gaussian emission、clip 600、config順tie、SHA256 local RNG shuffleを固定継承した。pooled exp280 strict比較、5/5 folds、1000+、hidden-like 2面、bank/quantization/finite/identity gateを実装した。
- A/B decision manifestは、両方PASS時だけStage Cの別実装承認を許す。Stage Cロジックは実装せず、inference Notebookもfail-closedとした。
- 専用synthetic/contract testsを10件追加した。

実装直後の実行量契約はactive variant 1、diagnostic contract 1、fold strata 5、model config / trained fold / booster / HMM / window decoder `0 / 0 / 0 / 0 / 0`、parent/control再実行0。後続の明示承認で同契約のKaggle CPU Run AB version 1を実行した。

```bash
.venv/bin/pytest -q tests/test_exp321_z_only_residual_gr_correction_ladder.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_train.py experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_train.py experiments/exp321_z_only_residual_gr_correction_ladder/exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_inference.py tests/test_exp321_z_only_residual_gr_correction_ladder.py --select F821
make validate-exp EXP=exp321_z_only_residual_gr_correction_ladder
```

- 専用10 tests PASS。Notebook/scaffold共通testを含む対象21 testsもPASS。
- Jupytext round-trip、`py_compile`、ruff F821、strict experiment validationはPASS。
- template validationはPASS。全体suiteは433 PASS / 1 SKIPで、既存`exp296`の実行前status/run approvalを期待する2 testsだけFAILした。exp321関連test、Notebook共通test、scaffold testはすべてPASSしており、exp296ファイルは本実装で変更していない。
- `__file__`依存はtrain/inference sourceにない。
- 構成参照元exp280 trainは9章 / 1,165行、exp298 compact trainは8章 / 2,283行。exp321 compact trainは10章 / 1,511行で、input preflight、Z-only path、GR scoring、freeze、Stage A/B late readout、gate、生成物保存をNotebook上で追える。
- exp305は科学的入力依存ではない。実装時点ではKaggle CPU train v2実行中であることを確認した。

## 2026-07-21 設計確定

```bash
make new-steering EXP=exp321_z_only_residual_gr_correction_ladder
make new-exp EXP=exp321_z_only_residual_gr_correction_ladder
```

- steeringを先に作成し、その後design-only scaffoldを作成した。
- `tvt_z=anchor_tvt-(Z-anchor_z)`、係数`-1`、last-known anchorを固定した。
- Stage A/Bは同一Run ABでtarget-free path/block/scoreをfreezeしてからtruthを結合する。
- Stage CはA/B全PASS後の別runとし、prediction freeze後だけtruthを結合する。
- Stage Bはexp280 parity、Stage Cはexp226 window GR correction parityとした。
- 案4/5は`reserved_followups.md`に未採番の別expとして固定し、exp321に含めない。
- 実装、test、Jupytext source、Kaggle package/push/run、inference、submissionは行っていない。

## 再現性メモ

- seed policy: real path/correctionはRNGなし。shuffleだけstable SHA256 per-well/block local RNG。
- stochastic components: matched shuffled negative controlのみ。
- CPU/GPU runtime: Kaggle CPU `611.963350 sec`、GPU/TPU/internet off。
- kernel id / version: `kentookumura/exp321-z-only-residual-gr-ladder-train` / version 1。
- input/schema/content SHA: target-free contract `8ab762faff47b5d402064c88cfd0b6cdbc271c92b18e4ff1de1342b6c37186c4`。全artifact SHAは`metrics.json`を正とする。
- prediction SHA: Stage C未実行のため未生成。
- model/submission SHA: 非該当。
- deterministic anchor: いいえ。train-side diagnostic/candidateでありinference未設計。

## 次のアクション

1. Stage B FAILによりStage C、inference、submission、予約案4/5を閉じる。
2. 同一truthでbank/threshold/sigma/decoderを救済せず、exp321を完了扱いにする。
3. 同系の新規backlogは追加せず、独立した既存優先実験へ戻る。
