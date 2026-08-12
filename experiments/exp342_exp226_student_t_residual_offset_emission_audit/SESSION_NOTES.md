# exp342 セッションノート

## 2026-07-22 設計確定

- 目的: robust emissionの価値をfull HMMの前にshift順位で反証可能にする。
- Stage 0規模: Student-t scientific readout 1、Gaussian保存済みcontrol、circular negative control、HMM run 0。
- Stage 1規模予約: Stage 0全通過後かつ再承認時のみvariant 1、fold 5、773 well HMM run。
- leakage guard: score SHAをtruth-nearest shift join前に固定する。
- long-tail guard: pooledだけでなくfold、stress、`|z| >= 3` bucketを固定した。
- 既知制約: lag-1相関、欠損補間、depth aliasそのものは本変更では解決しない。

## 2026-07-23 Stage 0 実装

- ユーザーの `exp342を実装してください` を Stage 0 implementation と正規
  train/inference notebook 採用の承認として記録した。
- compact self-contained Jupytext train source（10章、1,658行）と fail-closed
  inference sourceを実装し、両compact notebookと正規notebookへ変換した。
- 親 exp280 train source（9章、1,165行）と比較し、runtime/input/score/truth-late
  readout/orchestrationを維持したうえで saved control loader、Student-t score bundle、
  circular control、stress/extreme gateを追加した。
- Gaussian control は exp280 target-free score decompressed SHA
  `c6e9e39a...1d99c3`、declared content SHA `4a546cfe...3aa46`、
  scientific contract SHA `60d32ba9...f7978`を hard guard し、再生成しない。
- Student-t は `df=4`、exp281/exp280と同じ known-prefix residual std、
  sigma clip `[10, 60]`、missing補間、typewell補間、512-row block、13 shiftを固定した。
- `|z|>=3` block は truth-nearest shift の residual が1行以上該当する block と固定し、
  新しい share threshold は導入しない。
- Student-t/control bundle content SHAを truth join 前に固定し、両familyへ同じ
  SHA256由来 nonzero circular rotation を適用する。
- Stage 0 gateは pooled MRR/top3各`+0.01`、各4/5 folds、1000+、
  hidden-like spatial/typewell-purged、persistent-offset、circular gap、
  extreme top3/regretのANDとした。
- exp344の解禁patternは、extreme top3/regret改善、pooled gate失敗、
  Student-t top1 margin低下が同時成立した場合だけ明示する。
- Stage 1 decoder、HMM kernel、prediction、inference、submission は実装していない。

## 実行コスト契約

- Stage 0 scientific score: 1 (`student_t_df4`)
- saved Gaussian control: 1（load only、再生成0）
- shift candidates: 13
- reporting folds: 5（trained fold 0）
- model config / trained fold / booster: `0 / 0 / 0`
- HMM well-run: `0`
- parent/control再学習・再生成: 0
- Kaggle CPU、GPU/TPU/internet off
- Stage 1予約: 全gate PASSかつ別承認時だけ 1 variant / 773 HMM well-runs

## 実装・検証コマンド

```bash
.venv/bin/python -m py_compile <exp342 compact train/inference.py> <exp342 test.py>
.venv/bin/ruff check <exp342 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q experiments/exp342_exp226_student_t_residual_offset_emission_audit/tests/test_exp342_exp226_student_t_residual_offset_emission_audit.py
# 7 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 <compact source.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact source.py>
.venv/bin/python scripts/validate_experiment.py --experiment exp342_exp226_student_t_residual_offset_emission_audit
# strict validation passed
```

## 再現性メモ

- Student-t real scoreはRNGなし。
- negative controlは `SHA256(experiment, well, block)` 由来の1--12 slot nonzero rotation。
- exp226 OOF decompressed SHA `709eb726...e4c609`、hidden-like raw SHA
  `5f9ac9fa...ca6597`を hard guard する。
- gzipはmtime=0で保存し、raw/decompressed/content SHAを分離する。
- fixed-input diagnosticであり、prediction/submission deterministic anchorではない。
- implementation SHA: config `dcc9edd2...d9eab0`、train source
  `f529b813...9a5f7`、inference source `5aa30d85...3eee0`、test
  `d26acf19...c7484`。
- compact/canonical notebook SHAは train `c0e01d83...8e240`、inference
  `3db775e9...f881a`で各byte一致。

## Repository full test

```bash
.venv/bin/pytest -q
# 676 passed, 5 skipped, 2 failed
```

- 失敗2件はいずれも既存exp296の完了後configと旧test期待の不一致。
  `experiment.status=completed_train_side_guard_failed_closed`に対し旧testが
  `kaggle_cpu_*`を要求し、`execution.run_variant=false`に対し旧testが
  push approval guardを先に期待している。exp342の専用・親・隣接test 23件はPASSしており、
  この実装ではexp296を変更しない。

## 未実施

- Kaggle package / push / run
- Stage 0生成物と科学値
- Stage 1実装・773 HMM well-runs
- inference / submission

## 次のアクション

Stage 0のKaggle CPU実行が必要なら、上記実行量を再提示して別承認を得る。
Stage 0 PASSでも Stage 1へ自動進行しない。

## 2026-07-23 Stage 0 Kaggle CPU実行承認

- ユーザーの「実行してください」を、正規compact self-contained train notebookの
  package / push / Stage 0 CPU実行の承認として記録した。
- 実行量は scientific score 1 (`student_t_df4`)、
  SHA固定済みsaved Gaussian control 1（load only）、13 shifts、5 reporting folds。
- model config / trained fold / booster / HMM well-runは `0 / 0 / 0 / 0`。
  親実験の再学習とGaussian control再生成も0。
- Kaggleはprivate、CPU、GPU/TPU/internet offで実行する。
- canonical kernel idはKaggle slug長を抑えつつ仮説を保つ
  `kentookumura/exp342-student-t-residual-offset-emission-train`、
  titleは`exp342 student t residual offset emission train`とする。
- Stage 1実装・773 HMM well-runs、inference、submissionは今回の承認対象外。
  Stage 0がPASSしても自動進行しない。

## Kaggle push

- canonical private CPU kernel version 1をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp342-student-t-residual-offset-emission-train`
- packaged config SHA256:
  `fe83166ba1a80f57260ac538bb3a83967f3f566d6c24eb53f5998a1a28b5acfc`
- canonical source notebook SHA256:
  `c0e01d834a1cae3f7935b0a79bcbd8a509dd3d569ecacf3c491a92924148e240`
- push直後にローカルのrun-on-push承認フラグをfalseへ戻し、誤再実行を防止した。
  Kaggleへ送信済みのpackage内configは承認済みtrueのまま保持される。

## Kaggle Stage 0 version 1 結果

- Kaggle status: COMPLETE
- kernel id / version / id_no:
  `kentookumura/exp342-student-t-residual-offset-emission-train / 1 / 128356155`
- generated at: `2026-07-23T12:21:25.092017+00:00`
- scientific runtime: `468.1274166107178 sec`
- 773 wells / 3,783,989 rows / 7,787 blocks / 13 shifts
- technical coverage、saved Gaussian rank parity、circular MRR/top3 gapはPASS。
- pooled MRRはStudent-t `0.390292296`、Gaussian `0.389625985`、
  差`+0.000666311`で必要な`+0.01`をFAIL。
- pooled top3はStudent-t `0.453576474`、Gaussian `0.452420701`、
  差`+0.001155772`で必要な`+0.01`をFAIL。
- fold改善はMRR 5/5、top3 4/5でPASSしたが、事前指定4 stress scopeを束ねた
  MRR/top3非劣化は両方FAIL。
- truth-nearest shiftに`|z|>=3`がある174 blocksではtop3
  `+0.022988506`、mean regret `-0.692700755 ft`で両gateをPASS。
- AND gate判定は`stage_0_failed_close_without_rescue`。Stage 1 eligible=false。
- flattening signal=false、exp344 dependency pattern=false。
  極端残差改善だけを根拠にpost-hoc Huberへ進めない。
- HMM / model config / trained fold / booster / parent-control再実行は
  `0 / 0 / 0 / 0 / 0`。inference/submissionも0。
- target-free score content SHA:
  `37ef297488d87d732c634ee31dd3d463e2760d22631f8d757e7a735d4d4a941f`
- block readout content SHA:
  `79d40fec3163163822a8210b8313ecfa17a457b7d93ad4512a2bb1b34006f0c7`
- gate SHA:
  `adbe2476841a2b361f8aff4593283196c9700d8de4d83c8e9723a0aae6cdd95a`
- df/scale/temperature/grid、Huber/cap、missing/ACFの救済、再実行、
  Stage 1、inference、submissionなしでexp342を閉じる。

## 2026-07-23 Stage 1探索実行override

- ユーザー明示依頼「Stage1に進んでください」により、Stage 0 FAILを保持したまま
  Stage 1の実装・Kaggle private CPU実行を承認された。
- Stage 0 gateの事後緩和ではなく、結果の解釈に事前条件違反を残す探索実行とする。
- active variant: `student_t_df4_residual_offset_delta80_step035_rate41` 1件。
- HMM well-runs: 773。LightGBM config / trained fold / booster:
  `0 / 0 / 0`。
- 親Gaussian HMMは再実行せず、Kaggle exp281 version 1のSHA固定済みOOFを
  post-freeze controlとして読む。parent/control再実行は0。
- exp281と同じoffset grid `[-80,80]`、step `0.35`、41 rates、
  `sig_r=0.002`、`sig_p=0.02`、prior、transition、sigma、missing、
  posterior meanを固定する。変更はGaussianから固定`df=4` Student-tへの置換だけ。
- Kaggle CPU、GPU/TPU/internet off、runtime limit 8.5時間。
  親exp281実績から所要約4時間を見込む。
- inference、submission、Gaussian control再実行、df/scale/temperature/grid、
  Huber/cap、missing/ACF変更は未承認。

## Stage 1実装

- 既存compact self-contained train sourceへ、exp281 exact forward-backward kernel、
  fixed df=4 Student-t行別emission、773-well orchestrationを追加した。
- Student-t全pathとlogical content SHAをfreezeした後だけ、SHA固定済みexp281 OOFの
  Gaussian parent、exp226、exp263、truth、fold、`md_since`をjoinする。
- overall/fold/1000+/hidden-like 2面/by-well p95/worst、exp226 direct promotionを
  固定gateで出力し、Stage 0 prerequisite=falseとoverride根拠をsummaryへ残す。
- Stage 1生成物は`*_stage1_*`名でStage 0生成物と分離する。
- parent exp281 train sourceは1,526行/10章。exp342 sourceはStage 0とStage 1を
  同一正規notebookで追える2,942行/12章で、同一exp helper importと`__file__`依存なし。
- py_compileとRuff `F821,F401,F841,E722,E501`をPASSした。
- 保存済みexp281実OOFを`float_precision=round_trip`で読み、3,783,989 rows /
  773 wells、decompressed SHA `3a99b1d9...7386`、logical content SHA
  `d7f902b8...e440`、Gaussian parent RMSE `9.827419941`のhard guardをPASSした。
- exact kernelは同一emission入力でexp281 posterior/loglikと完全一致するunit testをPASS。
- parent runtimeと同じNumba 4 threads、well直列実行を固定する。
- Jupytext train/inference round-tripをPASSし、compact/canonical notebookを
  byte一致で採用した。Stage 1 train source SHA `8266cd2f...ca82`、
  train notebook SHA `fe238d3a...bdfd`、inference source SHA
  `11882c35...e790`、inference notebook SHA `738a8c00...86c2`。
- exp280/281/342を含む22 tests、py_compile、Ruff、strict exp/project validationを
  PASSし、Stage 1 package/push/runフラグを承認済みtrueへ切り替えた。
- canonical private CPU packageは3 kernel sources（exp226/exp281/exp115）、
  competition source 1、GPU/TPU/internet off、1.3 MB。
- loose/package/bootstrap config SHAは
  `601b4c2af0f5694c3f505396617b8958a8a62c174c8c968cfa1571b4a19ac599`
  で一致し、bootstrap 18 support filesにStage 1 sourceを含む。

## Stage 1 Kaggle push

- canonical kernel
  `kentookumura/exp342-student-t-residual-offset-emission-train`へversion 2をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp342-student-t-residual-offset-emission-train`
- version 2はStage 1のみ。Stage 0、Gaussian HMM、inference、submissionを実行しない。
- push直後にloose configのrun-on-push承認をfalseへ戻し、誤再実行を防止した。
  Kaggleへ送信済みpackage/bootstrap configは承認済みtrueを保持する。
- Kaggle側でversion 2が`RUNNING`であることを確認した。
- ユーザー指示により継続監視を停止した。Kaggle実行自体は停止せず、
  完了連絡を受けた後に確定logs、gate、結果記録を取得する。

## 2026-07-24 Stage 1完了・判定

- ユーザーの完了連絡後、canonical kernel version 2の通常logsを取得した。
  `kaggle kernels status`は`KernelWorkerStatus.COMPLETE`、logsは最終summaryまで
  あり、fatal error patternは0だった。
- scientific runtimeは`14,789.392992 sec`、完了時刻は
  `2026-07-23T17:01:51.609982+00:00`。773/773 wells、3,783,989 rows、
  1 Student-t variant / 773 HMM runsを完了した。
- Gaussian parent、LightGBM config、trained fold、booster、inference、submissionは
  いずれも再実行・生成していない。
- Student-t HMM RMSEは`9.779771695`、保存済みexp281 Gaussian HMMは
  `9.827419941`。全体gainは`0.047648245 ft`で、固定下限`0.05 ft`に
  `0.002351755 ft`届かなかった。
- fold 0--4のStudent-t minus Gaussian RMSEは
  `+0.168350551 / +0.172174600 / -0.054307994 / -0.138495263 /
  -0.336755680 ft`。改善は3/5で、必要4/5をFAILした。
- 1000+は`-0.049642958 ft`改善したが、hidden-like spatialは
  `+0.014173989 ft`、typewell-purgedは`+0.220136048 ft`悪化した。
- by-well delta p95は`+1.063792859 ft`、worst well `77b0d905`は
  `+12.893601646 ft`でtail safetyをFAILした。
- exp281/exp226 metric parity、finite coverage 1.0、row identity 1.0はPASS。
  失敗は入力・実装・数値異常ではなく、科学改善量と安定性の不足による。
- exp226 RMSE `9.427109597`に対して`+0.352662099 ft`悪く、
  direct promotionもFAILした。
- decisionは`stage_1_failed_close_without_rescue`。Stage 0 prerequisite=falseと
  `explicit_user_override_after_stage_0_fail`を結果へ保持した。
- candidate logical content SHA:
  `9af93eecb7bfcf0b43bbfbe9a0d759abf6031b0744da637147b05cbac72c38b7`
- decoder manifest content SHA:
  `17453228ad41ede225ff2b6b51e35c2908d20fbf4a58ef78d0b31cfb6246ff4f`
- prediction decompressed SHA:
  `767bf0726e696fc291923438ae2f87fafc1642b857f5c26bfaa9361505e9820b`
- gate SHA:
  `fe217ee638993ed3f7977ee5b479f56909520c22cb9eb0077bfadc70a3aaa790`
- CV、fold、scope、gate、SHAがKaggle logsの最終summaryに揃っていたため、
  AGENTS.mdの方針どおりoutput archiveは取得していない。
- Stage 0 proxy FAILでもfull HMMは全体で小幅改善したため、
  「HMMが絶対に改善しない」という結論ではない。ただしfull HMM自身の固定gateも
  FAILしたため採用しない。
- df/scale/temperature/grid、Huber/cap、missing/ACF、blend救済、再実行、
  inference、submissionなしでexp342をterminal closeする。exp344も閉鎖を維持する。
