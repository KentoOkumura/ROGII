# exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm セッションノート

## 目的

`exp223` self-GR HMMに対し、Type Well復元GRを「known prefix donorのraw missing cellだけ」に使う単一差分を設計する。欠損のないobserved GRまで置換する案、target区間をType Wellで復元する案、anchor数を増やす案は含めない。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 0 Kaggle CPU version 1 完了 / performance hard gate FAIL / branch closed
- Stage 0 RMSE: control 8.138530741 / variant 12.842185844 / delta `+4.703655102 ft`
- CV / LB: Stage 1 CVなし / LB未提出
- Stage 0 audit variant: 1
- Stage 1 HMM variant / well-runs: `0 / 0`実行（設計候補 `1 / 773` はStage 0 FAILで閉鎖）
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control retraining: なし
- GPU / inference / submit: なし
- 今回の実行規模: Stage 0 audit 1、LightGBM config 0、trained fold 0、booster 0、HMM/PF well-run 0
- 実行承認: 2026-07-19、親control再学習なし、Stage 1は未承認

## 根拠

- `exp223` の固定 `alpha=0.07` self-GR HMMはRMSE 11.349950650で、exp072 11.594897668を改善した。
- ただしexp209 HMM/likPF blend 10.269696に届かず、worst-wellは `+46.954683 ft`だった。
- `exp225` のstate-known curveはRMSE 14.212954500、1000+ `+2.931795 ft`、worst `+49.423573 ft`で失敗した。したがってType Well復元GRをtarget stateの直接emissionにせず、欠損donor信号の補完に限定する。
- recent physical/ML anchorsより優先度は低く、まず低コストStage 0でsignal reconstruction自体を反証可能にする。

## 確定した変更境界

- observed known GR exact保持。
- Type Well gap-fill対象は raw GR missing、finite `TVT_input`、Type Well range内のknown-prefix donor cellだけ。
- per-well deterministic Huber IRLS: minimum pairs 32、Type Well GR IQR 5以上、`k=1.345`、20反復、MAD scale floor 1.0。
- Type Well外挿なし。無効fit/範囲外はexp223線形補間へfallback。
- raw missing mask、anchor center/window eligibilityを固定。target receiverとbase HMM emissionも固定。
- active HMMはexp223 `alpha=0.07 / clip=1.0 / boost_only` 1本。

## 二段階ゲート

### Stage 0

stable SHA256 well-hash 5 foldsを作り、fold外の自然missing-run q25/q50/q90長でknown GRを決定的にmaskする。held-out rowをfit/interpolation入力から除外し、existing linear controlとType Well gap-fillを比較する。

PASS条件はpooled RMSE 5%以上改善、ZNCC `+0.02`、各方向4/5 folds、by-well p95非悪化、observed GR/raw mask parity、target fill 0、finite coverage 100%。FAIL時はStage 1もrescue gridも行わない。

### Stage 1

Stage 0全PASSと別承認後のみ1 HMM variant / 773 well-runsを実装・実行する。exp223比overall `-0.10 ft`、4/5 folds、1000+/hidden-like `+0.02 ft`以内、worst `+0.25 ft`以内、missing 0 wells exact prediction parity、low-missing scope非悪化を必須にする。

Stage 1 PASSでもexp209 10.269696以下でなければraw-test portへ自動昇格しない。

## コマンドログ

2026-07-19に設計スキャフォールドだけを作成した。

```bash
make new-steering EXP=exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm
make new-exp EXP=exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm
```

2026-07-19にユーザーの実装指示を受け、Stage 0だけを別名 compact self-contained notebookへ実装した。既存の正規名train notebookは上書きしていない。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/tests/test_exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm.py
make validate-exp EXP=exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm
make validate-template
make test
```

結果:

- Jupytext test / syntax / Ruff: PASS。
- `make validate-exp`: strict PASS。
- `make validate-template`: PASS。
- 専用contract tests: 11 PASS（合成データend-to-end生成物契約を含む）。
- repository tests: 298 PASS / 39.89秒（実行完了記録後の最終run）。
- `task` CLIはこの環境に存在しなかったため、同じMakefile targetを使用した。
- この実装直後の時点ではローカルnotebook実行、Kaggle prepare/push、Stage 0実データ実行は行っていなかった。後続の明示承認後にKaggle CPUだけを実行した。

2026-07-19にユーザーから実行承認を受けた。別名compact sourceを正規train notebookへ採用し、`execution.run_stage0: true`でKaggle CPU Stage 0を1 audit variantだけ実行する。実行前countはLightGBM config / trained fold / booster `0 / 0 / 0`、HMM/PF well-run 0、親control再学習なし。Stage 1 / inference / submissionは無効のままとする。

初回prepareではkernel slugへ実験名全体を使ったが、`kaggle kernels push`がnotebook実行前にHTTP 400で拒否した。`kaggle kernels list --search exp294`で作成済みversionは確認されず、実処理・boosterは0。64文字slugを再利用せず、意味を維持した43文字のcanonical id `kentookumura/exp294-typewell-gapfill-selfgr-stage0-train` / title `exp294 typewell gapfill selfgr stage0 train`へ短縮して再prepareする。

短縮後のpushは成功し、Kaggle CPU version 1（id_no `127890033`）が開始した。push直後に`kaggle kernels pull ... -m`でid/title、private、CPU、internet/GPU無効、competition sourceを照合し、初回statusは`RUNNING`。同じslug/versionを再pushせず監視する。

Kaggle CPU version 1はstatus `COMPLETE`、Stage 0本体160.32秒で完了した。773 wells / 2,319 blocks / 3,865 held-out rowsを評価し、control RMSE 8.138530741に対しvariant 12.842185844、delta `+4.703655102 ft`、relative improvement `-57.7949%`。fold deltaは`+4.773149 / +4.434234 / +2.476670 / +5.730062 / +6.159079 ft`で改善0/5、by-well p95 delta `+15.494310655 ft`、157 wells改善 / 610悪化 / 6同値だった。

hard gateはperformance 5件をFAILし、Stage 1は`stage1_authorized: false`。all-fold presence、finite coverage 1.0、observed GR exact parity、raw missing mask exact parity、pseudo-mask fit overlap 0、target-side fill 0のtechnical 6件はPASSした。自然欠損run長は全foldでq25/q50/q90 `1/1/3`行だったためminimum length 4のZNCCは未定義となったが、RMSEとp95だけでも棄却は確定する。契約どおりblock長/ZNCC定義を変更せず、Stage 1、救済grid、inference、submissionを閉じる。

`kaggle kernels output`で`kaggle/output/train_v1`へ生成物を取得した。artifact manifest 9 entriesのbyte数、raw SHA、gzip展開後SHAは全一致。主要SHAはinput manifest `62cb5af4...c9bf`、feature schema `b48ac632...13b0c`、pseudo-mask decompressed `92fe1685...49e33`、held-out prediction decompressed `fe147623...f451`、artifact manifest `b003790c...bca3`、kernel log `53d457aa...da96`。

## 実装内容

- `TVT`を読み込まないsafe horizontal loaderを実装し、mask/fit/prediction freeze前の入力を`TVT_input`と`GR`だけに固定した。
- stable SHA256 first-8-byte big-endian well fold、fold外自然missing-run q25/q50/q90、決定的・非重複block選択を実装した。
- pseudo-mask manifestをtruthなしでgzip保存しdecompressed content SHAを固定してから、held-out GRを評価側へjoinする三相構成にした。
- Type Well duplicate TVTのGR median集約、範囲内線形内挿、minimum pair/IQR付きdeterministic Huber IRLSを実装した。
- finite observed GR exact保持、known-prefix missingだけType Well gap-fill、失敗時exp223線形補間fallback、raw mask parity、target fill 0を実装した。
- pooled/fold/by-well RMSE、block長加重ZNCC、MAE、derivative NCC、全hard gate、input/output/SHA manifestを実装した。
- 実装時は`execution.run_stage0: false`でfail-closeし、2026-07-19の実行承認後にStage 0だけを`true`へ変更した。実行完了・FAIL判定後は再び`false`へ戻し、Stage 1関数なし、inference/submissionなしを維持している。

## 親notebookとの構成比較

親exp223には`*compact_selfcontained*_train.py`が存在しなかったため、正規Jupytext source（190行、6章）を比較対象にした。exp294 compactは1263行、8章で、runtime/path/SHA、input scan、pseudo-mask、Type Well校正、hybrid reconstruction、metrics/hard gate、artifact orchestration、cost guardをnotebookセル上で追える。exp294内helper `.py` importや`__file__`依存はない。

## 再現性メモ

- seed policy: `stable_sha256_per_well_fold_runlength_no_global_rng`
- stochastic components: なし
- CPU/GPU runtime: CPU-only、outer workers 2、GPUなし
- deterministic anchor: false。train-side no-training auditでありsubmission anchorではない
- gzip: decompressed content SHAを主証拠、raw gzip SHAを補助証拠にする
- Stage 0実装済み記録: raw horizontal/typewell file SHA、input manifest SHA、schema SHA、pseudo-mask gzip raw/decompressed SHA、held-out prediction gzip raw/decompressed SHA、metrics/summary SHA
- saved exp223 control / exp115 assignments: Stage 0では不使用。Stage 1承認後の実装時にのみ必要
- model manifest: `not_applicable_no_trained_model`
- submission SHA: `not_applicable_no_submission`

## 次のアクション

実験完了。Stage 0 FAILのためStage 1、affine/window/alpha/threshold救済grid、inference、submissionへ進めない。本結果だけから新しい救済backlogは追加しない。
