# exp505_exp490_tau500_fade_fixed13_on_exp413 セッションノート

## 目的

raw exp490をstandalone採用せず、tau=500 fade候補としてfixed13 selectorへ渡し、selectorの
相対tail改善を確認してからexp413 downstream TVT MLへ受け渡す二段階実験を設計する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage C complete / scientific gate FAIL / terminal close
- Stage C / D: 40 CPU boosters完走・FAIL / 未実装・実行禁止
- CV / LB: hard OOF `8.243315437` / 未提出
- inference / submission: 未承認
- 正規train Notebook: compact self-contained候補の採用承認済み
- 正規inference Notebook: markdown-only placeholder、code cell 0
- compact train候補: Jupytext percent `.py`と派生`.ipynb`を作成済み

## 変更点

- exp501のfixed13 candidate inventoryを増やさず、raw exp490 slotだけをtau=500 fadeへ置換する。
- Stage C selector-tail gateと、PASS後だけ有効なStage D exp413 replacement gateを追加する。
- 親control、候補生成器、fixed12 / fixed7、fold、モデル設定は固定する。

## 2026-08-02 設計確定

- exp504は既存のblock-rank仮説で使用済みのため、次番号exp505を採用した。
- ユーザー指定どおり、standaloneではなく全体MLへ渡す候補として設計した。
- fadeを`alpha=1`、`tau=500 ft`、
  `exp357 + (1-exp(-md_since/500))*(exp490-exp357)`に固定した。
- exp501のraw exp490 slotをfade候補へ1対1置換し、候補数13、fixed12、fixed7 fallback、
  outer 5 / inner 4、2 objectives、モデル設定、scopeを固定した。
- Stage C progression gateはraw exp501 pooled非劣化、4/5 folds、固定7 scope、fade利用、
  fixed12比by-well p95 0.10 ft以上縮小、worst 1.0 ft以上縮小の全ANDとした。
- Stage C PASS後だけ、exp413 clean273 + exp505 compact77 + signed23 = final373を評価する
  Stage Dを許可する設計にした。
- Stage D gateはexp413比gain 0.03 ft、3/5 folds、固定5 scope、by-well p95 / worst
  `+0.25 ft`以内、technical全PASSの全ANDとした。
- tau / alpha / cutoff grid、raw+fade 14候補、well gate、direct final blend、same-OOF rescueを禁止した。
- tau=500はexp503 full OOFの29 profile後に選ばれた探索的値であり、clean independent CVと
  呼ばないことを固定した。

## 将来の実行量契約

| 段階 | variant | config / objective | fold | 新規booster | runtime |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage C selector | 1 | 2 objectives | outer 5 × inner 4 | 40 | CPU |
| Stage D downstream | 1 | 3 configs | outer 5 | 15 | GPU |
| 最大合計 | - | - | - | 55 | CPU + GPU |

- exp264 / exp501 / exp413 control再学習: 0。
- exp490 / exp357 / HMM / PF / Beam再実行: 0。
- Stage Dの15 GPU boostersはStage C PASS後に別途明示承認を得る。

## 再現性メモ

- `docs/06_reproducibility.md`を確認済み。
- seed 42、exp263 outer/inner fold、candidate順を固定する。
- fade生成はdeterministic、global RNG 0、stable row orderとする。
- Stage CはCPU LightGBM `deterministic=true / force_col_wise=true / n_jobs=8`を継承する。
- Stage D GPUはbitwise reproducibleと仮定せず、model manifest / OOF SHAを記録する。
- exp490 gzipはraw SHAとdecompressed content SHAを分け、後者を主証拠にする。
- package作成時はmetadataとbootstrap ZIP内configを照合する。
- design時点のmodel / prediction / submission SHA: not applicable。

## 実装・実行状況

- scientific source: Stage C compact self-contained候補を実装済み。
- helper: 同一実験helper importなし。再利用`src/` pipelineだけを使用。
- contract: candidate / feature YAMLを作成済み。
- contract test: 5件PASS。
- canonical train / inferenceロジック: なし。
- Kaggle package / kernel / run: なし。
- model / OOF / current-test / submission: なし。

## 2026-08-02 Stage C実装

- ユーザーの`exp505を実装してください`をStage C実装承認として記録した。
- 既存正規Notebookは上書きせず、
  `exp505_exp490_tau500_fade_fixed13_on_exp413_compact_selfcontained_train.py`を作成した。
- exp490 sourceはfeature freeze前にexact 8列だけを`usecols`で読み、raw gzip SHAと
  decompressed payload SHAを検証する。
- fadeは`alpha=1 / tau=500`だけを許可し、negative `md_since`をclipせず拒否する。
- global `(well,row_idx)` join後、exp263 foldごとに`suffix_offset`と`md_since` parityを検証する。
- Stage Aはtruth-free mechanical dropだけ、Stage Cはexp501固定条件の1 variant × 2 objectives ×
  outer 5 × inner 4 = 40 CPU boostersとした。control再学習0、HMM/PF/Beam再実行0、GPU 0。
- score / choice / compact / SHA freeze後だけraw TVT、saved exp501 score、saved exp264 score、
  hidden-like roleを読み、direct fade parity、pooled/fold/固定7 scope、candidate利用率、
  fixed12比p95/worst縮小、rerankingを全AND判定する。
- Stage D code pathは追加せず、PASS・別承認前の実装/実行を防いだ。

### 静的検証

- `.venv/bin/python -m py_compile ...compact_selfcontained_train.py`: PASS。
- `.venv/bin/ruff check ...compact_selfcontained_train.py test_exp505_contract.py`: PASS。
- `.venv/bin/python -m pytest -q .../test_exp505_contract.py`: `5 passed`。
- `.venv/bin/python -m pytest -q tests/test_exp263_candidate_cache_contract.py
  tests/test_exp264_candidate_selector_pipeline.py`: `31 passed`。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: PASS。
- `make validate-exp EXP=exp505_exp490_tau500_fade_fixed13_on_exp413`: strict PASS。
- `task` commandは環境に存在しなかったため、skill記載のfallbackとして`make`を使用した。
- 全repo `make test`は1,842件を収集し、関連するexp263/264を含む11%までfailureなしを確認した。
  全実験横断suiteは本変更に対して過大なため手動停止し、上記関連31件を独立完走した。
- 親exp501 compactは624行 / 9章、exp505 compactは約1,280行 / 同じ9章。fade loader、
  truth-late raw-exp501比較、tail gateをNotebook上へ展開しており、薄いhelper呼び出しではない。
- runtime helperは`Path.cwd()`起点で、source-file locationへ依存しない。

## 2026-08-03 Stage C実行承認

- ユーザーの`実行してください`を、正規train Notebook採用、Kaggle CPU package、Stage C
  push/runの承認として記録した。
- 実行するのは1 variant、2 objectives、outer 5 × inner 4、合計40 CPU boostersだけ。
- exp264 / exp501 control再学習0、HMM / PF / Beam再実行0、GPU 0。
- Stage Dの15 GPU boosters、inference、submissionは承認対象外のまま維持する。
- 承認時刻: `2026-08-02 23:17:06 UTC` (`2026-08-03 08:17:06 JST`)。
- Kaggle入力Notebook outputとしてexp490 / exp263 / exp264 / exp501の4 sourceをAPIで確認した。
- bootstrap入力のexp251 selected feature schemaとhidden-like assignmentをローカルで確認した。
- 正規train Notebookは20 cells / 9 code cells / output 0で、Jupytext round-trip、py_compile、ruff、
  contract test 5件、strict experiment validationを再PASSした。
- Kaggle packageはbootstrap cell込み21 cells、support 38 files。manifest内の全byte count / SHA、
  必須2入力、config SHAの一致を確認した。
- kernel metadata: `kentookumura/exp505-exp490-tau500-fade-fixed13-on-exp413-train`、
  private、CPU、internet off、run-on-push、4 kernel sources、dataset sources 0。
- package Notebook SHA256: `6895a87b8318b7d930ebed70b24c2da5cf53b9f679b3e1fd44f9e2f6c4999907`。
- kernel metadata SHA256: `b26d7d22095d0a33d9b30a9f62790cae17dad413e677f5a071a2c68a7045f3d2`。
- embedded/local config SHA256: `919fd232cb1fcb1e4c088381d87c85d11e5725af59ebde424a8f6eef6cd946eb`。
- `2026-08-02 23:21:39 UTC`にKaggle kernel version 1をpushし、CPU runを開始した。
- remote kernel id_no: `129519165`。push直後のstatusは`RUNNING`。
- pullしたremote metadataでslug/title、private、CPU、internet off、competition source、
  4 kernel sources、dataset/model sources 0を再確認した。
- remote Notebookはbootstrap込み21 cells / outputs 0 / support manifest 38 filesで、
  pushed config SHA、scientific source SHA、hidden-like assignment SHAがlocal packageと一致した。

## 2026-08-03 Stage C完了

- Kaggle private CPU version 1（id_no `129519165`）が`COMPLETE`。
- scientific本体runtime `7312.583 sec`、最終log時刻 `7342.155 sec`。
- 1 variant / 2 objectives / outer 5 × inner 4 = 40 modelsを完走。control再学習0、
  HMM / PF / Beam再実行0、Stage D GPU 0、inference / submission 0。
- technical checksは全PASS。exact 8列allowlist、forbidden列pre-freeze読込0、raw/decompressed
  SHA、global key / suffix / `md_since` parity、source fold不使用、strict nested leakage、
  fixed7 parity、40 model / 25 partition / row count契約を満たした。
- direct fade RMSE `8.447032559794`はexp503期待値との差`2.06e-10 ft`でparity PASS。
- hard OOFはexp505 `8.243315437`、raw exp501 `8.264890209`、gain `0.021574771 ft`。
- fold 0--4のexp505 RMSEは`8.416177539 / 8.295453153 / 8.030499522 /
  8.232938395 / 8.236416367`。raw exp501比はfold 1だけ`+0.017154393 ft`で、4/5非劣化。
- 固定7 scopeは全てraw exp501を改善。fade top1率はpooled `55.2414%`、5/5 foldsで正。
- fixed12比by-well p95はraw `+2.904593926`、exp505 `+2.904557390`で縮小
  `0.000036536 ft < 0.10 ft`。worstはraw `+18.394664149`、exp505 `+18.221496070`で
  縮小`0.173168079 ft < 1.0 ft`。tail 2条件をFAIL。
- 判定は`FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`。Stage D、same-OOF救済、
  inference、submissionへ進まない。
- output archive全体は取得せず、完全logsと記録に必要なmetrics/fold/scope/usage/gateの
  小artifactだけを取得してSHAを照合した。

## 次のアクション

exp505は終端閉鎖済み。Stage Dを実装・実行せず、tau / alpha / threshold / feature / gateを
同一OOFで救済しない。保存exp501/exp505 artifactだけを使うtail不変原因readoutをP4候補に置き、
独立仮説の根拠が必要な場合にだけ別steering・別承認で着手する。
