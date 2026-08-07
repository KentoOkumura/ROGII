# exp284_prefix_masked_wrong_mode_branch_recovery_backtest セッションノート

## 目的

known prefix末尾640行のmasked pseudo suffixへwrong modeを注入し、safe base + self-GR top-3を
未来256行evidenceで比較するcontrolled recovery backtestを設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 2完了、scientific guard FAIL、branch closed
- CV / LB: controlled backtest / 対象外
- active variant / fixed policies: `1 / 5`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- HMM / PF regeneration: `0 / 0`
- parent/control再学習: 0
- GPU / inference / submission: off / disabled / disabled
- Kaggle push approval: true（2026-07-19 15:19 JST、ユーザー「実行してください」）

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp284_prefix_masked_wrong_mode_branch_recovery_backtest
make new-exp EXP=exp284_prefix_masked_wrong_mode_branch_recovery_backtest
```

上記は初回設計時のtemplate作成である。

### 2026-07-19 compact self-contained実装

ユーザーの「exp284の実装を進めてください」と、exp283とは独立に進められるかの確認を受け、exp283の
生成物へのruntime依存とscientific execution gateを分離した。exp283固定contractをexp284内へ
self-containedに実装し、exp283 PASSはKaggle実行・scientific promotionの先行条件として維持する。

実装ファイル:

- `exp284_*_compact_selfcontained_train.py` / `.ipynb`: 2,407行、10章。
- `exp284_*_compact_selfcontained_inference.py` / `.ipynb`: 127行、4章、fail-closed。
- `tests/test_exp284_prefix_masked_wrong_mode_branch_recovery_backtest.py`: 7 tests。

実装境界:

- exp226 OOFは元unknown suffixだけのためmasked prefixへ直接joinせず、保存済みfold/kappaとother-fold
  source geometry fieldからpseudo cutのexp226 geometry増分を再生する。
- current held-out foldのtarget readerは`TVT`をloadしない。source-well `TVT`はfold-safe geometry field
  構築だけに使い、held-out post-cut truth access before freezeは0にする。
- wrong shiftのvisible 128-row referenceは観測済み`TVT_input`。shift bank / local maximum / tieを固定する。
- proposalはeventでframeをtruncateしてからGR補間とcausal trailing rolling mean 5を行い、17/31/51、
  forward/reverse、dedup、global top-3を固定する。
- safe/wrong/real top-3/shuffled top-3の8 branch、H128/256/512、5 policiesをtarget-freeでfreezeする。
- mask / injection / proposal / branch path / evidence / policyのlogical content SHA固定後にだけtruthを読む。

検証コマンド:

```bash
.venv/bin/ruff format experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/*compact_selfcontained*.py tests/test_exp284_prefix_masked_wrong_mode_branch_recovery_backtest.py
.venv/bin/python -m py_compile experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/*compact_selfcontained*.py tests/test_exp284_prefix_masked_wrong_mode_branch_recovery_backtest.py
.venv/bin/ruff check experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/*compact_selfcontained*.py tests/test_exp284_prefix_masked_wrong_mode_branch_recovery_backtest.py --select F821
.venv/bin/pytest -q tests/test_exp284_prefix_masked_wrong_mode_branch_recovery_backtest.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/exp284_prefix_masked_wrong_mode_branch_recovery_backtest_compact_selfcontained_inference.py
```

結果:

- `py_compile`: PASS。
- ruff F821: PASS。
- ruff full check: PASS。
- 専用tests: 7/7 PASS。
- Jupytext train / inference sync: PASS。
- strict `make validate-exp`: PASS。
- repository full test: 192 passed / 1 failed。FAILは今回未変更のexp264で、config inference status
  `corrected_inference_v4_complete`とtest期待値`user_authorized_2026_07_19`の既存不一致。
- `__file__`参照: 0。
- 正規stub train/inference `.ipynb`: 未上書き。
- compact train: 22 cells、source SHA `6eb6e980...b350`、notebook SHA `8846cbcc...d38a`。
- compact inference: 11 cells、source SHA `94c88fa5...928e`、notebook SHA `d11cdba0...fd7`。
- Kaggle prepare/push/run、ローカルnotebook実行、推論、提出: 未実行。

## 固定scientific contract

- mask 640 rows、visible minimum 512 rows、observation 128 rows、K=3、primary H=256。
- wrong shiftはcut直前128行と固定13 shift bankから`abs>=10 ft`で決定。
- safe baseは常時保持。5 fixed policies、real/shuffled、no-injection paired control。
- exp283と同じproposal/evidence/vetoを変更せず使用。
- full target-free freeze後だけpost-cut truthをattach。
- pair AUC、wrong-only/pair-only gain、H512 persistence、false switchを固定guardする。

## 再現性メモ

- cut、shift、branch orderはRNGなしで決定的。
- shuffleのみstable SHA256 local RNG。
- mask/injection/proposal/evidenceのschema/content SHAを別保存する設計。
- model/prediction/submissionは作らないためSHA対象外。
- deterministic prediction anchorではなくcontrolled diagnosticとして扱う。

## 次のアクション

ユーザーの実行承認に基づき正規notebook採用とstrict package準備を完了した。2026-07-19 15:51 JSTの
追加実行指示によりexp283 scientific gateをoverrideし、固定契約のexp284単独Kaggle CPU v1へ進む。
実行契約は1 variant / 5 policies / LightGBM config 0 / trained fold 0 / booster 0 /
HMM/PF regeneration 0 / parent-control再学習0で固定する。

## 2026-07-19 Kaggle CPU v1 実行承認

ユーザーの「実行してください」を、別名compact notebookの正規notebook採用と、exp283全guard PASSを
scientific gateとする固定契約Kaggle CPU backtest 1回の実行承認として記録する。

- active backtest variant: 1
- fixed policy: 5
- LightGBM / model config: 0
- trained fold: 0
- total booster: 0
- HMM variant / well-run: 0 / 0
- PF variant / well-run: 0 / 0
- parent/control再学習・再生成: 0
- runtime: Kaggle CPU / GPU off / internet off / single process
- input kernel sources: exp226、exp209、exp115 train artifacts
- inference / submission: disabled / disabled
- prerequisite: exp283 technical / proposal / verifier scientific guardすべてPASS

credential preflightはAPI token未設定、OAuth credentialとlegacy credentialはOK。Kaggle CLIはOAuthを
利用できる。exp283のguard結果が確定するまではcanonical packageを準備してもpushしない。

exp283の73文字full slugがKaggle `SaveKernel 400`となり、49文字の意味を保った短縮slugでversion 1が
開始できた実測を受け、exp284も未pushの段階で62文字full slugを避ける。実行kernelは48文字の
`kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`、titleは
`exp284 masked wrong mode recovery backtest train`に固定し、id/titleのslugを一致させる。

## 2026-07-19 exp283 scientific gate override

ユーザーへ「exp284はruntime / artifact依存としてはexp283から独立し、依存しているのは実験順序だけ」
と説明した後、ユーザーから再度「実行してください」と明示指示を受けた。この指示を、元のexp283
all-guard scientific gateをoverrideし、self-containedなexp284固定backtestを単独で1回実行する承認と
して記録する。

- override status: `user_authorized_independent_execution`
- runtime / artifact dependency on exp283: なし
- 変更しないもの: mask 640、observation 128、K=3、H=256、5 policies、全guard、seed、input SHA
- 結果の解釈: exp284 controlled backtest単独の成否であり、exp283 PASSの代用や主張には使わない
- parameter rescue: 引き続き禁止
- monitoring: ユーザーの直前指示どおり、push後の定期監視は行わず完了連絡を待つ

override反映後のcompact trainは2,423行 / 22 cellsで、正規train notebookへ再採用した。専用tests
7/7、`py_compile`、ruff full、Jupytext train/inference round-trip、strict `validate-exp`はすべてPASS。
unauthorized overrideをfail-closedする合成確認も専用contract testへ含めた。

- train source SHA: `321d23c3...7073`
- compact train notebook SHA: `675597ac...6877`
- regular train notebook SHA: `59ad2806...6dee`
- inference source / regular notebook SHA: `94c88fa5...928e` / `5ec256da...8ba`

### 独立実行package監査

short canonical id/titleを同時指定してstrict packageを再生成した。package notebookはbootstrap 1 cell +
正規22 cells、cell output 0で、package内`config.yaml`はrepo側とbyte一致した。metadataはprivate、CPU、
GPU/TPU/internet off、run-on-push true、competition source 1件、固定kernel source 3件である。

- kernel: `kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`
- title: `exp284 masked wrong mode recovery backtest train`
- config SHA: `65eb54fe...f77e`
- train source SHA: `321d23c3...7073`
- regular train notebook SHA: `59ad2806...6dee`
- packaged train notebook SHA: `3d600f3b...af78`
- metadata SHA: `e8bd2219...caeb`
- package後tests: 7/7、strict `validate-exp`: PASS
- push前server metadata pull: 403。既存canonical kernelなし、重複versionなしと判断した。

### Kaggle CPU version 1 push

2026-07-19 15:56 JST、short canonical kernelへの初回pushが成功し、version 1を開始した。push直後の
server metadata pullも成功し、local packageと同じprivate / CPU / GPU・TPU・internet off、competition
source 1件、kernel source 3件を確認した。

- kernel: `kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`
- version / id_no: `1 / 127852894`
- URL: `https://www.kaggle.com/code/kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`
- executed config SHA: `65eb54fe...f77e`
- packaged notebook SHA: `3d600f3b...af78`
- monitoring: ユーザー指示により定期監視なし。完了連絡後にlogsと全guardを確認する。

### Kaggle CPU version 1 technical failure と version 2 修正

ユーザーの失敗連絡後にversion 1の最終logを1回取得した。fold 0のsource 618 wells / target 155 wellsを
確定した後、最初のtarget horizontal読込で停止していた。

- exception: `ValueError: Usecols do not match columns, columns expected but not found: ['id']`
- stage: fold 0 / first target `load_target_safe_horizontal`
- classification: evaluation前のtechnical input-schema mismatch
- scientific metrics / guards: 未生成・未評価
- root cause: 実horizontal CSVは`MD,X,Y,Z,GR,TVT_input`であり、合成test fixtureだけが`id`を持っていた

`id`はpredictionやbranch選択には使わず、mask対象行の監査SHAだけに使う。そこで物理入力6列だけを読み、
`<well>:<row_idx>`から決定的な監査専用`id`を生成するよう修正した。mask 640、observation 128、K=3、
H=256、5 policies、全guard、seed、fold、input、0 booster / 0 HMM / 0 PFは変更していない。version 2は
version 1と別variantではなく、同じ固定contractのtechnical retryとして扱う。

修正後は実test horizontal 7,559行を読み、`00bbac68:0`から`00bbac68:7558`までのaudit IDを確認した。
専用testsは8/8、`py_compile`、ruff full、Jupytext train round-tripをPASSし、正規train notebookも更新した。

- train source: 2,427 lines / SHA `4c23556c...0b0b`
- regular / compact train notebook: 22 cells / SHA `83765e45...a78`
- monitoring: ユーザー指示どおり定期監視は再開しない

version 2 packageは同じcanonical kernel id/titleへstrict生成した。package notebookはbootstrap 1 cell +
正規22 cellsで、cell outputは0。repo/packageのconfigとtrain sourceはそれぞれbyte一致した。metadataは
private、CPU、GPU/TPU/internet off、run-on-push true、competition source 1件、固定kernel source 3件で、
version 1から変更していない。package後も専用tests 8/8とstrict experiment validationをPASSした。

- v2 executed config SHA: `0308717d...96b9`
- v2 packaged notebook SHA: `13bfeca7...ae88`
- kernel metadata SHA: `e8bd2219...caeb`

2026-07-19 16:07 JST、同じcanonical kernelへのpushが成功し、version 2を開始した。これは入力schema
compatibilityだけを修正したtechnical retryで、実行variantやscientific contractの追加ではない。

- version: `2`
- URL: `https://www.kaggle.com/code/kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`
- executed config SHA: `0308717d...96b9`
- packaged notebook SHA: `13bfeca7...ae88`
- monitoring: ユーザー指示どおり定期監視なし

### Kaggle CPU version 2 完了・固定guard評価

ユーザーの完了連絡後、通常`kaggle kernels logs`を1回取得した。version 2は766 eligible / 7 ineligible
wells、5 foldsを11,717.244秒（約3時間15分17秒）で完走した。technical guardは全PASS、scientific / safety
guardはFAILし、summary判定は`close_without_parameter_rescue`だった。

- pair score-margin AUC pooled: `0.675153`
- pair AUC folds: `0.690068 / 0.833659 / 0.754505 / 0.509459 / 0.555936`
- pair AUC 5/5 `>=0.60`: FAIL
- pair choice accuracy pooled: `0.590078 < 0.60`、foldは全て`>0.50`
- H256 wrong-only / safe+wrong / full / shuffled / no-injection RMSE:
  `37.557085 / 23.633930 / 26.072230 / 25.520057 / 20.314398`
- full gain vs wrong H256: `+11.484854 ft`、5/5 folds改善でPASS
- full gain vs safe+wrong pair H256: `-2.438300 ft`、0/5 folds改善でFAIL
- full gain vs wrong H512: `+11.454901 ft`でH256を下回りpersistence FAIL
- real vs shuffled: real pooledが`+0.552173 ft`悪く、nonregressing 3/5 foldsでFAIL
- no-injection false switch: `30.1724% > 5%`でFAIL

technical側はeligible、5 folds、mask identity、branch/evidence finite coverage 1.0、固定branch identity、
minimum injected shiftを全PASSした。全target-free tableをpersist・content hashした後にtruthをattachし、
post-cut truth access before freezeは0だった。したがって結果は技術失敗ではなく、固定仮説のscientific FAIL
として信頼できる。

ログだけで総合判定は確定したが、fold別実数値とSHA照合のため、version 2 outputからsummary、contract、
overall/fold/pairwise/by-well metrics、mask/input/ineligible manifestだけをfile pattern指定で取得した。
3,137,536行のbranch pathなど大容量target-free table本体は取得していない。取得9ファイルのうちsummaryに
列挙された8ファイルはKaggle記録SHAと全件一致した。

- local metrics root:
  `/tmp/kaggle-output/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/train_v2_metrics`
- summary SHA: `3d9863ef...9e3e`
- overall / fold / pairwise: `d836b977...304` / `cbe7b823...9009` / `b025c18e...b623`
- by-well / contract / mask / input / ineligible:
  `8ebd8542...1a01` / `03d06cf1...d94e` / `74026afa...fbf` / `71b039ac...61e4` /
  `04b30d6b...68ae`
- target-free decompressed SHAは`metrics.json`へ記録した

safe baseを候補へ戻すことで意図的wrong-onlyからは回復したが、real self-GR top-3はsafe+wrong pairを全foldで
悪化させ、shuffled donorより悪く、false switchも高い。exp283 proposal-level signalを安全なbranch selectionへ
変換できないため、K/window/horizon/veto/margin/threshold救済、decoder接続、current-test生成、inference、
submissionへ進まない。exp285もlong-horizon prefix offset predictabilityを否定済みで、新規救済backlogは
追加しない。
