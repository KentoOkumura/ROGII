# exp301_gauge_invariant_multiformation_edge_potential セッションノート

## 目的

6 formationのwithin-well edge differenceを積分するgauge-invariant 2D potentialを、fold-safeかつ
反証可能なdirect candidate generatorとして設計・実装する。案2/案3は後続分岐契約として同じ実験配下に固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU v2完了・Stage 0 FAILでbranch closed
- CV / LB: Stage 1未実行 / 対象外
- solver / audit / Jupytext source / tests: 実装済み
- 実装承認 / Kaggle実行承認: `true / true`

## 2026-07-20 設計セッション

- exp301採番を確認した。
- steering、standard scaffold、config、fail-closed notebookを作成した。
- exp289と異なる観測・状態・failure policyを固定した。
- Stage 0 identity/support guard、Stage 1 sparse potential、outer-train-only lambda選択を固定した。
- direct qualityとexp293 H512 add-one candidate noveltyの両方をpromotion必須にした。
- 案2/案3の正本を`reserved_followup_contract.md`に固定した。
- 実装、ローカルnotebook実行、Kaggle prepare/push、artifact生成は行っていない。

## 2026-07-20 実装セッション

- ユーザーの`exp301を実装してください`を実装承認として記録した。
- compact self-contained train sourceへ、safe loader、Stage 0、固定sparse solver、条件付きStage 1、prediction freeze、late truth join、direct/novelty gate、SHA保存を実装した。
- 既存の正規train/inference placeholderは上書きせず、compact self-contained train notebook候補を別名で生成する。
- 専用unit tests 9件を追加した。synthetic testsは科学結果ではない。
- Kaggle prepare/push、ローカルnotebook実行、inference、submission、案2/案3は実行していない。

### 実装検証

- Jupytext: compact sourceから別名`.ipynb`を生成し、`jupytext --to ipynb --test`をPASS。
- static: `py_compile`とfull `ruff check`をPASS。compact sourceに`__file__`と同一exp helper importは0件。
- experiment: `task`は環境に存在しなかったため、同等の`make validate-exp EXP=exp301_gauge_invariant_multiformation_edge_potential`でstrict PASS。
- tests: exp301専用9件 + Kaggle notebook 4件の計13件をPASS。
- full suite: 366件中363 PASS / 1 skip / 2 FAIL。FAILは既存exp296の完了済みconfig（`completed_train_side_guard_failed_closed`、`run_variant=false`）と、旧testの実行中期待（`kaggle_cpu_*`、`run_variant=true`）の不一致であり、exp301変更外。exp296は変更していない。
- notebook構成: 21 cells（Markdown 10 / code 11）、output 0、execution countなし。
- 親compact比較: exp289 1,397行 / exp293 1,963行に対し、exp301は9章を持つ2,300行超のself-contained sourceで、safe input、solver、freeze、diagnostic、保存までをnotebook上で追える。

## Kaggle push前の計算量契約

実装値を再確認済み。2026-07-20のユーザー依頼`実行してください`により、compact候補の正規train notebook採用と
固定contractのprivate Kaggle CPU train 1回を承認済み。inference、submission、案2/案3は未承認のままとする。

| 項目 | 設計値 |
| --- | ---: |
| scientific variant | 1 |
| outer evaluation folds | 5 |
| inner folds per outer fold | 3 |
| lambda candidates | 3 |
| final outer solver fits | 5 |
| LightGBM configs | 0 |
| boosters | 0 |
| trained neural models | 0 |
| PF/Beam well-runs | 0 |
| parent/control retraining | 0 |
| GPU | なし |

inner selectionは`5 x 3 x 3 = 45` holdout solves、finalは5 solves、合計最大50 solver fit。Stage 0 runtime/nnz guardで
Kaggle CPU 9時間内に収まらないと判明した場合、勝手にgrid/strideを変えずユーザーへ設計変更を確認する。

## 2026-07-20 Kaggle CPU v1 実行セッション

- scientific variant 1、outer fold 5、inner fold 3、lambda 3を再確認した。
- 最大solver fitはinner `5 x 3 x 3 = 45`とfinal 5の合計50。
- LightGBM config 0、booster 0、parent/control再学習0、GPUなし。
- 入力kernel sourceは既存のexp226 OOF、exp263 cache、exp115 hidden-like assignmentの3件だけを使用する。
- compact self-contained train notebookを正規train notebookへ採用し、package/SHA検証後に1回だけpushする。
- local notebook実行、inference、submissionは行わない。
- 正規train notebook採用後、Jupytext、py_compile、ruff、専用+notebook tests 13件、strict experiment validationをPASSした。
- Kaggle metadataはprivate、CPU、internet off、run-on-pushで、exp226/exp263/exp115の3 kernel sourceを固定した。
- bootstrap 16ファイルを実際に展開し、loose/package/bootstrapの全SHA一致を確認した。
- push対象config SHA256は`58ea8abc2d612f98ff3b941f310e77a31aab85d053bd3b74b4ad769d2b914bd6`。
- 親kernel metadataの取得を確認した（exp226 id_no 126463591、exp263 id_no 127474050、exp115 id_no 124519917）。
- 初回のfull-name slug `exp301-gauge-invariant-multiformation-edge-potential-train`はKaggle `SaveKernel 400`で拒否され、kernel実行は作成されなかった。既存実験と同じ既知の長slug制約と判断し、科学contractを変えず短縮slug `exp301-gauge-edge-potential-train`へ再prepareする。
- 短縮slugの再packageもbootstrap/loose/config SHA一致をPASSし、Kaggle kernel version 1をpushした。
- pull metadataでcanonical id `kentookumura/exp301-gauge-edge-potential-train`、id_no `128007163`を確認した。
- version 1用の初回train push承認は消費済み。下記pre-solver実装バグだけは元の実行依頼の未達分としてversion 2技術リトライへ引き継ぐ。

### version 1 technical ERROR と version 2 修正

- version 1は約59秒、Stage 0 outer fold 0のconstraint matrix構築で`KeyError: 'x_start'`となり、Kaggle status `ERROR`で終了した。
- 原因は`edges_all["solver_eligible"]`というboolean Seriesをfiltered edge DataFrameとして渡した1行の型バグ。solver fitは0で、科学評価やartifact生成には到達していない。
- `edges_all.loc[edges_all["solver_eligible"]].copy().reset_index(drop=True)`へ局所修正した。grid、stride、lambda、閾値、入力、fold、科学contractは変更していない。
- 同じ`prepare_outer_fold`経路を通してedge DataFrame列とconstraint matrix行数を確認する回帰テストを追加し、専用+notebook tests 14件とruffをPASSした。
- 上記は科学的な再実験ではなく、元のユーザー実行依頼を満たすためのpre-solver技術リトライとして同一canonical kernel idのversion 2を許可する。version 2 push後は再度承認消費済みとして扱う。
- version 2 packageのbootstrap 16ファイルとloose/package/config SHA一致を再確認した。push対象config SHA256は`ed25a930d1b96d7a9f599a935bcc012391baa1f4e716829b55e3504d6e6bcebb`。
- 同一canonical kernel idへversion 2をpushし、pull metadataでid_no `128007163`が維持されていることを確認した。追加の再pushは行わない。

### version 2 Stage 0結果

- Kaggle statusは`COMPLETE`。Stage 0は約94.5秒で5 outer foldsを完了した。
- formation別identity最大RMSEは`0.00813285168692 ft`、median6最大RMSEは`0.00786966640898 ft`で、`<=0.02 ft`をPASSした。
- eligible edge fractionは全fold 1.0、eligible edgesは合計1,274,352。
- row identity、well/fold inventory、bilinear basis、forbidden column 0、pre-freeze truth access 0、全runtime guardはPASS。
- query component donor coverageはfold 0--4で`0.986728888295 / 0.979238359970 / 0.979066006568 / 0.969525458029 / 0.995853286990`。pooledは`0.982163697615`で、全query geometry 5,092,255 rows中90,827 rowsがunsupported。
- active component donor coverageはfold 0--4で`0.96 / 0.92 / 0.92 / 0.96 / 0.98`。各fold 50 componentsの一部にdonor constraintがなかった。
- 上記2 coverage guardがexact 1.0を満たさずStage 0 FAIL。事前policyどおりStage 1、lambda選択、solver、truth join、OOF、direct/H512診断は実行していない。solver fitは0。
- branchを閉じ、inference、submission、reserved proposal 2/3、同一OOFでのgrid/halo/adjacency救済は行わない。
- outputはStage 0数値とSHAの実ファイル確認が必要なため`kaggle/output/train_v2/`へ取得した。archive全体ではなく当該kernel outputのみ。

### version 2 artifact SHA

- push config file SHA: `ed25a930d1b96d7a9f599a935bcc012391baa1f4e716829b55e3504d6e6bcebb`
- input manifest logical content SHA: `8c213d886833a759ea193fcb0c5275e18187d37a9cb46e667663a6b86326f7ca`
- identity logical content SHA: `8fcf29d88693da05c2e36df13a96462a8ae0a3df8f07b46c69e19c1408f90d31`
- support logical content SHA: `fa6ed7a88ab4a348130a8d3dd0e26c41df64c50152272c1c64706f0a0925cae5`
- contract / input manifest / identity / support / summary file SHA: `ad04abe...c5c1 / 2ce87c...111c / 121a207...4368 / d2eac9d...3456 / cd76f65...075d`
- Stage 1を実行していないためgrid solution、solver structure/solution、OOF prediction、gzip decompressed SHAは未生成で対象外。

## 再現性メモ

- seed policy: RNGなし。stable sortとSHA256 well inner split。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle private CPU、Stage 0約94.5秒、GPUなし。
- Kaggle kernel id / version: `kentookumura/exp301-gauge-edge-potential-train` version 2、id_no `128007163`。
- input / identity / support SHA: 上記に記録済み。grid / solver / prediction SHAはStage 1未実行のため対象外。
- model manifest: learned modelなし。solverにも到達していない。
- submission SHA: inference/submission未承認のため対象外。
- rerun check: 未実行。確認まではdeterministic anchorではない。

## 次のアクション

1. exp301 branchをclosedのまま維持し、再push、inference、submission、案2/案3を開始しない。
2. 再訪する場合は別実験のgeometry-only component connectivity readoutとして事前設計する。
3. unsupported componentをtruth-freeに安全に連結できる固定contractが得られない場合はphysical potential familyを再開しない。
