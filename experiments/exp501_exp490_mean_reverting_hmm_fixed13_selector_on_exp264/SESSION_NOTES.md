# exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264 セッションノート

## 目的

exp490 mean-reverting HMMを、修正版exp264 fixed12 dual selectorの13番目のscore candidateへ
追加する実験について、固定済みの候補・特徴・split・gate・実行量・禁止事項どおりに
compact Stage A/C候補と契約テストを実装する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU version 2 COMPLETE、scientific FAIL、terminal close
- CV / Public LB / Private LB: hard OOF 8.264890209 / なし / なし
- scientific variant: 1
- score / primary / fixed fallback candidates: 13 / 12 / 7
- 実装・実行時のselector: 2 objectives、outer 5 × inner 4、40 CPU boosters
- trained selector / parent control / HMM / PF / Beam / GPU: 40 / 0 / 0 / 0 / 0 / 0
- current-test / downstream TVT / inference / submission: 0 / 0 / 0 / 0
- decision: `FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`

## コマンドログ

- 2026-08-01:
  `make new-steering EXP=exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`で
  steering scaffoldを作成した。
- 2026-08-01:
  `make new-exp EXP=exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`で
  空の実験scaffoldを作成した。
- 2026-08-01:
  exp264 / exp496 / exp490 / exp499、`docs/06_reproducibility.md`、現行backlogを読み、
  保存入力、13候補契約、phase分離、40 booster、全AND gate、禁止救済を確定した。
- 2026-08-01:
  `make update-summary`で`experiment_summary.md`へexp501のnode、lineage、design-only行を登録した。
- 2026-08-01:
  `make validate-exp EXP=exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`を実行し、
  strict validation PASSを確認した。experiment-doc reviewerでも中核証拠カテゴリを確認した。
- 2026-08-01: ユーザーの`exp501を実装してください`を実装承認として受け、
  `candidate_contract.yaml`、`feature_contract.yaml`、`output_contract.md`、
  `experiments/exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264/exp490_fixed13_candidate_cache.py`、compact train / fail-closed inference候補を実装した。
- 2026-08-01: exp490 loaderを`well,row_idx,suffix_offset,prediction,2 confidence`の
  allowlistに限定し、gzip raw/decompressed SHA、global key、suffix sequence、finite、source-fold
  非読込を実装した。`id`はallowlist keyから機械生成し、selector cacheでもexp263 global keyと
  suffix offsetをfold別に照合する。
- 2026-08-01: Jupytextで別名compact `.ipynb`へ変換した。正規train/inference Notebookは
  placeholderのままで上書きしていない。
- 2026-08-01: 専用test `10 passed`、exp264 / exp496を含む関連回帰test `37 passed`、
  `py_compile`、Ruff、train/inference Jupytext roundtrip、`validate-exp` strictをPASSした。
- 2026-08-01: 保存済み315 MB exp490 full OOFに対する追加の全量loader readbackは、
  ローカルで5分を超えたため生成・学習を行わず手動停止した。この試行を静的PASS根拠には
  含めない。入力raw/decompressed SHAは保存済みexp490 manifestとconfigで固定済みであり、
  実行時には同loaderが必須検証する。
- Kaggle package/push/run、output取得、current-test、downstream TVT、inference、submissionは
  行っていない。

## 変更点

exp264 fixed12 score bankへ次の1候補だけを追加する。

- candidate ID: `exp490_geometry_mean_reverting_hmm`
- prediction: `geometry_mean_reverting_hmm`
- native target-free fields:
  `geometry_mean_reverting_delta_mean`、`geometry_mean_reverting_hmm_std`
- exp490入力: 3,783,989 rows / 773 wells
- raw gzip SHA:
  `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c`
- decompressed content SHA:
  `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`

親12候補、fixed7 fallback、2 objectives、fold、sampling、LightGBM、scope、thresholdは固定する。
exp498/499 feature、truth/error/role/episode/outcome、candidate追加後の救済調整は使用しない。

## 実行前契約

- active variants: 1
- LightGBM objectives: 2
- outer folds: 5
- inner folds: 4
- total selector boosters: 40
- compact partitions / rows: 25 / 18,919,945
- outer-valid candidate-long rows: 49,191,857
- parent/control retraining: 0
- HMM / PF / Beam / GPU: 0 / 0 / 0 / 0
- downstream TVT: 0
- technical / leakage / score / integration / tailを全AND判定する。
- FAIL時は`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`で閉じる。

2026-08-01のユーザー指示「実行してください」により、この固定スコープの
Kaggle CPU Stage A/C実行が承認された。

## 再現性メモ

- seed policy: exp264互換の固定seed 42。
- stochastic components: 実装済みCPU LightGBMだけ。exp490 HMMは保存生成物を使い再生成しない。
- parallel RNG: LightGBMの固定`n_jobs=8`、`deterministic=true`、`force_col_wise=true`。
- CPU/GPU runtime: Kaggle private CPU、GPU/TPU/internet off。version 2を`7082.113 sec`で完走。
- input SHA: exp490 raw gzip/decompressed、exp263 manifest/catalog、exp264 score、hidden-likeを一致確認。
- feature schema / model manifest / compact manifest / outer-valid score SHA:
  `2eb780b9...63e96e` / `3adb894d...cabbc` / `32317a71...9c257` / `1641b9cb...1599e`。
- submission SHA: submissionを作らないため非該当。
- rerun check: 未実施。scientific FAILのため再学習しない。
- deterministic anchor: いいえ。train OOFでtail FAILし、submissionへ進めない。

## 戦略上の位置づけ

- Phase: Late。
- 優先度: P3、CPU selector。現行P1/P2を追い越さない。
- exp490の平均改善は強いがtail FAIL、exp499 well router FAIL、複数fixed13実験の
  incumbent reranking不安定性がnegative warningである。
- exp500と同じP3だが、本実験は保存OOFだけを使う40-booster branchであり、PF再実行はない。
- pooled/fold/scopeは改善したがtail FAILのためroute anchor、LB、提出候補を更新しない。

## Notebook構成比較

- exp496 compact train: 9章 / 647行。
- exp501 compact train: 9章 / 624行。
- exp501はexp496のauthorization、input checks、Stage A、Stage C、scientific/reranking、
  feature importance、reproducibility summaryを全て保持し、exp486の二入力ledger/freeze処理を
  exp490の単一allowlist入力とsuffix-offset parityへ置換した。
- 同一exp helper import、source-file位置依存、薄い`main()`呼出しはない。

## 次のアクション

same-OOF rescue、current-test、downstream TVT、inference、submissionへ進まずterminal closeする。
原因確認は新規predictionを作らないcross-fixed13 reranking/tail readoutだけを独立P4で検討する。

## 実行承認と正規Notebook採用

- 2026-08-01: ユーザーの「実行してください」で、正規train Notebook採用と
  Kaggle Stage A/C runの承認を受けた。承認scopeは`1 variant / 2 objectives /
  outer 5 / inner 4 / 40 CPU selector boosters / parent-control retraining 0 /
  candidate HMM-PF-Beam rerun 0 / GPU 0 / downstream TVT 0 / inference 0 /
  submission 0`。保存済み親結果だけを比較対象に使い、controlを再学習しない。
- 2026-08-01: compact train `.ipynb`を正規train Notebookへ採用した。exp496で
  54文字slugがKaggle `SaveKernel 400`になった既知履歴を踏まえ、科学契約を変えず
  canonical id/titleを`kentookumura/exp501-exp490-fixed13-selector-train` /
  `exp501 exp490 fixed13 selector train`へ揃えた。
- 2026-08-01: canonical trainを`run_on_push=true`、private、CPU、
  GPU/TPU/internet falseでstrict package化した。dataset sourceはexp490 full merge、
  kernel sourceはexp263/264だけ。bootstrapは38 filesでexp490 fixed13 helper、
  exp251 schema、hidden-like assignmentを含む。package notebook / metadata / embedded
  support zip SHAは`65dc2bd9ac9273cb3b49cac620ffd16a70b4534a53d116ac26dc85035078b838` /
  `d6dd714746a422e321b832a238830c96eeacf081a34af1ee6bd1211f546ddb82` /
  `06e6f1e55b33618594c2e0446317772b4de7ae48b499367cb134405d110c74eb`。
  専用+exp496回帰20 testsとstrict validationを再度PASSした。
- 2026-08-01: 初回pushでversion 1（id_no `129379922`）は作成されたが、Kaggleが
  `kentookumura/exp490-mean-revert-full-merge`をdataset sourceとして拒否し、実際の
  metadataはdataset sourceなし、exp263/264 kernel sourceだけになった。exp490は
  COMPLETE済みNotebook outputであり対象prediction fileを持つことを`kernels files`
  で確認したため、科学設定・入力SHA・実行量を変えず、exp490をkernel sourceへ訂正する。
  version 1は`31.444 sec`で`ERROR`、入力解決前停止、trained booster 0だったため、
  有効な科学実行として数えない。訂正後version 2 packageのnotebook / metadata /
  embedded support zip SHAは`49e82f1b3e823101e65ccb98c390492b59886eb42c35a6e1e971ab9722472acd` /
  `31d6fafbf7c40bc7acfe3677b30f9586e5c269f1667e43ccce6a5484fd3557af` /
  `8bb20c89d578063ce480578731edf49725b5881d233040d22f148a636c0f7c4d`。
- 2026-08-01 12:43 UTC: 訂正packageをversion 2としてpushした。id_noは
  `129379922`、statusは`RUNNING`。push後metadataはprivate、CPU、
  GPU/TPU/internet false、exp490/exp263/exp264の3 kernel sources、competition
  sourceを確認した。同じversionを再pushせずterminal statusまで監視する。

## Kaggle Stage A/C完了

- 2026-08-01: version 2は`7082.113 sec`で`COMPLETE`。40/40 models、
  25 compact partitions / 18,919,945 rows、49,191,857 outer-valid score-long rows、
  technical checks 10/10、leakage audit、selector score guardを全PASSした。
- Stage Aは155候補featureから92列を固定し、compact metaは契約どおり77列。
  exp490 3,783,989 rows / 773 wells、raw/decompressed SHA、global key、suffix-offset、
  exp263 fold repartition、finite 100%、forbidden column read 0を確認した。
- fixed13 hard OOFはparent fixed12 `8.652531955610227`から
  `8.264890208588357`へ`0.387641747022 ft`改善。全5 foldsと固定7 scopesを改善し、
  exp490 top1は2,093,883 rows / `55.335335%`、全5 foldsで正だった。
- by-wellは493改善 / 280悪化、delta p95 `+2.904593926 ft`、worst
  `896d15b9 +18.394664149 ft`で、固定`+0.25 ft`上限を両方FAIL。
  decision=`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`。pooled / fold /
  scope改善でtail FAILを救済せず、downstream TVT、current-test、inference、submissionへ進まない。
- post-freeze診断はH512 / whole-well oracle headroom `0.272805 / 0.355756 ft`、
  exp490非top1行のincumbent change率`35.007153%`、usage-delta Pearson / Spearman
  `-0.172649 / -0.203881`。利用0は2 wellsで1改善 / 1悪化だった。
- 小型metrics / manifest / scope / usage / by-well / diagnosticと完全logsだけを
  `kaggle/output/train_v2_selected/`へ取得し、各SHAをKaggle summaryと照合した。
  大容量score parquet / compact partitionsは取得していない。
