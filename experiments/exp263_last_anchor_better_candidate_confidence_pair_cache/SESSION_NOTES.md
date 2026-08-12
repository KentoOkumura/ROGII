# exp263_last_anchor_better_candidate_confidence_pair_cache セッションノート

## 目的

`KAGGLE_DIRECTION.md`の`last_anchor_better_candidate_confidence_pair_cache`実装契約を、known 33
reference / core 12 primitive / raw-test 6 primitive / 8 pair / w500 alias / 3 named formula / virtual
loaderとして実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0完了・Stage 1値parity完了・namespaced confidence Kaggle v3 parity完了
- CV: 新規学習なし
- LB: v2提出 Public 7.800、v3は同一submissionのため再提出なし
- Stage 0 full cache: version 1完了
- Stage 1: raw-test 6 primitive再生成、固定formula、submissionはv2完了。v3で21 confidence列を実値検証済み

## 実行scope確認

- active variant: 0
- LightGBM config: 0
- fold training: 0
- 合計booster: 0
- parent/control再学習: なし
- PF/Beam再生成: Stage 0ではなし。Stage 1ではexp073 stable per-well seed契約でraw testから再生成する。

## 実装内容

- `candidate_cache_contract.py`
  - reference 33、core 12、raw-test 6、pair 8、w500 alias、3 named formulaを固定。
  - HMM+LGB、selector/TVT outputs、exp104 pair sweep、pair/triple closureをguard。
  - pair/tripleのOOF metricsとfold別outer-crossfit係数をformula manifestへ固定。
- `candidate_cache_builder.py`
  - exp072 canonical ID/well-row/foldを固定し、external sourceを同じspanへjoin。
  - candidate-major / outer-fold Parquet、sparse confidence、outer-train-only eligibility、pair readout、
    SHA manifest、small parity sampleを生成。
  - confidence未提供値は推測せずNaN + missing contract。
  - Stage 1は6 primitiveのidentity/finite guard後、5 pairと固定exp226+w500だけを再構成。
- `candidate_cache_loader.py`
  - primitive / pair / named formulaをfoldとrow sliceでmaterialize。
  -異family confidenceを無理に同一尺度へ集約せずparent別namespaceで返す。
  - w500 aliasとprimitive親の同時selectable、再帰closureを拒否。
- Jupytext percent形式から正規train/inference notebookを生成。
- `loader_contract.md`、`cache_schema.json`、exp263専用回帰テストを追加。
- Parquet backendを明示するため`pyarrow`をproject dependencyへ追加し`uv.lock`を更新。

## コマンドログ

### 2026-07-16 実装

```bash
make new-steering EXP=exp263_last_anchor_better_candidate_confidence_pair_cache
make new-exp EXP=exp263_last_anchor_better_candidate_confidence_pair_cache
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/exp263_last_anchor_better_candidate_confidence_pair_cache_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/exp263_last_anchor_better_candidate_confidence_pair_cache_inference.py
.venv/bin/python -m py_compile experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/*.py
.venv/bin/ruff check experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/*.py experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/tests/test_exp263_candidate_cache_contract.py
.venv/bin/pytest -q experiments/exp263_last_anchor_better_candidate_confidence_pair_cache/tests/test_exp263_candidate_cache_contract.py
```

- `task` executableは環境になかったため、repo規約のfallbackで`make`を使用した。
- ローカルcompetition dataでnotebookは実行していない。
- synthetic 15-row / 5-well sourceでStage 0 builder、60 value partitions、60 confidence partitions、
  fixed formula loaderまでend-to-end検証した。
- 実装時点のexp263 targeted 11 tests / repo全61 tests、`validate-template`、`validate-exp` strictはPASS。
- train/inference packageはcanonical id/title、private CPU、GPU/internet offでstrict生成した。
- `--no-src`で無関係なrepo `src/`を外し、bootstrap 10 filesにconfig、builder、contract、
  loader、`cache_schema.json`、`loader_contract.md`、settings/projectが入ることを確認した。

## Notebook構成確認

- 親exp072はcache sourceであり、この契約に対応するcompact self-contained notebookはない。
- train notebookは10章で、contract、input preflight、confidence inventory、formula DAG、Stage 0生成、
  SHA、virtual parity、metricsをセル上で追える。
- inference notebookは7章で、Stage 0依存、deployability tier、6 source preflight、identity/finite
  guard、5 pair + 1 fixed formula parityを追える。
- cache buildは3,783,989行を扱うheavy feature generationのためhelperへ分離したが、notebookに
  source、scope、禁止事項、実行対象、生成物、検証を展開した。

## 再現性メモ

- seed policy: `not_applicable_deterministic_artifact_transform`
- stochastic components: なし
- CPU/GPU runtime: CPU、GPUなし
- Kaggle kernel id / version: 未実行
- input SHA: Stage 0実行時にraw file SHAとgzip decompressed content SHAを記録
- feature schema/content SHA: 各Parquet partitionのschema/content SHAを記録する実装済み
- model manifest / model SHA: `not_applicable_no_training`
- prediction SHA: `not_applicable_cache_only`
- submission SHA: `not_applicable_no_submission`
- rerun check: Stage 0 v1後に同一sourceでmanifest/content SHAを比較予定

## 未実行事項

- Stage 1 Kaggle inference、submit-check、code competition submit、LB記録

## 2026-07-16 Kaggle Stage 0実行準備

- 実行対象はprivate Kaggle CPU notebook 1本。GPU/internetはoff。
- active variant 0、LightGBM config 0、fold training 0、booster 0、parent/control再学習なし。
- 12 core primitiveの既存sourceを12 Kaggle kernel inputへ固定した。exp072/103/192/209/223/225/
  226/231は各1本、exp243は実装契約のRMSE 12.499353を生成した固定4 shardを使う。
- exp243の後発single-notebook outputはK8 m0 RMSEが12.734466で契約値と異なるため使わない。
- `kaggle kernels files --page-size 200`で12 kernelすべてに必要ファイルがあることを確認した。
- local mirrorでsource resolverが8 source各1 file、exp243 4 filesを一意に解決することを確認した。
- 全external source gzip headerにbuilderが要求するidentity/value/confidence列が存在することを確認した。
- Kaggle明示パスを`data.inputs`へ追加し、同名artifactのglob誤解決を避けた。
- 実験名全体のtrain slugは63文字で、exp226/231で確認済みのKaggle長slug `SaveKernel 400`
  リスクがある。未pushの初回packageから意味を保つ短縮id/title
  `kentookumura/exp263-last-anchor-pair-cache-train` / `exp263 last anchor pair cache train`を使う。
- strict packageをCPU / internet off / run-on-push、12 kernel sources、bootstrap 10 filesで生成した。
- `kentookumura/exp263-last-anchor-pair-cache-train` version 1をpush。kernel id_noは`127474050`。
- push後pullしたKaggle metadataでprivate、GPU/TPU/internet off、12 kernel sourcesを確認した。
- push直後の状態は`KernelWorkerStatus.RUNNING`。同じslugを完了まで監視する。

## 2026-07-16 Kaggle Stage 0 version 1完了監査

- user completion連絡後、CLI statusで`KernelWorkerStatus.COMPLETE`を確認した。
- Kaggle logsはbootstrap 10 files、実験/route/output、0 variant/config/fold/booster、
  kernel開始981.749秒でのmetrics保存を示した。error tracebackはない。
- Stage 0 builder runtimeは951.444秒。3,783,989 rows / 773 wells / 5 foldsを処理した。
- 12 primitive × 5 foldsでvalue 60、confidence 60、合計120 Parquet partitionを生成した。
  valueは2,665,368,732 bytes、confidenceは515,849,184 bytes。
- sourceは9 groups / 12 gzip files。全core candidateのrows loadedは3,783,989でcoverage 100%。
- cache manifest SHAは`85e60ac1...a26bb9e`、catalog SHAは`7cd74866...e9e6e0`、canonical ID
  SHAは`de07df32...4ae9d3`、generation config SHAは`3cf69b76...0451b`。
- best fixed pair readoutはexp226 + self-GR HMM a070 50/50、RMSE 8.532715037、
  better parent比-0.894394637、481 wells改善 / 292悪化。
- outer-train-only eligibilityは100 candidate×fold中99件true。`beam_mean:fold1`のみfalse。
- Kaggle output 143 files、`part-000.parquet` 120件をCLIで確認した。全archiveは取得せず、
  manifest/catalog/readout/eligibility/named formula/parity/metrics/summaryを限定取得した。
- 代表としてbest pair両親のfold 0 value/confidence 4 Parquetだけを取得し、rows、bytes、file SHA、
  logical content SHA、schema SHAがmanifestと一致することを確認した。
- 同4 Parquetからvirtual loaderでbest pair 757,738行を再構成し、直接50/50平均との差は最大0。
- Kaggle pandas 2系`object`とlocal pandas 3系`str`のdtype推論差でlogical hash helperがそのままでは
  local再検証できなかった。hash前のstring→object正規化を追加すると既存manifest値へ一致したため、
  loaderを補強してParquet round-trip回帰テストを追加した。cache再実行は不要。
- Stage 0は後続OOF cacheのcanonical inputとして採用する。Stage 1は6 current-test sourceと
  identity/finite/SHA/formula parityが揃うまで停止する。
- hash正規化回帰追加後はexp263 targeted 12 tests / repo全62 tests、ruff、py_compile、Jupytext、
  `validate-template`、`validate-exp` strictをすべてPASSした。

## 次のアクション

1. Stage 1 current-test source 6本を解決する。
2. 6 primitive / 5 pair / fixed `exp226_w500_50_50`のID/SHA/formula parityを確認する。
3. parity後にfixed blendまたはselectorの別実験へ進むか判断する。

## 2026-07-16 Stage 1 hidden-safe inference / submit準備

- ユーザーがtrain-sideで最も良い提出可能なcandidate/combinationを推論し、提出まで行うことを明示した。
- scope内diagnostic最良`exp226 + exp192 likPF + exact HMM` cross-fit 8.209225はexp192がtrain-onlyなので除外した。
- raw-test componentsのcross-fit 8.231651はfold別fit係数をtestへ一意に固定できずdiagnostic-onlyなので除外した。
- 提出対象は追加fit不要、5/5 folds改善、全成分raw-test再生成可能な固定
  `exp226_w500_50_50 = 0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`、OOF RMSE 8.238331。
- Stage 1 inferenceは静的exp237 current-test predictionを提出入力にしない。raw competition testから以下を再生成する。
  - exp073 deterministic replay: `likpf_mean` / `pf_ancc` / `beam_mean`
  - exp209 exact HMM: `exact_hmm`
  - exp223 self-GR HMM: `selfgr_hmm_a070`
  - exp226 deterministic K16: `exp226_k16`
- exp237 current-test candidate frameはpublic current-test時の数値parity referenceだけに使用し、hidden ID集合と異なる場合は比較だけskipする。
- planned Kaggle inference workload: active variant 0、LightGBM config 0、fold training 0、booster 0、parent/control再学習0。CPU、GPU/internet off。
- inference kernel sourcesはexp263 Stage 0、exp073 inference、exp209 train、exp223 train、exp226 inference、exp237 inferenceの6本。
- 出力は6 primitive、5 pair、固定formula Parquet、`submission.csv`、primitive/prediction/submission SHA。competition submitはsubmit-check PASS後に1件だけ行う。

### Kaggle inference v1 failure

- kernel: `kentookumura/exp263-last-anchor-pair-cache-inference` v1 / id_no `127480072`、CPU、internet off、6 kernel sources。
- exp073 replayは約134秒までに完了し、K16 full-train fit/test 3 wellsは約176秒、exact/self-GR HMMを含む6 primitive生成は約364秒まで進んだ。
- failureは候補生成後のformula parity guard。float32で保存した約10,000ftの固定formulaをfloat64で再計算し、絶対`1e-5`を要求したため、通常のfloat32丸めを誤ってfailure扱いした。
- 修正は固定weightsと同じfloat32演算順でdirect formulaを再計算しbitwise parityを見るもの。candidate値、source、weights、PF seeds/particles、HMM/K16設定は変更しない。
- v1は`submission.csv`生成前に停止しており、competition submitは0件。

## 2026-07-16 Stage 1 namespaced confidence拡張

- 既存Stage 1 inference v2のlocal mirrorは14,151 rows / 3 wellsを225.459秒で完走し、6 primitiveの
  exp237値parityは全候補最大0.000484375以内、fixed formula parity最大0だった。
- exp264 Stage A採用100特徴のうち21列がsource-nativeまたはformula親confidenceに依存したため、
  `current_test_formula_parity.parquet`へ21列を`confidence__<primitive>__<field>`で追加した。
- HMMは同一`run_hmm2()`の`std_eval` / well-level `loglik` / self-GR診断を使い、
  `loglik_per_row=source_loglik/unknown_rows_in_well`をOOFと同じ規則で派生する。
- exp226は別proxyを作らず、同一`predict_well()`の`PredictionResult.delta`を
  `geometry_gr_delta`として候補値と同時取得する。このため旧`run_inference()`を二重実行せず、K16 fit/predictは1回だけ。
- PF-ANCCは`pf_ancc_std`、Beamは`beam_std_d`を使う。likPFはnative scalarなしとして
  `confidence_valid=False`を明示し、0や疑似sigmaを作らない。
- formula confidenceは平均せずprimitive namespaceだけを保存し、exp264がparent別に決定展開する。
- config、loader contract、steering、Jupytext inference notebook、unit testを同期した。targeted exp263/264は
  19 targeted tests / repo全69 tests PASS、ruff / py_compile PASS。competition dataでのローカルnotebook実行は行っていない。
- 次は同じexp263 inference kernelをCPU / GPU・internet off / 0 boosterで再実行し、21列のcoverage、
  content SHA、既存候補値parity不変を確認する。

## 2026-07-17 Stage 1 fixed blend scoring完了

- inference kernel `kentookumura/exp263-last-anchor-pair-cache-inference` v2は14,151 rows / 3 wellsを
  225.459秒で完走した。6 primitive / 5 pair / fixed formulaをraw testから生成し、formula parity最大0、
  exp237 current-test referenceとの差は6 primitiveすべて最大0.000484375で許容0.001以内だった。
- 提出候補は`exp226_w500_50_50 = 0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`、
  OOF RMSE 8.238331。submission SHAは
  `6316695197ee67c9a2aaa23754e6f2a5cf30dd0ec4ef1a018921f9ea640a1dbc`、submit-check PASS。
- code submission ref `54761954`、提出時刻`2026-07-16 13:49:16.687 UTC`、status `COMPLETE`、
  Public LB **7.800**、Private LB未表示。
- Public LBはOOFより-0.438331、exp226単体9.837より-2.037、exp218 7.843より-0.043改善した。
  一方、exp257 ML submitted anchor 7.718より+0.082、exp082 ensemble anchor 7.601より+0.199悪い。
- fixed blendのhidden-safe生成と補完性は採用するが、全体anchorは更新しない。LBを見たweight grid、
  係数再fit、追加submitは行わない。21 namespaced confidence列の再実行はこの提出結果と分離して継続する。

## 2026-07-17 Stage 1 namespaced confidence Kaggle v3実行開始

- 最新inferenceはactive variant / LightGBM config / fold training / boosterが`0 / 0 / 0 / 0`、
  parent/control再学習なし、CPU、GPU/internet off。競技への再提出はscope外。
- exp263/264 targeted tests 19件、ruff、py_compile、Jupytext test、`validate-exp` strictをPASSした。
- canonical packageを同じkernel idで再生成し、21列の`confidence__<candidate_id>__<field>`、最新config、
  `run_on_push=true`を確認した。package notebook SHAは
  `623c68280ca996194cf9d015678f28cf42c9474f4cf8a69a796d4283bb5dceda`。
- `kentookumura/exp263-last-anchor-pair-cache-inference` version 3をpushし、`RUNNING`を確認した。
  ユーザー指示により約6分時点でローカル監視を停止した。Kaggle側の実行は継続し、competition submitは0件。
- 完了連絡後にlogsと必要成果物だけを取得し、21 confidence列のcoverage/content SHA、既存6 primitive / 5 pair /
  fixed formula / submission SHAのv2 parityを確認する。

## 2026-07-17 Stage 1 namespaced confidence Kaggle v3完了監査

- CLIと成果物で`kentookumura/exp263-last-anchor-pair-cache-inference` version 3の
  `KernelWorkerStatus.COMPLETE`を確認した。14,151 rows / 3 wells、runtime 354.341秒、CPU、
  GPU/internet off、0 boosterで完走し、tracebackはなかった。
- `current_test_formula_parity.parquet`は14,151 rows × 36 columns。v2の15列を同じ順序で保持し、
  21列の`confidence__<primitive>__<field>`を追加した。旧15列はDataFrame exact equality、全数値列の
  最大絶対差0だった。
- 21 confidence列は全行non-nullかつfinite。`confidence_valid`はexp226/self-GR HMM/exact HMM/
  PF-ANCC/Beamで1.0、native scalarを持たないlikPFだけ契約どおり0.0だった。
- formula parity最大0、exp237 referenceとの差は全6 primitiveで最大0.000484375、許容0.001以内。
  拡張Parquet SHAは`bda0502894d6a20cc3c332d729cf120b17ceed2e1773093bd7140c6df71e360c`。
- prediction content SHAは`a418876d319301702cc6c3e28b0d30e95518510ef9c83823197c4ecff2e3ce4b`、
  submission SHAは`6316695197ee67c9a2aaa23754e6f2a5cf30dd0ec4ef1a018921f9ea640a1dbc`でv2と一致。
  CSVもbyte-identicalで、Kaggle出力に対する`submit-check`はPASSした。
- v3の目的はconfidence artifact parityだけであり、competition submitは0件。exp263のStage 1契約を
  完了とし、exp264 inferenceのconfidence前提は解消した。exp264 Stage Bの学習実行は別承認のまま。
