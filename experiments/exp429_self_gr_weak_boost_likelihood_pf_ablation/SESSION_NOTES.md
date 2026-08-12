# exp429_self_gr_weak_boost_likelihood_pf_ablation セッションノート

## 目的

`exp223`で支持されたsame-well self-GR weak boostを、exp072互換likelihood-PFの
particle observation likelihoodへ直接組み込む独立ablationを設計する。
`exp091/128`はparticle weightを変更していないため、本実験の結果として扱わない。

## 現在の状態

- Route: `pf_beam`
- 状態: `preflight_technical_fail_v2_pending_user_decision`
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-07-28 scaffold作成

```bash
make new-steering EXP=exp429_self_gr_weak_boost_likelihood_pf_ablation
make new-exp EXP=exp429_self_gr_weak_boost_likelihood_pf_ablation
```

### 2026-07-28 実装

```bash
.venv/bin/python -m py_compile experiments/exp429_self_gr_weak_boost_likelihood_pf_ablation/exp429_self_gr_weak_boost_likelihood_pf_ablation_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp429_self_gr_weak_boost_likelihood_pf_ablation/exp429_self_gr_weak_boost_likelihood_pf_ablation_compact_selfcontained_train.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp429_self_gr_weak_boost_likelihood_pf_ablation/exp429_self_gr_weak_boost_likelihood_pf_ablation_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp429_self_gr_weak_boost_likelihood_pf_ablation/exp429_self_gr_weak_boost_likelihood_pf_ablation_compact_selfcontained_inference.py
EXP429_IMPORT_ONLY=1 .venv/bin/pytest -q experiments/exp429_self_gr_weak_boost_likelihood_pf_ablation/tests/test_exp429_self_gr_weak_boost_likelihood_pf_ablation.py
make validate-exp EXP=exp429_self_gr_weak_boost_likelihood_pf_ablation
```

- `task` executableは環境に無かったため、同等の`make validate-exp`を使用した。
- 専用contract test: `11 passed`
- strict experiment validation: PASS
- Kaggle package / push / run: 未実施
- `make test`は1,373件をcollect中、既存のexp297 / exp301 / exp333 /
  exp336 / exp349がimport時に各自のconfig contract不一致で5件collection errorとなり
  停止した。exp429専用11件は独立実行で全PASSしており、これら5件はexp429変更外。

## 変更点

- steering 3文書とtemplate experiment scaffoldを作成した。
- routeを`pf_beam`、親をexp417、PF controlをexp072、self-GR式参照をexp223に固定した。
- self-GRはparticle log-likelihoodへ
  `0.07 * quality * clip(centered_self_loglik, 0, 1)`として加える。
- primaryはfixed temperature-5、secondaryはarithmetic meanとした。
- technical preflightとfullの実行量、truth-late freeze、SHA、gateを固定した。
- exp223と同じdescriptor / anchor / centered likelihood / quality式をcompact
  self-contained trainへ持ち込み、padded state grid上でboostを先に`[0,1]`へclipした。
- exp072 x1.0 PF kernelのRNG消費順、transition、clamp、ESS resampling、rougheningを
  保ち、observation likelihoodだけへ固定self-GR boostを追加した。
- alpha0 / alpha0.07 technical preflight、deterministic LPT 4 shard、strict merge、
  prediction / surface / schema / manifest SHA freeze、truth-late join、固定gateを実装した。
- full shardはSHA固定済みpreflight summaryのtechnical PASSを必須とし、未承認・
  未完了preflightからの直行をfail closedにした。
- fixed exp209 HMM 50:50 controlは、exp417のarithmetic+HMM
  `10.269696146642758`からscale5 gain `0.18478646708237534`を差し引いた、
  今回のprimary controlと同式のscale5+HMM `10.084909679560383`に固定した。
- target-free固定4 wellsは`24d8997e / e7818f7a / fd710aea / ea41324e`。
  asset SHAは
  `24358da10d2d853b25b4eeb68446c005e34364c78d7f0185af4ceb601effd876`。
- compact train / inferenceを作成し、正規train / inference notebookへ採用した。
  inferenceはraw-test predictionとsubmissionを生成しないfail-closed実装である。
- 同一実験helper importはなく、train sourceは3,121行。参照したparent compactは
  exp400が2,016行、exp417が1,637行で、親のruntime/input/PF/freeze/late join/
  metrics/生成物の章を欠かさず、self-GR/preflight/shard章を追加した。
- Kaggle package / preflight / full runは作成・実行していない。

## 実行量契約

- technical preflight variants / wells / PF well-runs: `2 / 4 / 8`
- preflight seed-well trajectories / particle starts: `1,024 / 512,000`
- full scientific variants / PF well-runs: `1 / 773`
- full seed-well trajectories / particle starts: `98,944 / 49,472,000`
- full shards / merge PF runs: `4 / 0`
- full parent control rerun: `0`
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 再現性メモ

- seed policy: exp072互換
  `stable_seed("likpf", "train", well_id) + seed_index`
- stochastic components: particle初期化、propagation、conditional resampling、
  roughening
- parallel policy: well別stable seed、worker間global RNG共有なし、
  deterministic LPT 4 shards
- CPU/GPU runtime: CPU-only、GPU 0、internet disabled
- Kaggle kernel id / version:
  `kentookumura/exp429-self-gr-weak-boost-likpf-ablation-train / 2`
- Kaggle kernel id_no: `128934717`
- input / feature schema SHA: configに既知control SHAを固定。新規run SHAは未生成
- preflight prediction logical SHA:
  `997713bd08559411135bd48e9a19594fe4141885c08da3fd66b3070e96b009f3`
- preflight surface manifest logical SHA:
  `6c4876f94fe94ec63da95b6b5f270cdc519bc2f50d3a9a992e6200cc46ac0c35`
- model manifest / model SHA: 非該当
- preflight summary file SHA:
  `715ce13b5f918184017678e1087b0ebf5c3607b262ed2f73caa8d1408adf0dd0`
- submission SHA: 非該当
- rerun check: 未実行。初回fullだけではdeterministic anchorと呼ばない

## 次のアクション

1. 現契約のtechnical FAILで閉じるか、alpha0 comparatorを保存exp404 x1.0
   arithmetic predictionへ訂正してversion 3を実行するか、ユーザー判断を得る。
2. version 3を承認された場合もtolerance緩和やPF/self-GR parameter変更は行わない。
3. preflight PASS後もfull 4 shard + mergeは別承認とする。

## 2026-07-28 Kaggle CPU preflight実行承認

- ユーザーの「実行してください」を、直前に提示したKaggle CPU preflightの
  package / push / run承認として記録した。full 4 shard、merge、inference、
  submissionは承認範囲外のまま。
- 実行対象: `alpha0_parity`と`alpha07_candidate`のtechnical variant 2、
  target-free固定4 wells、合計8 PF well-runs。
- 実行量: 1,024 seed-well trajectories、512,000 particle starts。
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- 親full control再実行: 0。alpha0は固定4 wellsだけのkernel parity確認。
- packageには固定4-well assetをbootstrapし、exp072 / exp209 / exp226 /
  exp115 kernel inputとexp404 frozen prediction datasetを接続する。
- Kaggleの50文字制限に合わせ、canonical kernel id/titleでは
  `likelihood-pf`を`likpf`へ短縮し、slugを一致させる。
- canonical private CPU packageを`--run-on-push --strict --no-src`で生成した。
  metadataはGPU/internet無効、competition input、4 kernel inputs、
  exp404 frozen dataset、固定4-well bootstrap assetを含む。
- 2026-07-28 22:39 JSTにcanonical kernel version 1をpushした。
  kernel: `kentookumura/exp429-self-gr-weak-boost-likpf-ablation-train`
  (`id_no=128934717`)。pullによる存在確認、GPU/internet無効を確認済み。
- version 1は約19秒で、固定preflight assetのpath解決前にfail closedした。
  bootstrapは`/kaggle/working/assets/...`へ正しく展開したが、config candidateが
  repo内`experiments/.../assets`だけだったことが原因。PF well-runは0。
- 科学式・variant・seed・particle・gateを変更せず、Kaggle working相対の
  `assets` candidateだけを追加し、同じcanonical kernel idのversion 2で再実行する。
- version 2を同じcanonical kernel idへpushし、preflightを再開した。

## 2026-07-28 Kaggle CPU preflight version 2結果

- runtime to gate: `634.395 sec`
- variants / wells / PF well-runs: `2 / 4 / 8`
- seed-well trajectories / particle starts: `1,024 / 512,000`
- candidate positive-quality rows / positive boost applications:
  `12,239 / 777,858,990`
- technical gate: FAIL
- 唯一のFAIL:
  alpha0対保存exp072 `last_known_tvt + likpf_mean_d` absolute再構成の
  row最大差`0.00035199999911128543 ft`が固定上限`0.00001 ft`を超過。
- 切り分け:
  同じ保存exp404 artifact（raw SHA
  `b3699432a691229da5a6562ce74e0b84f1bee3021bd80d650526906f5aa390f8`）
  の`likpf_mean_x1p0`とalpha0を18,055行・4 wellsで照合し、最大差`0.0 ft`、
  全行bit-exactを確認した。PF replayの実装差ではなく、exp072 deltaからabsoluteへ
  戻す保存表現とのrow-level丸め差が現gateを超えた。
- summary / audit / prediction gzip file SHA:
  `715ce13b5f918184017678e1087b0ebf5c3607b262ed2f73caa8d1408adf0dd0` /
  `d713ad1e3df4df8e81b3347f6f64b2566a3e4f0dd666ed2083ae15143c37a01e` /
  `3f2b8acfa07b027e369fb79ab7e13741dd9b9cab514d163999165f345cf6af2d`
- full runはtechnical PASS必須のため無効化し、`run_preflight/run_full/run_merge`
  をすべてfalseに戻した。gate変更やversion 3は未承認。

## 2026-07-28 Kaggle CPU preflight version 3承認

- ユーザーの`version3を実行してください`により、保存exp404
  `likpf_mean_x1p0`をalpha0 bit-exact comparatorにする訂正と、同じcanonical
  kernel idでのversion 3 preflight実行を承認済み。
- tolerance `1e-5 ft`、self-GR式、PF dynamics、500 particles、128 seeds、
  固定4 wells、technical/scientific gateは変更しない。
- version 3実行量: 2 technical variants / 4 wells / 8 PF runs /
  1,024 seed-well / 512,000 particle starts。
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU /
  parent full control rerunはすべて0。
- full 4 shard、merge、inference、submissionは未承認。

## 2026-07-28 Kaggle CPU preflight version 3 push

- 同じcanonical private CPU kernel
  `kentookumura/exp429-self-gr-weak-boost-likpf-ablation-train`
  （`id_no=128934717`）の現行version 2をpullしてID、title、CPU/offline設定を
  照合した。
- config SHA
  `efe9da65f9b0a3c9ec900c5a06e2dee6d7693fd5f8a4fa54aa6cfc3b42177685`
  を含むself-contained packageを検証し、version 3をpushした。
- 実行状態は`preflight_v3_running`。full 4 shard、merge、inference、
  submissionは引き続き無効・未承認。

## 2026-07-28 Kaggle CPU preflight version 3結果とdebug修正

- version 3は`522.617 sec`で8/8 PF runsを完了後、technical gateでfail closed。
  予測logical SHA、surface logical SHA、activation countsはversion 2と一致した。
- 唯一のFAILは、保存exp404 `likpf_mean_x1p0`との最大差
  `0.00048437499935971573 ft > 1e-5 ft`。
- exp404とexp429はいずれもPF meanを`float32`へcastして凍結する。version 3は
  exp404 CSVの最短10進表現を`float64`で読み、exp429のメモリ上`float32`値と
  比較したため、同じfloat32値のserialization差を誤検出した。
- version 3 predictionと同じraw SHAのexp404 artifactを保存semantic dtype
  `float32`へ復元したposthoc照合は18,055/18,055行bit-exact、最大差`0.0 ft`。
- version 3 summary / audit / prediction gzip file SHA:
  `b28a08566d4a4abc3c635aec5d1014f6c4a7ecfe9384cf83492b5244bc1eec87` /
  `2ac3720e92e57dc19d8da4383e4fac6ad6cefb7855f435752d3e0f9d0a9ef5d4` /
  `3f2b8acfa07b027e369fb79ab7e13741dd9b9cab514d163999165f345cf6af2d`。
- comparatorをconfig固定の`float32`へ復元してからbit比較する最小修正と
  regression testを追加した。12 tests、構文、Ruff F821をPASS。
- version 4は同じ承認済みpreflightのdebug retryとし、2 variants / 4 wells /
  8 PF runs / 1,024 seed-well / 512,000 particle startsを変更しない。
  tolerance、科学式、PF parameter、gate、full/inference/submissionの承認状態も
  変更しない。
- config SHA
  `82b3930aaffb57a8f2be76af9a2328ce57c0e985ad77892882cec5a428dbf55d`
  のself-contained packageを同じcanonical private CPU kernelのversion 4へ
  pushした。状態は`preflight_v4_running`。

## 2026-07-28 Kaggle CPU preflight version 4結果

- canonical private CPU version 4はKaggle status `COMPLETE`。notebook終了ログ時刻は
  `472.457 sec`。
- 2 variants / 4 wells / 8 PF runs / 1,024 seed-well /
  512,000 particle startsを完走し、technical gate `PASS`。
- 保存exp404 comparatorはsemantic dtype `float32`、18,055/18,055行bit-exact、
  最大差`0.0 ft <= 1e-5 ft`。
- candidate positive-quality rows / positive boost applicationsは
  `12,239 / 777,858,990`。version 2/3と一致。
- prediction / surface logical SHAは
  `997713bd08559411135bd48e9a19594fe4141885c08da3fd66b3070e96b009f3` /
  `6c4876f94fe94ec63da95b6b5f270cdc519bc2f50d3a9a992e6200cc46ac0c35`。
- summary / audit / prediction gzip file SHAは
  `2e9f066fd80813862d1e232ad66fd965020e63a3bddd976ef68303e79fe0d190` /
  `eb29bc5506e0b1ddb65cb1a4909fec52939fe2f7d3d35eb9471d41b9fd5fc65a` /
  `3f2b8acfa07b027e369fb79ab7e13741dd9b9cab514d163999165f345cf6af2d`。
- `run_preflight/run_full/run_merge`をfalseへ戻し、
  `preflight_passed_v4_awaiting_full_approval`とした。full 4 shard + merge、
  inference、submissionは未実行・未承認。
- ローカルKaggle packageも完了後configで再生成し、metadata/config双方の
  `run_on_push`をfalseへ戻した。別承認前の再pushでは実行されない。

## 2026-07-29 full 4 shard + merge承認

- ユーザーの`full実行してください`により、technical PASS済みの固定fullを
  4 Kaggle CPU shardで実行し、4生成物のstrict mergeとtrain-side科学gate判定まで
  行うことを承認済み。
- 実行量:
  - scientific variant: `1`
  - candidate PF well-runs: `773`
  - seed-well trajectories: `98,944`
  - particle starts: `49,472,000`
  - reporting folds: `5`
  - LightGBM configs / trained folds / boosters / models: `0 / 0 / 0 / 0`
  - parent PF control / HMM / Beam / GPU reruns: `0 / 0 / 0 / 0`
  - merge PF well-runs: `0`
- deterministic LPT shard inventory:
  - shard 0: `193 wells / 946,128 rows / 12,352,000 particle starts`
  - shard 1: `193 wells / 946,017 rows / 12,352,000 particle starts`
  - shard 2: `193 wells / 946,112 rows / 12,352,000 particle starts`
  - shard 3: `194 wells / 945,732 rows / 12,416,000 particle starts`
- 4 shardは同じsource/scientific contractでshard indexだけを変える。mergeは
  保存済み4 shardをkernel inputとして読み、PFを再実行しない。
- inference、submission、parameter/grid救済は未承認のまま。
- 4 packageをprivate / CPU / internet off / run-on-pushで生成し、共通
  scientific contract SHA
  `b35b215b0d5c363d5c1606e7c1aa2a2d0ff0eabc7bb57c9aced053f7a792adba`、
  bootstrap除外実行code SHA
  `a85318b584ff0abaa2a9e0341e35b9c4ec2b692637a65588b4b0ae7d657259d8`
  の一致を確認した。
- shard 0--3 package config SHA:
  `8059e85b0f74e2f103220cf5a350970678f006a58aa290828e4705229fd737de` /
  `88896d97fd1633be5b445b6ec94f66a48b5c0f873ccc55f1db4fb756c73b6579` /
  `2d89875a87cc4fd34a84125d8da1281a8ac7ea7ffca72e3bf43a3887fcdfd1ca` /
  `57f6d8ec988a66ca7eed1b357b035f960e64d4158159e1309eef53098fc7fe25`。
- push前の直近CPU statusは他実験2本RUNNING、3枠空き。既存実行を停止せず、
  shard 0--2をversion 1としてpushした。
- shard 0 / 1 / 2 id_no:
  `128976012 / 128976015 / 128976019`。push後pullでprivate、CPU、
  GPU/internet off、固定inputを確認した。3本とも`COMPLETE`し、予測行数は
  `946,128 / 946,017 / 946,112`、runtimeは
  `4,782.682 / 4,080.912 / 4,740.168 sec`、particle startsは各
  `12,352,000`、freeze-before-truth監査値は全て0だった。
- shard 0 / 1 / 2 prediction logical SHA:
  `f53fce6aefd05cdd086ac87144e304663d98126edd3fd471478cd73dee2fa8d4` /
  `42b5970007133b105841d4f7e0a22a6d9ef556a557578a9aef7ab1ebdd84a22f` /
  `ffc6c6fd4b1943ce83b087c34cd2918fb46818a2f9bef8e4fd4730b5bc07cafe`。
  3本のprediction schema SHAは
  `619ed754c5c3194be4044b468dfef6e0f0a45fec1e6a59e2bd5822d37c0d5641`
  で一致した。
- CPU枠解放後、shard 3をversion 1としてpushした。id_noは`129040683`。
  push後pullでprivate、CPU、GPU/internet off、固定inputを確認し、
  現在`RUNNING`。
- shard 3待機中にmerge package `kaggle/merge_v1`を生成した。
  private、CPU、GPU/internet off、run-on-push、4 shardを含む9 kernel
  sources、4 ordered shard roots、`run_merge=true`、`run_full=false`を確認した。
  package config SHAは
  `349752d892cc55e07feb4270d69cbe8af1e5cc1d362c39cef2d0d51fdd9ab127`、
  notebook SHAは
  `4bc9d99e999ac460816bc3996ca727752e02d4ea1f556562473d9eae264c1ec0`。
  shard 3が`COMPLETE`するまではpushしない。
- shard 3は`COMPLETE`。`194 wells / 945,732 rows`、runtime
  `2,505.599 sec`、`24,832` seed-well trajectories、`12,416,000`
  particle starts、positive valid rows `495,024`、positive boost
  applications `30,957,074,020`。freeze-before-truth監査値は全て0。
  prediction logical SHAは
  `430b081a9ab0dffc536a24b744255534110ae777d2aa008633b6a15485f8e052`、
  schema SHAは他3本と同じ
  `619ed754c5c3194be4044b468dfef6e0f0a45fec1e6a59e2bd5822d37c0d5641`。
- 4 shard合計は`773 wells / 3,783,989 rows / 98,944`
  seed-well trajectories / `49,472,000` particle starts。positive valid
  rowsは`2,096,654`、positive boost applicationsは`131,102,661,385`。
- merge kernelは`2026-07-29 10:02:48 UTC`にversion 1が開始済みだったため、
  再pushせずpullして照合した。remote/localの24セル連結source SHAはともに
  `d8d952db20edd46fb828083081a57977c79f52f53fd91140375a4cae068f1b1b`
  で一致し、`run_merge=true`、`run_full=false`も一致。既存実行を正として
  監視する。
- merge v1は`merged shard manifest differs from deterministic raw LPT`で
  technical ERROR。4 manifestを必要最小限で取得して比較すると、`773 x 5`
  の値は全て一致し、唯一の差は保存CSV読込時の`shard_index int64`と、
  deterministic LPT側の`int8`だった。科学値、予測、割当の差ではない。
- `read_shard_manifest()`で`shard_index`を契約どおり`int8`復元する
  one-line semantics fixを実装し、CSV roundtrip回帰testを追加した。
  13 tests、Jupytext `--test`、py_compile、Ruff F821、strict validationが
  PASS。取得済み4 manifestでも`773 wells`、shard count
  `193 / 193 / 193 / 194`、5列strict equalityを確認した。
- 正規notebookは上書きせず、修正版Jupytext compact notebookを公式
  bootstrap処理へ渡して`kaggle/merge_v2`を生成した。config SHA
  `65c01a17e28fd813b875b092779c8afd8816daaaa26eb4a6fdd39e5e2f0d81b4`、
  notebook SHA
  `b796d9e95716e101225df0167717994a88c03f563d96e764d263efa02e19f4eb`。
- merge version 2をpushし、push後pullしたremote/localの24セル連結source
  SHAが
  `83da38e6486fde2d13501e4acc72479cff8fd2a2476c77ebc0580bf1e0cd9470`
  で一致し、remoteにint8復元fixがあることを確認。現在`RUNNING`。
- merge version 2は`COMPLETE`。runtime `222.187 sec`、prediction freeze
  `115.685 sec`。`3,783,989 rows / 773 wells / folds 0--4`、finite coverage
  1.0、4 shard runtime上限、実行量、preflight、3 control parity、
  freeze-before-truth、SHA completenessを満たしtechnical gateはPASS。
- primary scale5はcandidate `11.127406421`、control `10.914522073`、
  gain `-0.212884347 ft`で固定`>=+0.05 ft`をFAIL。改善foldは`1/5`で
  固定`>=4/5`をFAIL。fold delta candidate-controlは
  `-0.320141 / +0.284662 / +0.059311 / +0.056554 / +0.768297 ft`。
- arithmetic secondary deltaは`-0.023416828 ft`でPASSしたが、primary
  scopeはhigh-missing以外をFAIL。raw observed `+0.298609`、raw missing
  `+0.026313`、1000+ `+0.245068`、hidden-like spatial `+0.098136`、
  hidden-like typewell-purged `+0.100388 ft`。
- by-well p95 `+0.770627049 ft`（上限0）、worst well
  `+34.862601957 ft`（上限0.25）、fixed HMM/PF 50:50
  `+0.070319209 ft`（上限0）もFAIL。scientific gateはFAILし、
  `terminal_close_without_self_gr_or_pf_rescue_grid`で閉鎖。
- merged prediction logical SHA
  `d7677deb40526274853178290d316efcc0b1bafe629d13c669f50ac062689ff0`、
  surface logical SHA
  `2bfc1a996c4f7ad01a48ef34ba333f56907e5f732c3f599d77cc4f27c58a2ba7`、
  artifact manifest SHA
  `3620944f8ab0c6cf0b85c9fd7c11a9ed07897965324c774e43aa528dedbf694e`。
  小さいmetrics/gate/manifest/log実ファイルを
  `artifacts/kaggle_merge_v2/`へ保存した。
- 同一OOFのalpha/clip/window/top-k/temperature/PF/grid、blend、selector救済、
  inference、submissionは実行しない。direct PF boostは局所的な正例より
  wrong-basin/tail増幅が勝ったと解釈し、既存の低優先
  `self_gr_quality_addonly_features_on_exp092`だけを独立feature-only候補として
  維持する。
