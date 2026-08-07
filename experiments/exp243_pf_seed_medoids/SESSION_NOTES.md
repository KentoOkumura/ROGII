# exp243_pf_seed_medoids セッションノート

## 目的

`KAGGLE_DIRECTION.md` の高優先度backlog `pf_seed_medoids` を実装する。exp072互換
likelihood-PFの128 seed trajectoryをtarget-free距離でcluster化し、実在trajectoryである
medoidをcandidate pathとして残す。

## 現在の状態

- Route: `pf_beam`
- 状態: v3 full exact parity PASS。candidate bankとして支持、direct replacementは不採用で完了
- train-side値: saved/replay `likpf_mean` RMSE 11.594898、最良medoid単体12.296667
- inference / submission: 無効

## 2026-07-13 実装

- steering: `.steering/20260713-exp243-pf-seed-medoids/`
- 親: `exp072_exp063_full_replay_feature_cache`
- 実装参照: exp241のraw train / pseudo-tail path契約。
- exp072の`sha256("likpf::train::<well>") % 2147483647 + 1 + seed_index`を再現するPF kernelを実装した。
- PFはraw GR Gaussian likelihood、500 particles、128 seeds、classic transition / resamplingを固定した。
- distanceは承認済みのtail前半1.0 / 後半1.5 weighted trajectory RMSE。
- clusteringは決定的BUILD初期化 + best-improvement PAM、K=3/5/8。
- candidate slotはcluster seed mass降順、likelihood mass降順、seed index昇順で並べる。
- 全seed trajectoryはwell処理中だけ保持し、medoid row candidateとcluster membershipへ縮約する。
- exp237 base8をcache deltaから再構成し、base8+medoid unionのrow/block/whole-well oracleを実装した。
- unique-best、cluster mass、likelihood mass、entropy/HHI、within/between distance、1000+、hidden-like、by-wellを保存する。
- inference notebookはdisabled contractのみとした。

## Kaggle train push前のコスト確認

- active variant: 1 (`pf_seed_medoids`)
- PF replay: 1/well
- particles × seeds: 500 × 128
- K-medoids postprocess: 3 (`K=3/5/8`、同じPF replayを使用)
- LightGBM configs: 0
- folds: 0
- total boosters: 0
- parent/control retraining: なし
- GPU: なし、Kaggle CPU 4 deterministic well shards
- raw-test inference / competition submit: 0

実装完了はKaggle CPU full auditの実行承認を含まない。push前にユーザー判断を得る。

## 再現性

- seed: exp072 stable SHA256 modulo + 1 per well + seed index。
- dtype: seed trajectory、replay mean、K-medoids入力までfloat64を維持し、保存列だけfloat32へ変換する。
- global/shared parallel RNG: なし。Numba single worker。
- input cache/schema、row candidate、cluster manifest、summaryのSHAを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- saved exp072 `likpf_mean`とのreplay parityを測定し、exactと仮定しない。
- model / manifest / submissionは生成しない。

## 静的確認

- `py_compile`: PASS（helper / train / inference）。
- `ruff --select F821,E9`: PASS。
- synthetic 3-mode / 6-trajectory、K=3でmedoid `[0,2,4]`、cluster size `[2,2,2]`、再実行一致: PASS。
- Jupytext train / inference変換と`--to ipynb --test`: PASS。
- `make validate-exp EXP=exp243_pf_seed_medoids`: strict PASS。
- canonical train package `kentookumura/exp243-pf-seed-medoids-train`をprepare済み。metadataはCPU、internet off、run-on-push、kernel source 3本。
- generated bootstrap内configはparticles 500、seeds 128、K 3/5/8、distance weight 1.0/1.5、LightGBM/fold/booster 0、4 shards / active shard 0と一致。
- shard 0/1/2/3のJupytext wrapperを追加し、すべて変換・`--test`・py_compile・ruffを通した。
- shard 0/1/2/3 packageをcanonical slug `kentookumura/exp243-pf-seed-medoids-shard{0,1,2,3}`でprepareした。shard 0でwrapperが`EXP243_ACTIVE_WELL_SHARD_INDEX=0`を設定しcanonical trainを実行すること、bootstrapにcanonical train/helper/configが入ることを確認した。pushは未実行。
- parentにcompact self-contained notebookはなく、exp241正規train notebookと同様に重いNumba PFをhelperへ残し、notebookで設定・入力・実行・評価・生成物を展開した。

## 2026-07-13 Kaggle CPU実行承認

- ユーザーがKaggleでの実行を明示承認した。
- push対象はshard 0/1/2/3の4 private CPU notebooks。
- 各wellのPF replayは1本、500 particles × 128 seeds。K=3/5/8は同一trajectory matrixの決定的後処理。
- LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、competition submit 0。

## 2026-07-13 Kaggle push

- push直前に既存kernelを確認し、exp242とexp241 shard 0/2/3は`COMPLETE`、exp240 v2は`RUNNING`だった。
- shard 0: `kentookumura/exp243-pf-seed-medoids-shard0` version 1をpush。Kaggle statusは`KernelWorkerStatus.RUNNING`。
- URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-shard0
- shard 1/2/3: canonical slugへのpushは`Maximum batch CPU session count of 5 reached`で拒否された。
- shard 1だけ再確認したが`Notebook not found`。shard 1/2/3のstatusは404で、canonical notebookが未作成であることを確認した。
- CPU枠上限を回避するrecovery slugは作らない。枠解放後、同じcanonical slugへpushする。
- inference / submissionは実行していない。

## 2026-07-13 exp072 replay parity修正

- shard 0 v1はユーザーが停止した。
- v1実装ではexp072 `stable_seed()`末尾の`+1`が欠け、seed bankが1つずれていた。
- v1実装ではPF内部のfloat64 trajectoryを関数return時にfloat32化し、その後にreplay meanを計算していた。exp072はfloat64でmeanを計算し、保存時だけfloat32へ変換する。
- seed baseを`sha256 modulo + 1`へ修正し、trajectoryをreplay mean / K-medoids計算までfloat64で維持するよう修正した。
- shard 1/2/3は変更後もpushしない。修正版はshard 0だけ同じcanonical kernelへversion追加で再実行する。
- 修正後の最小smokeで独立計算したseed値との一致、trajectory dtype `float64`、mean dtype `float64`を確認した。
- `py_compile`、`ruff --select F821,E9`、train / shard 0 Jupytext `--test`、strict experiment validationをPASS。
- shard 0 packageを同じcanonical slugで再生成し、通常コピーとbootstrap ZIP内の両方でseed `+1`、float64 return、config dtype契約を確認した。
- v2 push前コスト: active variant 1、PF replay 1/well、500 particles × 128 seeds、K-medoids postprocess 3、LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、submit 0。
- push前pullでcanonical kernel id `126934265`の存在を確認。statusは`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。
- 修正版を同じcanonical kernelへversion 2としてpushした。Kaggle statusは`KernelWorkerStatus.RUNNING`。
- push後pullしたv2 notebookのbootstrap ZIPでもseed `+1`、float64 trajectory return、config dtype契約を確認した。kernel id_noはv1と同じ`126934265`。
- canonical trainとshard 1/2/3のローカルpackageも修正版bootstrapへ再生成したが、Kaggleへpushしたのはshard 0だけ。

## 2026-07-13 shard 1実行承認

- ユーザーがshard 1のKaggle実行を明示指示したため、shard 0 parity確定前の待機方針をshard 1に限って解除する。
- push対象: `kentookumura/exp243-pf-seed-medoids-shard1`のみ。shard 0/2/3には操作しない。
- push前status: shard 0 v2は`RUNNING`、shard 1は404で未作成。
- package: active shard index 1、private CPU、internet off、run-on-push、修正版seed `modulo + 1`、trajectory / mean / K-medoidsまでfloat64。
- コスト: active variant 1、PF replay 1/well、500 particles × 128 seeds、K-medoids postprocess 3、LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、submit 0。
- canonical slugへの初回pushは`Notebook not found`で拒否された。事前statusは404、失敗後pullは500で、実行kernelは作成されていない。
- 過去のCPU上限失敗でcanonical slugがKaggle側の不完全状態になったと判断し、recovery slugを`kentookumura/exp243-pf-seed-medoids-shard1-v1`に限定してpackageを再生成した。
- recovery kernel version 1をpush。URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-shard1-v1
- push後pull成功、kernel id_no `126942881`、status `KernelWorkerStatus.RUNNING`。
- pullしたKaggle notebook bootstrapでactive shard 1、seed `modulo + 1`、float64 trajectory returnを確認した。
- shard 0/2/3にはpush・停止などの操作を行っていない。

## 2026-07-13 shard 2実行承認

- ユーザーがshard 2のKaggle実行を明示指示した。
- push対象: `kentookumura/exp243-pf-seed-medoids-shard2`のみ。shard 0/1/3には操作しない。
- push前status: shard 0 v2とshard 1 recovery v1は`RUNNING`、shard 2は404で未作成。
- package: active shard index 2、private CPU、internet off、run-on-push、修正版seed `modulo + 1`、trajectory / mean / K-medoidsまでfloat64。
- コスト: active variant 1、PF replay 1/well、500 particles × 128 seeds、K-medoids postprocess 3、LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、submit 0。
- canonical slugへの初回pushは`Notebook not found`で拒否された。事前statusは404、失敗後pullは500で、実行kernelは作成されていない。
- 過去のCPU上限失敗でcanonical slugがKaggle側の不完全状態になったと判断し、recovery slugを`kentookumura/exp243-pf-seed-medoids-shard2-v1`に限定してpackageを再生成した。
- recovery kernel version 1をpush。URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-shard2-v1
- push後pull成功、kernel id_no `126947222`、status `KernelWorkerStatus.RUNNING`。
- pullしたKaggle notebook bootstrapでactive shard 2、seed `modulo + 1`、float64 trajectory returnを確認した。
- shard 0/1/3にはpush・停止などの操作を行っていない。

## 2026-07-14 shard 3実行承認

- ユーザーがshard 3のKaggle実行を明示指示した。
- push対象: `kentookumura/exp243-pf-seed-medoids-shard3`のみ。shard 0/1/2には操作しない。
- push前status: shard 0 v2、shard 1/2 recovery v1は`COMPLETE`、shard 3は404で未作成。
- package: active shard index 3、private CPU、internet off、run-on-push、修正版seed `modulo + 1`、trajectory / mean / K-medoidsまでfloat64。
- コスト: active variant 1、PF replay 1/well、500 particles × 128 seeds、K-medoids postprocess 3、LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、submit 0。
- canonical slugへの初回pushは`Notebook not found`で拒否された。事前statusは404、失敗後pullは500で、実行kernelは作成されていない。
- 過去のCPU上限失敗でcanonical slugがKaggle側の不完全状態になったと判断し、recovery slugを`kentookumura/exp243-pf-seed-medoids-shard3-v1`に限定してpackageを再生成した。
- recovery kernel version 1をpush。URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-shard3-v1
- push後pull成功、kernel id_no `126987915`、status `KernelWorkerStatus.RUNNING`。
- pullしたKaggle notebook bootstrapでactive shard 3、seed `modulo + 1`、float64 trajectory returnを確認した。
- shard 0/1/2にはpush・停止などの操作を行っていない。

## 当時の次のアクション

この時点ではfull rerunのCPUコストを提示し、明示承認後にのみ行う方針とした。

## 2026-07-14 4 shard完了・strict merge監査

- shard 0 v2、shard 1/2/3 recovery v1はいずれも`KernelWorkerStatus.COMPLETE`。
- runtimeは9,106.137 / 9,663.959 / 5,857.999 / 8,430.196秒。
- shard別rows/wellsは911,800/188、984,162/196、960,908/193、927,119/196。
- 全体は3,783,989 rows / 773 wells、全well status `ok`。ID unionとwell unionはいずれも重複0。
- replay parityの全体差はmean absolute 0.373289、RMSE 0.743077、最大9.847657。
  `1e-6`超は3,714,240行でexact parity false。
- schema SHA `700d3814...b8`は全shard一致。一方、validation source decompressed SHAは
  shard 0/1の`0503de05...536`とshard 2/3の`99a3c70a...350`に分かれた。
- よってID/well gateはPASS、replay parityとinput SHA gateはFAIL。strict mergeは棄却した。
- 参考加重集計ではsaved `likpf_mean` RMSE 11.594898、replay mean 11.601992、最良medoid
  `pf_seed_medoid_k3_m0` 12.232271。最良medoidは338 wells改善 / 435悪化、最大回帰+18.542614。
- all-K unionの参考oracle差はexp237 base8比でrow -1.104800、block 128 -1.169740、
  block 256 -1.138025、block 512 -1.095331、whole-well -0.961581。ただし採用根拠にはしない。
- inference / submissionは実行していない。

## 2026-07-14 v3 parity修正

- 新規実験には分けない。仮説、候補集合、PF設定、K-medoids、評価条件は不変で、exp072互換性の
  実装バグ修正だから同じ`exp243`のv3として扱う。
- 根本原因: `numeric_array()`が常にfloat32を返し、typewell TVTと評価MD/Zがfloat32へ丸められた
  後でfloat64へ拡張されていた。trajectoryだけをfloat64で保ってもPF入力は再現できていなかった。
- `fba7683c`のGR sigmaはraw float64で`22.658358259724082`、旧float32経由で
  `22.658950811666866`。旧Kaggle記録`22.658950811666863`と後者が一致した。
- `numeric_array64()`を追加し、typewell TVT、評価MD/Zをfloat32経由なしでPFへ渡すよう修正した。
- canonical exp072 v2 cacheのraw SHA `14faee3a...f18`、decompressed SHA
  `99a3c70a...350`、schema SHA `700d3814...b8`をconfigに固定し、不一致ならPF前に停止する。
- PF入力はcanonical exp072 v2 cacheへ固定する。saved `likpf_mean`はSHA固定したexp209 enriched
  cacheから別名`likpf_mean_exp209_reconstructed`で復元し、ID/well/target/last-known/md-sinceを
  一対一確認してから結合する。
- `py_compile`、`ruff --select F821,E9`、Jupytext train/inference/4 wrapper `--test`、
  `make validate-exp EXP=exp243_pf_seed_medoids`をPASS。
- full packageはshard 0/1/2/3とも`parity_probe=false`、4 shards、全well設定。
- parity probe packageは`kentookumura/exp243-pf-seed-medoids-parity-probe-v1`、
  `fba7683c` 1 well / 407 rows、`parity_probe=true`、1 shard。bootstrap ZIP内でもfloat64入力、
  canonical SHA guard、config/helper一致を確認した。
- probeコスト: active variant 1、PF replay 1 well、500 particles × 128 seeds、K-medoids
  postprocess 3、LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、submit 0。
- packageはprepareのみで、Kaggle pushはまだ行っていない。

## 2026-07-14 v3 1 well parity probe実行

- ユーザーがKaggle実行を明示承認した。
- push対象は`kentookumura/exp243-pf-seed-medoids-parity-probe-v1`のみ。full shard 0/1/2/3は操作しない。
- version 1をpush。URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-parity-probe-v1
- pull成功、kernel id_no `127056541`、status `KernelWorkerStatus.RUNNING`。
- pullしたKaggle notebookのbootstrapで`fba7683c`、407 rows、1 shard、canonical cache SHA、
  typewell TVTと評価MD/Zのfloat64入力を確認した。
- CPU-only、internet off、500 particles × 128 seeds、PF replay 1 well、K=3/5/8、
  LightGBM config 0、fold 0、booster 0、parent/control再学習なし、inference/submit 0。
- probe v1はcanonical exp072 cacheに存在しない`likpf_mean`列を要求し、約17秒でPF前にERROR。
- v1の前提を修正し、canonical exp072 inputとexp209 parity controlを別々にSHA固定。復元列を別名にして
  merge衝突をhard error化した。回帰テスト2件、py_compile、ruff、strict validationを再PASS。
- 同じkernelへversion 2をpush。kernel id_noは`127056541`のまま、status `KernelWorkerStatus.RUNNING`。
- full shard 0/1/2/3は操作していない。
- version 2は`KernelWorkerStatus.COMPLETE`。PF runtime 7.003974秒、407 rows / 1 well。
- parityはmean absolute / RMSE / max absoluteすべて0.0、`1e-6`超0行、exact parity true。
- output row candidatesから独立再計算しても同じ0差を確認した。
- canonical exp072 input raw/decompressed SHA、schema SHA、exp209 control raw/decompressed SHAは期待値と一致。
- GR sigma `22.65835825972408`、seed base `787424823`で、float64修正契約を確認した。

## 2026-07-14 v3 full single-notebook実行承認

- 過去4 shardのruntime合計33,058.291秒（約9時間11分）を提示した。
- ユーザーが、合計時間が約9時間ならfull rerunを1 notebookで実行してよいと明示した。
- 実行対象はKaggle private CPU notebook 1本、全773 eligible wells。`well_shard_count=1`、active index 0。
- active variant 1、PF replay 1/well、500 particles × 128 seeds、K-medoids postprocess 3（K=3/5/8）。
- LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、inference/submit 0。
- seed、PF入力float64、trajectory/replay mean/K-medoidsまでfloat64、canonical exp072/exp209 SHA guardはprobe v2から変更しない。
- 過去実績から約9時間11分を見込む。Kaggle CPU 12時間枠に対する余裕は約2時間49分で、runtime変動によるtimeoutリスクは残る。
- canonical kernel予定: `kentookumura/exp243-pf-seed-medoids-train`。package検証後にpushする。
- Jupytext同期/`--test`、py_compile、ruff F821/E9、回帰テスト2件、strict experiment validationをPASS。
- canonical slugのpre-push pullは403、自分のkernel一覧には同slugが存在しないことを確認した。既存のprobe/shard slugは変更しない。
- canonical packageをCPU / internet off / run-on-pushで生成。bootstrap ZIP内は12 files、実行時config SHA
  `665af58943407e5913c5a1bd08eca7cc5218afa5ace0c8e67b68c3498712c536`。
- bootstrap内で`parity_probe=false`、1 shard / index 0、well include空、max wells null、全3 kernel sources、
  `numeric_array64`とsingle-notebook guardを確認した。
- `kaggle kernels push -p experiments/exp243_pf_seed_medoids/kaggle/train`でcanonical kernel version 1をpush。
- URL: https://www.kaggle.com/code/kentookumura/exp243-pf-seed-medoids-train
- pull成功、kernel id_no `127058309`、status `KernelWorkerStatus.RUNNING`。
- Kaggle側からpullしたbootstrapでもconfig SHA `665af589...2c536`、1 shard / index 0、all wells、
  `parity_probe=false`、float64入力、CPU/internet off、3 kernel sourcesの一致を確認した。
- completion後はlogsで773 wells、replay exact parity、source/schema SHA、candidate/oracle readoutを確認する。

## 2026-07-15 v3 full single-notebook完了監査

- canonical kernel version 1 / id_no `127058309`は`KernelWorkerStatus.COMPLETE`。
- logsは8-well間隔で768 / 773 wellsまで進捗を示し、その後summary・metrics・全生成物保存まで完了した。
- PF audit runtime 37,067.406秒（約10時間18分）、notebook全体ログ約37,369秒。
- 3,783,989 rows / 773 wells、target/well-status/diagnosticはいずれも773 wells、全status `ok`。
- replay parityはmean absolute / RMSE / max absoluteすべて0、`1e-6`超0 / 3,783,989行、exact parity true。
- saved/replay `likpf_mean` RMSEはともに11.594897672。
- canonical exp072 raw/decompressed SHA `14faee3a...f18` / `99a3c70a...350`、schema
  `700d3814...b8`、exp209 control raw/decompressed `b50b4d1e...b64b` / `ee3b548b...e3f4`は期待値と一致。
- rich display数値とSHA確認が必要なため、巨大なrow candidates / cluster manifestは取得せず、
  `metrics.json`、summaryと小さい10 CSVだけを`--file-pattern`で取得した。
- 取得した10 CSVはKaggle metrics記録SHAと全て一致。downloaded metrics SHA
  `693efe12ab9d379a5afe321d31a6ba8d28c38e85f50e3f529d0101d08f4cfc89`。
- Kaggle出力`metrics.json`を正確にコピーして上記SHAを確認後、local machine-readable recordには
  kernel metadataと最終decisionだけを追記した。Kaggle生成指標配列とartifact SHAは変更していない。
- 最良direct medoid `pf_seed_medoid_k3_m0`はRMSE 12.296667365（base比+0.701769693）。
  344 wells改善 / 429悪化、median +0.054901、worst +20.953998。
- 同候補は全distance bucketで悪化。1000+ +0.769445、hidden-like spatial +0.705972、
  typewell-purged +0.616778でdirect replacement guard不通過。
- exp237 base8 + K8 oracleはrow -1.348387、block 128 -1.405104、block 256 -1.372076、
  block 512 -1.316683、whole-well -1.092839。K8 medoidはunion内unique-best 43.8800%。
- whole-well oracleはK8追加で374 / 773 wells改善。all-Kは376 wells、whole-well RMSE 5.493181で、
  K8単独5.499587から追加-0.006406のみ。後続候補はK8へ限定する。
- cluster median max-pairwise seed distance 15.603104、normalized entropy 0.891478、max cluster mass
  0.3828125で、近重複や単一cluster collapseだけではない。
- 判定: exp243のcandidate generation仮説は支持。medoid単体・無条件平均は不採用。selector、raw-test
  inference、submissionは未実行。
- 次は保存済みK8候補を再利用するtarget-free selectability audit。PF再実行やcandidate追加は行わない。
