# exp376_exp226_formation_conditioned_k16_donor_kernel セッションノート

## 目的

exp226の正解TVT由来K=16 donor slopeと後段処理を固定し、同じXY近傍50の
weightだけを、outer-train wellから推定する6地層の相対座標でsoft reweightする。
初回セッションで設計を固定し、2026-07-24の追加指示でcompact self-contained
train候補とStage 0/1/2を実装した。続くユーザー指示を、正規train notebook採用と
Kaggle CPU run 1回の承認として受けた。推論・提出は行わない。

## 現在の状態

- Route: `pf_beam`
- 状態: `kaggle_cpu_v2_completed_technical_pass_direct_fail_novelty_fail`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- バックログ: 中・P2・CPU・implementation-only。既存P1を追い越さない。
- 予定量: 1 scientific variant / 0 model config / 5 reporting folds /
  0 trained fold / 0 booster / parent control再実行0。
- CV / LB: v2 direct `9.443257189578157` / 未提出。
- notebook: 4,007行・9章・19セルのcompact self-contained trainを正規trainへ採用。
  正規inference notebookはscaffoldのまま変更しない。
- implementation / execution: v2まで完了。technical / Stage 0 PASS、
  direct / novelty FAIL。inference / submissionは未実施・未承認。

## Kaggle CPU run承認とpush前コスト確認

- 承認日: 2026-07-24
- 承認元: ユーザーの「実行してください」
- 実行対象: 正規train notebook採用とKaggle CPU run 1回だけ
- kernel id: `kentookumura/exp376-formation-k16-donor-train`
- title: `exp376 formation k16 donor train`
- 実行量: 1 scientific variant / 0 LightGBM config / 5 reporting folds /
  0 trained fold / 0 booster
- 親/control: 保存済みexp226 OOFだけを読み、再学習・再生成0
- runtime: CPU、GPU無効、internet無効、nearest-neighbor worker 1
- kernel sources:
  - `kentookumura/exp226-k16-kappa-repro-train`
  - `kentookumura/exp263-last-anchor-pair-cache-train`
  - `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- 非対象: current-test生成、inference、submission
- package確認:
  - `run_on_push=true`、`enable_gpu=false`、`enable_internet=false`
  - competition source 1件、kernel source 3件、dataset/model source 0件
  - packaged config SHA256:
    `a67d6f65307434fd3b8f1df8e0a7dca7ed9f0db51c5dad18dfee8fd2157b86e6`
  - packaged source SHA256:
    `625f05620785687a9cc8d6154d8eb63a30a4350b83ab7ad44a57c92915a5d8e5`
  - packaged notebook SHA256:
    `a92fed7191f436d3e17a69182e89450352a53b4b4a5154330bf91ceb9aa1781e`

## コマンドログ

2026-07-24に実行したscaffold作成:

```bash
make new-steering EXP=exp376_exp226_formation_conditioned_k16_donor_kernel
make new-exp EXP=exp376_exp226_formation_conditioned_k16_donor_kernel
```

作成先:

- `docs/legacy/steering/20260724-exp376-exp226-formation-conditioned-k16-donor-kernel/`
- `experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/`

初期design/implementation段階ではKaggle package/push/run、推論、提出を
実行していない。

Kaggle CPU run開始:

```bash
make push-kaggle-train \
  EXP=exp376_exp226_formation_conditioned_k16_donor_kernel
```

- push時刻: `2026-07-24T05:00:14Z`
- kernel: `kentookumura/exp376-formation-k16-donor-train`
- kernel id_no: `128436621`
- version: 1
- push結果: success、`run_on_push=true`で開始
- 1回分の実行承認は消費済み。再pushはしない。

### v1確定結果

- Kaggle status: `ERROR`
- 5 foldsのformation-conditioned予測: 完了
- fold開始時刻:
  - fold 1: `205.240573s`
  - fold 2: `397.605534s`
  - fold 3: `592.611775s`
  - fold 4: `787.040015s`
  - fold 5: `982.759395s`
- error: `TypeError: unhashable type: 'list'`
- error時刻: `1237.617791s`
- error位置:
  `freeze_target_free_bundle -> frame_content_sha256(reference_manifest)`
- 原因: `unavailable_reference_wells`を含むlist-valued object列を
  `pd.util.hash_pandas_object(..., categorize=True)`へ直接渡した。
- truth load: 0。Stage 0/1/2、CVは未評価。
- log最終時刻: `1248.846058s`
- failed kernelのoutput file一覧は空で、partial artifactは取得できなかった。
- 科学的なPASS/FAILではなく、pre-truth artifact hash境界の技術エラー。

### v1後の局所修正

- object列のうち`Mapping/list/tuple/ndarray/set`だけをcanonical JSONへ正規化し、
  scalar string/numberとformation/K16/weight/gateロジックは変更しない。
- list-valued manifest cellのhash再現testを追加した。
- 専用test: `4 passed`
- `py_compile`: PASS
- ruff F821: PASS
- Jupytext conversion / round-trip: PASS
- 修正後source SHA256:
  `aaaa75a4378b858d6dc180010e9f7cdaf2a72107e644287ae6ccdbcb0a822bea`
- 修正後正規notebook SHA256:
  `9bf891557b59c5409b04db42bc432bd09de83b44f96c136360523f1e019f7980`
- v2 package/push/run: 2026-07-24にユーザーが明示承認。未実施。

### v2 push前コスト再確認

- 承認元: ユーザーの再度の「実行してください」
- target: 同じcanonical kernelのversion 2
- 実行量: 1 scientific variant / 0 LightGBM config / 5 reporting folds /
  0 trained fold / 0 booster
- 親/control: 保存済みexp226 OOFのみ。再学習・再生成0
- runtime: CPU、GPU無効、internet無効、nearest-neighbor worker 1
- kernel sources: v1と同じ3件
- 科学設定差分: なし
- コード差分: container-valued reference manifest cellのlogical hash正規化だけ
- 非対象: current-test生成、inference、submission
- v2 package確認:
  - `run_on_push=true`、`enable_gpu=false`、`enable_internet=false`
  - competition source 1件、kernel source 3件、dataset/model source 0件
  - packaged config/source byte parity: PASS
  - packaged config SHA256:
    `b6ea3eaa05a9179d259e66a2bbf3b4ceda460d0647af9610bb7a6653bec77358`
  - packaged source SHA256:
    `aaaa75a4378b858d6dc180010e9f7cdaf2a72107e644287ae6ccdbcb0a822bea`
  - packaged notebook SHA256:
    `3d60cc2b6aa1f7c4f502ce8156333b47ae832e9c62fca8b806933ed8d3b23b7a`
- push時刻: `2026-07-24T06:06:24Z`
- push結果: 同じcanonical kernelへversion 2としてsuccess
- v2の実行承認は消費済み。再pushはしない。

### v2確定結果

- kernel: `kentookumura/exp376-formation-k16-donor-train`
- id_no / version / status: `128436621` / `2` / `COMPLETE`
- completed_at: `2026-07-24T06:32:35.271078+00:00`
- metrics完了 / log最終: `1574.961352 / 1585.333365 sec`
- execution: 1 variant / 5 reporting folds / 0 config / 0 trained fold /
  0 booster / parent control再実行0 / GPU false
- Technical: PASS
- Stage 0 target-free: PASS
  - rows / wells / segments: `3,783,989 / 773 / 12,368`
  - formation factor min/max: `0.5115013215 / 0.9999999997`
  - finite / fallback: `1.0 / 0.0`
  - ESS比p05: `0.9271732756`
  - valid reference/truth/formation read: `0 / 0 / 0`
  - freeze前truth access: 0
- Stage 1 direct: FAIL
  - control / variant RMSE: `9.4271095966 / 9.4432571896`
  - delta: `+0.0161475930 ft`
  - fold delta:
    `[+0.023440409,+0.041883306,-0.020347810,+0.029009494,+0.010357627]`
  - nonworse folds: `1/5`
  - near/mid/1000+/hidden spatial/hidden typewell delta:
    `-0.010632796 / +0.009193834 / +0.017799272 /
    +0.018538133 / +0.019237282 ft`
  - by-well: 376改善 / 397悪化、median`+0.002754783 ft`、
    p95`+0.376679336 ft`
  - worst: `a3518960`、`+1.891559930 ft`
- Stage 2 fixed12 add-one novelty: FAIL
  - H512 base / add-one: `3.683762664 / 3.664359132`
  - H512 gain: `0.019403532 ft < 0.05`
  - whole-well base / add-one: `4.784903881 / 4.769361863`
  - whole-well gain: `0.015542019 ft < 0.05`
  - H512 strict unique-best: `0.093874406`、PASS
  - fold gain:
    `[0.013931849,0.022711883,0.021749835,0.014598981,0.023914830]`
  - improved folds: `5/5`、PASS
- exp226 prediction correlation: `0.999999782066`
- decision:
  `close_formation_conditioned_donor_branch_without_rescue_grid`
- 選択取得した小さいoutput:
  `/tmp/kaggle-output/exp376_exp226_formation_conditioned_k16_donor_kernel/train_v2`
  - summary / freeze / SHA manifest / Stage 0 guard
  - direct / novelty metricsとby-well、correlation
  - input / raw input / truth / reference manifest、formation schema
  - 45.5 MB OOF本体と33.5 MB block assignmentは全量取得せず、
    manifest内のfile/decompressed/logical SHAだけを記録

2026-07-24の実装・検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/\
exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/\
exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py
.venv/bin/python -m py_compile \
  experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/\
exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py
.venv/bin/ruff check \
  experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/\
exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py \
  --select F821
.venv/bin/pytest -q \
  experiments/exp376_exp226_formation_conditioned_k16_donor_kernel/tests/test_exp376_formation_conditioned_k16_donor_kernel.py
make validate-exp EXP=exp376_exp226_formation_conditioned_k16_donor_kernel
make prepare-kaggle-notebooks \
  EXP=exp376_exp226_formation_conditioned_k16_donor_kernel \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp376-formation-k16-donor-train \
  --title 'exp376 formation k16 donor train' --run-on-push --strict"
```

結果:

- 専用test: 初回`3 passed`、v1エラー再現修正後`4 passed`
- `py_compile`: PASS
- ruff F821: PASS
- Jupytext conversion / round-trip: PASS
- strict experiment validation: PASS
- canonical train adoption: PASS
- packaged config/source byte parity: PASS
- metadata/bootstrap config/source/run flags: PASS
- `task validate-exp`はTask CLI未導入で実行不能だったため、
  同等の`make validate-exp`を実行してPASS。

## 固定した変更点

- exp226のXYだけで選んだ同じ50 donor segmentを使う。
- 6地層面から、segment midpointの面相対距離6個と隣接面厚5個を作る。
- fold内outer-train donorだけのmedian/MADで11次元signatureをrobust標準化する。
- `g_form = 0.5 + 0.5 * exp(-0.5 * d_form^2)`、
  `w_new = w_xy * g_form`の1式だけを使用する。
- nonfinite signatureは`g_form=1.0`として親weightへ戻す。
- K16、正解TVT由来raw/smoothed slope、XY近傍50、bandwidth 500 ft、
  ridge 1、rho 10、kappa 12項、ANCC local theta、GR correction、
  U-projectionは変更しない。

## 実装内容

- target-free well loaderは`X/Y/Z/TVT_input/GR`だけを読み、outer-validの
  `TVT`と6地層列をobjectへ載せない。
- 各foldでouter-train wellだけの完全な6地層行からwell median XY/地層値を作り、
  `FormationPlaneKNN(k=10)`を構築する。
- outer-train donorは自身を参照から除外し、outer-valid queryはouter-train
  referenceだけで6面を推定する。
- segment midpointの`Z-F_m` 6次元と`F_{m+1}-F_m` 5次元を作り、
  donor segmentだけのmedian/MADで標準化する。
- exp226と同じstable XY近傍50を選び、raw/smoothed local-linearの両方へ
  同じfixed formation factorを適用する。near-strike committeeは変更しない。
- Stage 0ではfold/reference manifest、signature schema、support ledger、
  OOF prediction、kappa、fixed12 bank/blockをtruth前に保存・SHA freezeする。
  FAIL時はtruth loaderを呼ばずbranchを閉じる。
- Stage 0 PASS時だけ、paired direct RMSEとH512/whole-well add-one oracleを評価する。

## Notebook構成比較

親exp226にはcompact self-contained版がなく、正規train Jupytextは111行・6章で
1,180行のhelperを呼ぶ既存構成だった。exp376候補は同一exp helper importを使わず、
exp226 full downstream、formation生成、fixed12 bank、freeze、Stage 0/1/2を
4,007行・9章・19セルへ展開した。`__file__`は含まない。

## 設計根拠

- exp226保存OOF: CV `9.427109596582213`、decompressed SHA256
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- exp287はfold-safe formation追加でexp264比`-0.324103 ft`かつ5/5 folds改善、
  Public LB `7.530`だった一方、worst-wellは`+8.228410 ft`でtail gateに失敗した。
- exp009/exp150の単純なformation direct利用はnegative evidence。
- exp362のlocal donor gradientは0/12,368 segments採用で完全退化したため、
  本設計では既存XY supportを残し`g_form>=0.5`とする。
- 6地層面はwell内でほぼ平行なので、面ごとの傾きではなく
  relative depth 6次元とformation thickness 5次元を使用する。

## 固定した評価gate

- Stage 0: 3,783,989 rows / 773 wells / 12,368 segments、formation factor
  finite 100%かつ`[0.5,1.0]`、fallback`<=1%`、ESS比p05`>=0.75`、
  valid reference/truth read 0。
- Stage 1: exp226比pooled gain`>=0.05 ft`、4/5 folds、
  5 scope delta各`<=+0.02 ft`、by-well p95`<=0 ft`、
  worst-well`<=+0.25 ft`。
- Stage 2: exp293 fixed12へのadd-oneでH512とwhole-well oracle gain各
  `>=0.05 ft`、H512 strict unique-best block率`>=5%`、正方向5/5 folds。
- Stage 1/2のどちらかをPASSしてもcurrent-test生成、selector組み込み、
  inference、submissionは別承認まで行わない。

## 再現性メモ

- seed policy: stable SHA256 well-id 5-fold assignment。RNGなし。
- stochastic components: なし。
- CPU/GPU runtime: v1/v2ともKaggle CPU、GPU/internet無効、
  nearest-neighbor worker 1。
- SHA: input/fold/reference/schema/logical feature/predictionを記録し、
  gzipはdecompressed content SHAを主証拠にする。
- model manifest: 学習モデルがないため非該当。
- kernel version: v1 ERROR / v2 COMPLETE、id_no `128436621`。
- v2 config / source SHA:
  `b6ea3eaa05a9179d259e66a2bbf3b4ceda460d0647af9610bb7a6653bec77358` /
  `aaaa75a4378b858d6dc180010e9f7cdaf2a72107e644287ae6ccdbcb0a822bea`
- input manifest file / logical SHA:
  `a4a4a99ce3edab30aa52c41079aebdd47d7bc9f2fb93c9c44b84b98a5e81fbad` /
  `fee6dfc3d33656b087d55890597ebad0cd84e8476160c74b0c0bcd535bd0e279`
- formation schema file SHA:
  `1f9dec7ac0025213629f45685d772e27fa067169cf3ee610a0c59cd04bcc1470`
- support / reference logical SHA:
  `1b14e696817dded0ec81d357c9529fdee263b6a7dc6fa432c8427ad57dc19258` /
  `92dd0be43ac74c1210a5babdea256eec271ee430cf5c2659bb8ce69b923137a8`
- prediction logical / decompressed SHA:
  `5205c67f6cad8d549863f122ab989bf2874587c574494b59b639a1bc5d66fb25` /
  `49621fb7838bb5234553d507d9e6fe38a55127b25e82621ee655091fe6b340a0`
- truth logical SHA:
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`
- candidate bank / block decompressed SHA:
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474` /
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
- submission SHA: submission未実施。
- deterministic anchor: いいえ。成功run間のlogical prediction SHA一致を
  確認していないため、v2単発成功として扱う。

## 次のアクション

fixed soft factorはsupportを安全に維持したが、direct/tailとnoveltyの固定gateを
満たさないためbranchを閉じる。weight/surface/signature/K/bandwidth救済grid、
current-test生成、selector組み込み、inference、submission、version 3は行わない。
同じK16 donor kernel条件付けの新規backlogは追加せず、既存の独立候補の優先順位を
維持する。
