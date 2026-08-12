# exp252_pf_seed_medoid_selectability_audit セッションノート

## 目的

`KAGGLE_DIRECTION.md` の高優先度backlog `pf_seed_medoid_selectability_audit`を実装する。
exp243 v3 exact-parity K8 medoidがbase8へ追加したtrajectory-mode headroomを、固定した
target-free PF/cluster scoreで識別できるか監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v1完了。candidate likelihood signal部分支持、bank gate不採用
- inference / submission: disabled

## 実装前コストガード

- active diagnostic variant: 1
- score: bank 10本、candidate 7本（すべてstandalone、合成なし）
- scope: row / block 128 / block 256 / block 512 / whole-well
- LightGBM config: 0
- fold: 0
- booster: 0
- PF/Beam/likPF replay: 0
- parent/control再学習: なし
- runtime: Kaggle CPU、GPU/internet disabled、single process
- raw-test inference / submission: なし

## 固定評価contract

- K8のみ。base8 8本 + K8 medoid 8本の候補値はexp243保存値から変更しない。
- rowはabsolute error、block/wellはRMSE。blockはwell先頭から非重複128/256/512行、末尾も保持。
- bank labelはbest K8がbest base8を`1e-6 ft`より大きく改善したunit。
- candidate primary labelはuseful K8 bank内でunion-bestとtie tolerance内のmedoid。
- scoreはtrue TVTを受け取らない処理で先に固定し、その後だけloss/labelを作る。
- coverageは固定top 10%。shuffleはscore/scope名を含むstable SHA seedで固定する。
- score/threshold grid、score合成、selector学習、candidate平均、K3/K5、PF再実行は禁止。

## 再現性

- `docs/06_reproducibility.md`を2026-07-15に確認。
- exp243 row candidates / cluster manifestのdecompressed SHAと、cluster summary / PF diagnosticsの
  raw SHAをhard guardする。
- audit内の乱数はshuffled-score controlだけ。local RNG、stable SHA seed、single process。
- gzip出力は作らない。出力CSV/JSON/metricsのfile SHAを記録する。
- 固定exp243入力に対するdiagnostic determinismだけを主張し、prediction/submission anchorとはしない。

## コマンドログ

### 2026-07-15 作成

    make new-steering EXP=exp252_pf_seed_medoid_selectability_audit
    make new-exp EXP=exp252_pf_seed_medoid_selectability_audit

- steering: `docs/legacy/steering/20260715-exp252-pf-seed-medoid-selectability-audit/`
- 親: `exp243_pf_seed_medoids`
- route: `pf_beam`
- exp243 canonical v3 kernel: `kentookumura/exp243-pf-seed-medoids-train` version 1。

### 2026-07-15 実装

- compact self-contained Jupytext trainを10章 / 1,345行で実装した。
- exp243 4生成物のpath解決、decompressed/raw SHA hard guard、3,783,989 rows / 773 wells、
  K8 manifest 8 slots/well、cluster summary / PF diagnostics parityをpreflightする。
- score stageとlabel stageを関数境界で分離し、`freeze_target_free_scores()`はtrue TVTを参照しない。
- bank 10 score、candidate 7 scoreをstandaloneで固定し、合成・grid・score選択を実装していない。
- row / block 128/256/512 / whole-wellのloss、best-source label、AUC、固定top-decile
  coverage、top1 regret、stable shuffled controlを実装した。
- 3.78M rows × 16候補をlong 30M-row tableとして常設せず、static scoreはwell別positive count、
  dynamic scoreだけscopeごとに逐次sortするmemory contractにした。
- 出力はscore contract、scope metrics、score metrics、top1 regret、by-well、summary、metrics。
- inference notebookは明示的なdisabled guardで停止し、submission.csvを生成しない。

### 2026-07-15 静的検証・package

    .venv/bin/python -m py_compile experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_train.py experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_inference.py
    .venv/bin/ruff check experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_train.py experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_inference.py experiments/exp252_pf_seed_medoid_selectability_audit/tests/test_exp252_pf_seed_medoid_selectability_contract.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp252_pf_seed_medoid_selectability_audit/exp252_pf_seed_medoid_selectability_audit_inference.py
    make validate-exp EXP=exp252_pf_seed_medoid_selectability_audit
    make validate-template
    make test
    make prepare-kaggle-notebooks EXP=exp252_pf_seed_medoid_selectability_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp252-pf-seed-medoid-selectability-audit-train --title 'exp252 pf seed medoid selectability audit train' --run-on-push --strict"

- py_compile / Ruff / Jupytext round-trip / strict experiment validation / template validation: PASS。
- synthetic 2-well auditで5 scope、各48 score metric rows / 14 top1 rows、label / regret contract: PASS。
- repo pytest: 23 passed。exp252 contract test 4件を追加した。
- notebook: train 22 cells（10 code / 12 markdown）、inference 6 cells（2 code / 4 markdown）。
- 親exp243正規trainは238行で重いhelper依存。本実験は1,345行self-containedで、input preflight、
  score、scope、label、AUC/control、保存をnotebook上へ展開した。同一exp helper importなし、`__file__`なし。
- canonical kernel: `kentookumura/exp252-pf-seed-medoid-selectability-audit-train`。
- metadata: private CPU、GPU/TPU/internet off、run-on-push、competition source 1、kernel source exp243 1本。
- source / loose package / bootstrap内configとtrain sourceはbyte-identical。
- config SHA: `9ff06807819959b9c675ec44e5c1b841bb255cc6498070856e783077d65cf60e`。
- train source SHA: `2b29ff0a30fa5eaea718cf0fb33050c54301042185d9027198a3e01aee75e255`。
- bootstrap ZIP: 7 files、SHA `cc68c6a3956a3172054972357e9331533b62a2960f7adb9668b3a80d89a11a85`。
- package prepareのみ。Kaggle push / run / output取得は行っていない。

## 次のアクション

likelihood mass / rank / gapは、base8 fallbackを保持するfold-safe selectorまたは既存
`topk_path_confidence_features`へadd-only candidate-ranking特徴量として再利用できる。
3 score単独の固定selector、現時点でのraw-test inference、submissionへは進めない。

### 2026-07-15 Kaggle CPU実行承認・push前ガード

- ユーザーが「実行してください」と明示したため、canonical Kaggle CPU trainのpushを承認済みと扱う。
- 実行対象: private CPU notebook 1本、active diagnostic variant 1、bank score 10本、
  candidate score 7本、scope 5面。
- LightGBM config 0、fold 0、booster 0、PF/Beam/likPF replay 0、GPU 0、
  parent/control再学習なし、raw-test inference / submission 0。
- strict validation / Ruff / source-package parityを再確認してPASS。
- package metadataはCPU、GPU/TPU/internet off、run-on-push、exp243 kernel source 1本。
- config SHA `9ff06807819959b9c675ec44e5c1b841bb255cc6498070856e783077d65cf60e`、
  train source SHA `2b29ff0a30fa5eaea718cf0fb33050c54301042185d9027198a3e01aee75e255`。
- canonical kernel pullは403、自分のkernel list exact searchは`Not found`で、既存kernelを確認できなかった。
- push予定: `kentookumura/exp252-pf-seed-medoid-selectability-audit-train` version 1。
- canonical kernel version 1を正常push。URL:
  https://www.kaggle.com/code/kentookumura/exp252-pf-seed-medoid-selectability-audit-train
- push後pull成功、kernel id_no `127304849`、status `KernelWorkerStatus.RUNNING`。
- Kaggle側からpullしたbootstrapのconfig / train sourceはlocal sourceとbyte-identical。
- pulled bootstrap config SHA `9ff06807819959b9c675ec44e5c1b841bb255cc6498070856e783077d65cf60e`、
  train source SHA `2b29ff0a30fa5eaea718cf0fb33050c54301042185d9027198a3e01aee75e255`。
- 同じcanonical IDで監視し、RUNNING中のlogs空やstatus API揺れを理由に再pushしない。

### 2026-07-15 Kaggle CPU train v1完了

- canonical kernelは`COMPLETE`。Kaggle id_no `127304849`。
- runtime summary 86.053秒、notebook全体約108秒。
- Kaggle docker image:
  `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`。
- 3,783,989 rows / 773 wells、K8 manifest 6,184 rows。4入力のSHA hard guardはすべてPASS。
- row candidates decompressed SHA:
  `0583836a76e1b9515f8289965a25b0f41cf661294a20195a3021efbcd43e32bd`。
- cluster manifest decompressed SHA:
  `5b8f3d90e4b858aa1607e26a8c7fb30432d0d4c041bdf677365bd2f6dd9b9dca`。
- cluster summary raw SHA:
  `d3eea9f7c4ff777bed0a3c2ff2c60a076a1d49257c85b547a7b279df9735d333`。
- PF diagnostics raw SHA:
  `acb89b30d65c7676133097964910b7bd788b593205a2b68d1de6c1df6a57bf72`。
- target-free score freeze後にtrue targetを読むcontractを実runでも通過。
- score metrics 240 rows、top1 regret 70 rowsを生成した。

#### Scope oracle

- row: base8 4.564605 -> union 3.216218、delta -1.348387、useful 1,666,895 / 3,783,989。
- block128: 4.805040 -> 3.399936、delta -1.405104、useful 13,186 / 29,948。
- block256: 4.883135 -> 3.511059、delta -1.372076、useful 6,539 / 15,174。
- block512: 5.036480 -> 3.719798、delta -1.316683、useful 3,274 / 7,787。
- whole-well: 6.592426 -> 5.499587、delta -1.092839、useful 374 / 773。

#### Selectability readout

- bank最良`resampling_rate`: 5 scope平均AUC 0.541785、shuffled比+0.040204、
  whole-well AUC 0.560593。base8 fallbackを安全に切るgateとしては弱い。
- candidate `cluster_likelihood_mass`: 平均AUC 0.574731、shuffled比+0.061858、
  whole-well AUC 0.675214。
- candidate `medoid_likelihood_rank_score`: 平均AUC 0.578665、shuffled比+0.076162、
  whole-well AUC 0.655102。
- candidate `medoid_likelihood_gap_from_best`: 平均AUC 0.575143、shuffled比+0.072390、
  whole-well AUC 0.654235。
- 上記3 candidate scoreはすべて5/5 scopeでreal AUCがshuffledを上回った。
- whole-well likelihood-mass top1はuseful coverage 0.516043、union-best match 0.280749、
  regret平均2.416045 ft、p90 6.275659 ft、best base8比loss平均+3.194947 ft。

#### 判断とartifact

- candidate likelihood順位付けは部分支持。fold-safe selectorへのadd-only特徴量候補になり得る。
- bank gate、fixed top1、3 score単独selectorは不採用。現時点でraw-test PF再生成、inference、
  submissionは実施しない。
- Kaggle outputから必要な小型CSV/JSONだけをpattern取得し、SHAを検証した。大きいoutput archiveは取得していない。
- by-well SHA `2ad021ca53c13478cfe0d2cc453fc521e704722a0dd2430e323de9f5c7a79ba1`。
- scope metrics SHA `4026b414486fe1cfb24ed46014aa25e3b2712d40a413080b183158fbbc0b3ad9`。
- score contract SHA `fe7c4fc2589e6e1723ce6e7c139826ad693762792efa7c330d46c907c73a8514`。
- score metrics SHA `fb95a238403d1e3c5bb6b58a1f56f1766678456904417ef995afc8f51f99fb81`。
- summary SHA `62ecb2a3747e83da5dba75ab9ac4ac0b885975d17b78e8a492570bd6d0ff9f98`。
- top1 regret SHA `70ca703bb0c0740f500db2493ec2a18f2d8f35471987b4296e919938bddfdd89`。

### 2026-07-15 Selector候補と生成時間の追記

- ユーザー確認に基づき、likelihood mass / rank / gapはK8 medoid内のcandidate-ranking特徴量として
  selector候補になり得ることを明記した。
- 採用形はbase8 fallbackを維持する二段構成。別のraw-test-safe bank gateを先に置き、K8使用時だけ
  3特徴を既存candidate selectorへadd-only投入する。
- exp252単体はbank gateやselector全体を支持していない。outer-well OOF、raw-test parity、fold / near /
  hidden-like / worst-well guardsを別途必要とする。
- 保存済み候補からのscore readoutはexp252実測86.053秒 / 773 wells。
- raw入力からのPF seed bank + medoid生成はexp243 v3実測37,067.406秒 / 773 wells、
  平均47.953秒/well。
- hidden test約200 wellsを同じwell長分布と仮定した単純比例は約9,590.5秒、2時間40分。
  raw-test未実測であり、K8-only短縮とselector学習・推論時間は未計測。
