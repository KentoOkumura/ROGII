# exp250_segment_local_negative_space_gr_corridor_audit セッションノート

## 目的

exp246 の full-tail global surface / valid_after_history と、局所segmentの別仮説を分離する。既存の exp249 は変更せず、現行の segment_local_negative_space_gr_corridor_audit 実装契約を exp250 として実装する。

fixed exp072 candidateを変更せず、MD-local GR mismatch DAGのminimum-bottleneck corridorがbad candidate rowをtarget-freeに濃縮できるかをStage 0 / Stage 1で監査する。

## 現在の状態

- Route: pf_beam
- 状態: Stage 0 manual parity PASS、Stage 1完了、guard 2/8 PASSで不採用
- active mode: stage1_full_audit
- decision: `fail_close_segment_local_hard_use_and_grid_search`
- inference / submission: disabled
- exp249: 既存experimentを保持し、本実験から分離

## 実装前コストガード

- active diagnostic surfaces: 2 (real_gr, shuffled_typewell_gr)
- LightGBM / CNN / HMM config: 0 / 0 / 0
- model training fold: 0
- booster: 0
- PF/Beam/likPF再生成: 0
- parent/control再学習: なし
- runtime: Kaggle CPU、GPU/internet disabled、single process
- raw-test inference / submit: なし

## 固定科学contract

- horizontal segment: MD 256 ft、stride 128 ft、4 ft/bin、通常64 columns。
- 末尾はright-alignし、短tailは実在範囲のみ。16 columns未満はprimary topologyから除外する。
- typewell grid: flat-Z prior中心±256 ft、4 ft/state、129 states、endpoint extrapolationなし。
- GR normalization: horizontal/typewellをwell全体で別々にmedian/IQR robust-z、[-8, 8] clip。
- control: SHA256(experiment_name, well, seed=42)の25–75% circular shift。
- graph: 右向きdy=-1/0/+1、単独unsupported columnだけdx=2, dy=-2..+2。
- primary: minimum-bottleneck tau_star、tieは累積node cost。corridorはtau_star + 0.25以下のforward/backward reachable node。
- first segmentはlast_known_tvt ±8 ft anchored graph、後続はunanchored spanning graph。segment間historyなし。
- fixed candidate 5本のprimary riskはcorridor_outside_fraction。truth/errorはgraph完成後の評価だけに使う。
- overlapはviewを統合せず、path TVT差、corridor Jaccard、risk相関、event一致だけを保存する。

## 再現性

- docs/06_reproducibility.mdを2026-07-14に確認。
- audit内の新規乱数なし。shuffled controlだけstable SHAから決定する。
- sorted well / segment / candidate / state、single process、Python hash()・global RNG不使用。
- candidate cacheはraw/decompressed SHA、gzip出力はmtime 0でraw/decompressed SHAを分ける。
- raw file inventory、hidden-like assignment、config、plot/CSV/JSON SHAをsummaryへ保存する。
- fixed inputに対するaudit determinismだけを主張し、upstream exp072 cacheのstochastic provenanceを継承する。prediction/submission deterministic anchorではない。

## コマンドログ

### 2026-07-15 Stage 0 push前ガード

    make validate-exp EXP=exp250_segment_local_negative_space_gr_corridor_audit

- strict experiment validation: PASS。
- active mode: `stage0_preview`、diagnostic surface: 2 (`real_gr`, `shuffled_typewell_gr`)。
- LightGBM / CNN / HMM config: 0 / 0 / 0、fold: 0、booster: 0。
- PF/Beam/likPF再生成: 0、parent/control再学習: なし。
- runtime: Kaggle CPU、GPU/internet disabled、single process。
- 初回指定kernel: `kentookumura/exp250-segment-local-negative-space-gr-corridor-audit-train`（slug 59文字）。
- source / loose package configはbyte-identical、SHA256: `5562a968e997463a61504f15d29a6fae0e2d4790cdbeb66fee2749077d7c6611`。
- exp249は操作対象外とし、exp250だけをpushする。
- push前の59文字ID pullはKaggle API 403で、既存kernelとして確認できなかった。
- 59文字IDへの初回pushはKaggle `SaveKernel 400`（詳細なし）で失敗した。既存の正常なmetadataが50文字以下であることと整合するため、Kaggle IDだけを全要素を残した48文字の `kentookumura/exp250-seglocal-negspace-gr-corridor-audit-train` に短縮した。実験名、科学contract、configは変更していない。
- 短縮ID/titleを同時指定してpackageを再prepareし、strict validationとsource/package config一致を再確認した。以後はこのIDをexp250のcanonical kernelとする。
- 短縮IDへのpushはKaggle側の `Maximum batch CPU session count of 5 reached` により未開始。占有中はexp244 cache 4件とexp243 train 1件で、すべてRUNNING。別実験を停止せず自然終了後に同じIDを再pushする。
- ユーザー指示によりCPU枠の監視を停止した。枠解放の連絡後、同じcanonical package / kernel IDでStage 0 pushから再開する。
- 枠解放後の再pushでは、前回容量エラー時にKaggle側へ空のkernel shellだけが作られていたため、pull 500 / status 404 / push `Notebook not found` となった。ユーザー承認付きでsession/outputのない空shellを削除し、同じcanonical IDを作り直した。
- Stage 0 version 1を正常push。kernel ID: `kentookumura/exp250-seglocal-negspace-gr-corridor-audit-train`、Kaggle `id_no`: `127222097`。push直後のmetadata pull: PASS、status: RUNNING。
- ユーザー指示によりversion 1の監視を停止した。停止時点はRUNNING、CLI logsは空。完了連絡後に同じkernelのlogsとStage 0 plot/manifest取得・manual parity判定から再開し、再pushしない。
- Stage 0 version 1はCOMPLETE。notebook全体は約304秒、audit runtimeは36.548秒。synthetic DAG/DP、773 raw/candidate wells、3,783,989 candidate rows、cache decompressed SHA `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350` のpreflightはPASS。
- 必要なStage 0生成物だけを取得した。12 wells × first/middle/last = 36 PNG、real/shuffled paired manifest 72 rows。manifest SHA `31146a646eb41686ba8c287f81ecf0c62fa21c815271239ab9f69f58490da2f3`、preview manifest SHA `3ce88a67a71e619008f938343a9ff972f5e2b67b2342e43f54a20143286c80cb`、summary SHA `59fb68e6290fbcc6bd8dc7f5e67b13ca1d2d92939355e5db28978308bc2d4ae9`。
- manual parity: PASS。全plotがMD 256 ft、64 columns、horizontal/typewell 4 ft grid、129 states。MD/TVT方向、flat-Z center、first anchored / later spanning、candidate/truth overlayを目視確認した。real/shuffled 36 pairでhorizontal/typewell support SHA、source SHA、source/sink state count、candidate plot、prior、path existenceが一致。no-padding duplicate、target-conditioned crop、segment historyの兆候なし。
- high-GR-missing / flat-GRではpreview pathなしが多いが、両surfaceで同じsupport topologyから同じ不成立になっておりcontract parity上は正常。性能・coverage判定はStage 1 guardへ委ね、科学設定は変更しない。
- Stage 0 gate通過により、`active_mode=stage1_full_audit`、`manual_parity_confirmed=true`、`enabled_after_stage0_confirmation=true`だけを変更した。

### 2026-07-15 Stage 1 push前ガード

- active diagnostic surfaces: 2 (`real_gr`, `shuffled_typewell_gr`)。
- LightGBM / CNN / HMM config: 0 / 0 / 0、fold: 0、booster: 0。
- PF/Beam/likPF再生成: 0、parent/control再学習: なし。
- 773 wells、single process、Kaggle CPU、GPU/internet disabled。
- segment 256 / stride 128 / bin 4 / typewell ±256 / state 4 / 129 states / anchor ±8 / slack 0.25はStage 0から不変。
- candidate変更、hard prune、window統合、raw-test inference、submissionは行わない。
- Stage 1 canonical packageを同じkernel ID/titleで再prepareし、strict validation PASS。source / loose package / bootstrap configはbyte-identical、Stage 1 config SHA256は`4d1407c66753d2255d34547f0523a97e5a87039640d2b742c8a32a6603b715d5`。
- Stage 1 version 2を同じcanonical kernelへ正常push。push直後のmetadata pull: PASS、status: RUNNING。

### 2026-07-15 Stage 1完了・採否判定

- canonical kernel `kentookumura/exp250-seglocal-negspace-gr-corridor-audit-train` version 2は`COMPLETE`。773 wells / 3,783,989 candidate rows、candidate-segment sample 145,855、評価row weight 3,652,581を処理した。
- Stage 1 audit runtimeは7,633.823秒、notebook全体は約7,898.7秒。2 surfaces / 0 model config / 0 fold / 0 booster / PF・Beam再生成0 / parent-control再学習なしで契約どおり完了した。
- primary real pooled AUC 0.530134、shuffled AUC 0.494199、real-shuffled差+0.035934。q90 thresholdは1.0へ飽和し、bad-rate lift 0.776971、good false-alert 0.232020だった。
- 8 guard中PASSは`pooled_real_minus_shuffled_auc`と`truth_corridor_coverage_real_vs_shuffled`の2件。pooled AUC、family q90 lift/control差、good false-alert、overlap、hidden-like、by-well false-alertの6件はFAIL。
- family real AUCは`beam_mean` 0.516251、`hyb` 0.524232、`likpf_mean` 0.507163、`pf_ancc` 0.518633、`sc_ens` 0.521765。q90 liftは0.997198–1.078168、good false-alertは0.095876–0.262491で、family単位にも採用水準へ届かなかった。
- overlapはprimary path差median 57.61 ft / p90 258.684 ft、risk Spearman 0.448723。hidden-like AUCはspatial 0.531044、typewell-purged 0.532323。by-well good false-alertはp95 0.757381、max 0.984733だった。
- truth corridor coverageのreal-shuffled差はoverall +0.059580、1000+ +0.061222、hidden-like +0.047406 / +0.048384でPASSした。GR topology固有の弱い情報はあるが、candidate誤りの安定なrisk signalではない。
- distance 0–50 / 50–100 ftはreal AUC 0.820631 / 0.819121、shuffled AUC 0.775910 / 0.771389、q90 lift 2.042545 / 2.002634。一方、評価weight 2,950,940を占める1000+はreal AUC 0.515575、q90 lift 0.785417。near signalはdistance/base-error構造の交絡を含むと判断した。
- worst `d07aed8f`はAUC 0.005368、good false-alert 0.984733。well安定性guardの大幅な不合格を確認した。
- logsには定数配列に対するNumPy/Pandas correlation warningが反復したが、処理と成果物保存は完了し、undefined correlationは集計時に有限pairへ限定されている。科学設定を変える再実行は行わない。
- output archive全体は取得せず、採否と記録に必要なsummary / candidate / group / overlap / by-well / metricsだけを取得した。
- output SHA:
  - summary raw: `79448ad77882e3ecf6fd95e0044b59965ffe8cc7fa00603459be3e786824a396`
  - metrics raw: `ff873a02e3d0b0cc38265da04116b9d7f5392355a29e199764ef7b6470df915f`
  - candidate metrics raw: `540e8d22fed5ddbce9f75776585bed9e0e1b1432e11b54343e266d4540e83283`
  - group metrics raw: `333c3c2a869db976ca0fc42741897bb50cdad42a91fd682e40c1996626fe8d96`
  - overlap metrics raw: `bf5b0e0f38b12c248bdd4a2a885b3d187f0ea041fdc23de952964ef88de316ff`
  - by-well raw: `14e226cb05072b3195306b92b7587efbe6300fe6cb1da8e7b5db3aff1a4973b8`
  - segment metrics decompressed: `988c80cc66326b3b19e3307ab6a80d27b6549053fea4adb83d98067014232a27`
  - candidate-segment metrics decompressed: `0fd6241b27fdd778e202d9f81df859fdad1dd228eeea872630261c5a30a48a9a`
- 判定は`fail_close_segment_local_hard_use_and_grid_search`。candidate hard use、threshold/slack/segment grid、direct path、downstream feature化、raw-test inference、submissionを閉じる。

### 2026-07-14 分離・作成

    make new-steering EXP=exp250_segment_local_negative_space_gr_corridor_audit
    make new-exp EXP=exp250_segment_local_negative_space_gr_corridor_audit SOURCE=experiments/exp246_negative_space_gr_barrier_audit

- ユーザー指示により、誤って結び付いていたexp249から現行contractを分離した。
- .steering/20260714-exp250-segment-local-negative-space-gr-corridor-audit/に現行contract、Stage 0/1、guard、再現性を記録した。

### 2026-07-14 実装

- compact self-contained Jupytext trainに、raw inventory / cache preflight、well-wide normalization、MD segment、stable shuffled control、DAG DP、corridor、candidate/truth readoutを実装。
- Stage 0にraw-only 12-well選択、first/middle/last plot、real/shuffled parity、plot/source SHAを実装。
- Stage 1にsegment/candidate/group/overlap/by-well生成物、weighted row-label AUC、q90 lift/false-alert、8 guardを実装。
- inference notebookはtrain-side-only guardで停止し、submission.csvを作らない。

### 2026-07-14 静的検証・Kaggle package

    .venv/bin/python -m py_compile experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_train.py experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_inference.py
    .venv/bin/ruff check experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_train.py experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_inference.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp250_segment_local_negative_space_gr_corridor_audit/exp250_segment_local_negative_space_gr_corridor_audit_inference.py
    make validate-exp EXP=exp250_segment_local_negative_space_gr_corridor_audit
    make validate-template
    make prepare-kaggle-notebooks EXP=exp250_segment_local_negative_space_gr_corridor_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp250-segment-local-negative-space-gr-corridor-audit-train --title 'exp250 segment local negative space gr corridor audit train' --run-on-push --strict"

- synthetic DAG/DP contract: PASS。
- py_compile / Ruff / Jupytext test: PASS。
- strict experiment validation / project template validation: PASS。
- canonical package: kentookumura/exp250-segment-local-negative-space-gr-corridor-audit-train。
- metadata: CPU、internet disabled、run-on-push、competition source 1、kernel sources exp072/exp115の2本。
- package config: active mode stage0_preview、256/128/4/±256/0.25、2 surfaces、0 booster。
- bootstrap ZIP内config、loose package config、source configはbyte-identical。config SHA256は5562a968e997463a61504f15d29a6fae0e2d4790cdbeb66fee2749077d7c6611。
- Kaggle pushとnotebook executionはこの実装ターンでは行っていない。

## Notebook構成比較

- 親exp246 train: 10章 / 1,454行。
- exp250 train source: 11章。入力preflight、segment grid、minimum-bottleneck DP、Stage 0 plot、Stage 1 guardをnotebook cell上に展開した。
- 同一exp helper importに依存せず、正規notebookは薄いmain()呼出ではない。
- notebook sourceでは__file__を使わない。

## 完了した実行順

1. canonical Kaggle CPU Stage 0を実行。
2. PNGでMD/TVT軸、4 ft grid、flat-Z center、real/shuffled support/source parity、candidate/truth overlayを手動確認してPASS。
3. 科学設定を変えずStage 1 gateだけを有効化。
4. 同一固定設定で773-well Stage 1を実行し、8 guardを判定。

## 次のアクション

本branchは不採用として完了。parameter grid、hard prune、downstream feature化、raw-test inference、submitは行わない。再訪する場合だけ、保存済みcandidate-segment成果物で0–100 ftの見かけのsignalがdistance-conditioned base rateで説明されるかを切り分ける低優先readoutとし、新規corridor計算は行わない。
