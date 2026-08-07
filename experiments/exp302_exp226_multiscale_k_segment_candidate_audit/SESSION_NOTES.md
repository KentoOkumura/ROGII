# exp302_exp226_multiscale_k_segment_candidate_audit セッションノート

## 目的

exp226の`k_segments`だけをK=12/K=24へ変更し、保存済みK16 controlに対するdirect qualityと、
exp293 fixed deployable12に対するadd-one candidate noveltyを監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU version 2完了・technical PASS・direct FAIL・candidate novelty PASS
- CV: K12 `9.5519380655` / K24 `9.4132443152` / saved K16 `9.4271095966`
- LB: なし
- 実装承認: あり（2026-07-20のユーザー依頼`exp302を実装してください`）
- Kaggle実行承認: あり（2026-07-20のユーザー依頼`実行してください`）
- inference/submission: 対象外

## 2026-07-20 設計確定

- `kaggle-strategy`で、exp295/exp276を優先し、本件を中低優先のCPU候補とした。
- `kaggle-review-exp`の手順でsteeringを先に作り、空テンプレートからexperiment scaffoldを作成した。
- 親exp226のコードやNotebookはコピーしていない。
- fixed variants: `K=12`, `K=24`。
- saved control: `K=16`。再生成禁止。
- direct PASSとcandidate novelty PASSを独立に固定した。
- candidate novelty PASSだけをexp303開始条件とした。
- HMM `step=0.35, n_rates=41`を含む他パラメータ探索は追加していない。

## 実行量

| 項目 | 数 |
| --- | ---: |
| scientific variants | 2 |
| outer folds | 5 |
| variant-fold runs | 10 |
| LightGBM configs | 0 |
| trained folds | 0 |
| boosters | 0 |
| parent/control regeneration | 0 |
| GPU | 0 |

## 2026-07-20 実装セッション

- ユーザー依頼を実装承認として記録した。Kaggle実行、正規Notebook採用、inference、submissionの承認には拡張していない。
- Jupytext percent形式のcompact self-contained train sourceを別名で実装した。
- exp226からtrainに必要な数値核だけをNotebookへ持ち込み、同一exp helper importと`__file__`依存をなくした。
- `K=12` / `K=24`以外を拒否し、exp226の残り全パラメータがdataclass既定値と一致しない場合に停止するcontract guardを入れた。
- outer-train donor objectだけへTVT/residual/segment coefficientを接続し、outer-valid objectはそれら4属性を`None`のまま保つ。
- 保存済みK16 OOFはallowlist 5列だけを読み、decompressed SHA、exp293 bank K16との最大差、fold/row identityを照合する。
- exp263 manifestからexp293 fixed deployable12を同じfloat32演算順で再構成し、candidate bank content SHA
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`を要求する。
- H128/H256/H512/whole-well blockをexp293と同じ規則で再構成し、decompressed SHA
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`を要求する。
- K12/K24 prediction、foldwise kappa、raw input manifest、candidate bank、block assignmentをfreezeしてから、別loaderで評価用suffix TVTを接続する。
- direct pooled/fold/distance/hidden-like/by-wellと、各variantを別々にfixed12へ追加するrow/H128/H256/H512/whole-well oracle、
  strict unique-best、fold、prediction correlation、by-well readoutを実装した。K12+K24同時追加とoracle prediction保存は行わない。
- fail-closed inference候補を別名で実装し、raw test読込とsubmission生成を禁止した。
- 正規train/inference placeholderは上書きしていない。

### 実装検証

- compact train: 3,100行超、9章、19 cells（Markdown 10 / code 9）、output 0、execution countなし。
- 構成比較: 親exp226通常train source 111行 / exp293 compact self-contained train 1,963行に対し、
  exp302はexp226数値核とexp293 bank/readoutをNotebook上で追える構成にした。
- `py_compile`: train / inference / settings / testをPASS。
- `ruff`: compact train / inference / 専用testをPASS。
- 専用test 12件 + Kaggle notebook共通test 4件をPASS。
- Jupytext: train / inference候補の生成と`--to ipynb --test`をPASS。
- `make validate-exp EXP=exp302_exp226_multiscale_k_segment_candidate_audit`: strict PASS。
- `make validate-template`: PASS。
- full suite: 378件中375 PASS / 1 skip / 2 FAIL。FAILは既存exp296の完了済みconfig
  （`completed_train_side_guard_failed_closed`、`run_variant=false`）と旧testの実行中期待
  （status `kaggle_cpu_*`、`run_variant=true`）の不一致で、exp302変更外。exp296は変更していない。
- read-only入力preflightでは保存済みexp226 OOFを期待decompressed SHAで解決できた。exp263 full partitionは
  ローカル未配置でKaggle kernel sourceから供給する契約のため、ローカルfull auditは実行していない。
- ローカルNotebook実行、Kaggle package作成、push、artifact/prediction生成は行っていない。

## 2026-07-20 Kaggle実行セッション

- ユーザーの`実行してください`を、正規train Notebookへのcompact版採用と、固定contractによるKaggle CPU train-side audit 1回の明示承認として記録した。
- push前の計算量を再確認した: scientific variants 2（K12/K24）、outer folds 5、variant-fold runs 10、LightGBM configs 0、trained folds 0、boosters 0、K16 control再生成 0、GPU 0。
- inferenceとsubmissionは承認範囲外のため実行しない。
- 初回のfull-name slug `exp302-exp226-multiscale-k-segment-candidate-audit-train`はKaggle `SaveKernel 400`で拒否され、kernel実行は作成されなかった。既知の長slug制約として、科学contractを変えず短縮slug `exp302-ksegment-candidate-audit-train`へ切り替える。
- 短縮slugにfull titleを組み合わせた再pushも、Kaggleからtitle/slug正規化不一致を明示され`SaveKernel 400`で実行未作成となった。titleも`exp302 ksegment candidate audit train`へ揃えて再prepareする。
- version 1は約2分で`ValueError: saved control fold identity mismatch`により停止した。variant生成前のためK12/K24の10 runsは開始されていない。
- 原因はexp226の評価foldとexp263 candidate-bank provenance foldを同一列として比較した実装バグ。ローカルのexp263 K16 fold0 sample 757,738行では、exp226とのfold一致率は13.7051%だがprediction最大差は0.000488281 ftで、固定parity許容0.001 ft内だった。
- configで既に固定した`fold_source=exp226_saved_oof_well_fold_identity`に従い、exp226 foldをK12/K24生成・direct guard・novelty fold guardへ使う。exp263 `outer_fold`はcandidate bank content SHAとexp293 block assignment SHAのprovenance専用として保持する。candidate値、block ID、truth freeze順序、PASS条件、計算量は変更しない。

## 2026-07-21 完了確認と結果

- ユーザーから完了連絡を受け、同じkernel `kentookumura/exp302-ksegment-candidate-audit-train`のstatus `COMPLETE`とversion 2 logsを確認した。id_noは`128010921`。
- audit完了は`1281.068 sec`、最終logは`1291.656 sec`。2 variants × 5 foldsの10 runsを完走し、LightGBM/booster、control再生成、GPU、inference、submissionはすべて0。
- technical guardは全PASS。3,783,989 rows / 773 wells、finite coverage 1.0、outer-valid truth state 0、evaluation truth access before freeze 0だった。
- direct guardは両variant FAIL:
  - K12 RMSE `9.5519380655`、K16比`+0.1248284689 ft`、改善0/5 folds。
  - K24 RMSE `9.4132443152`、K16比`-0.0138652814 ft`、改善3/5 folds。1000+、hidden-like 2面、p95/worstはPASSしたが、pooled閾値`9.3771096741`と4/5 foldsを満たさない。
- candidate novelty guardは両variant PASS:
  - K12 H512 / whole-well oracle改善`+0.0660951065 / +0.0684655672 ft`、H512 strict unique-best `10.697316%`、改善5/5 folds。
  - K24 H512 / whole-well oracle改善`+0.0839013329 / +0.0662307649 ft`、H512 strict unique-best `10.889945%`、改善5/5 folds。
- direct候補昇格はなし。exp302側のexp303 dependencyは充足したが、別条件のexp276完了+promotion guard FAILは未充足。
- 一括output取得はK24 OOF開始時に進捗停止したため中断し、`--file-pattern`でsmall artifactとK24 OOFを個別取得した。Kaggle実行結果には影響していない。
- `/tmp/kaggle-output/exp302_exp226_multiscale_k_segment_candidate_audit/train_v2`と`train_v2_k24_retry`でmanifest 16/16件のfile SHAを照合した。K12/K24/blockのgzipはdecompressed SHAも一致し、K12/K24 OOFは各3,783,989行、schema `id,well,well_row_idx,outer_fold,variant,candidate_tvt`だった。
- 専用testはfold-role regression 1件追加後13件、共通Notebook test 4件と合わせ17件PASS。

## 再現性メモ

- seed policy: exp226のdeterministic sortingとSHA256 fold identityを再利用。
- stochastic components: なし。
- CPU/GPU: Kaggle CPU想定、初回`num_workers=1`、GPUなし。
- input SHA: exp226 OOF decompressed content SHA
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`を要求。
- exp293 bank/block SHA: 実装preflightで保存済みartifactをresolveし、truth前に固定する。
- feature/prediction SHA: K12/K24をvariant別に記録する。
- model/submission SHA: model/submissionを生成しないため対象外。
- deterministic anchor: false。診断OOFだけでsubmission anchorにはしない。
- Kaggle run config SHA: `a7838b30f8c76c05e00f4d7fc1b09f0fd8b3e00c52c382f760742229e88012ba`。
- K12 prediction content / decompressed SHA:
  `c3d7dfe20ad3b8c7d6d5220023bbb4526fb90d10cc73f01e612db847af70da63` /
  `63b381299ee46fa172680af57959d675c68b6b24af05664c8689dd291961f22d`。
- K24 prediction content / decompressed SHA:
  `dca92e8f21d3b8b33d1543fe3df0bf586be3a2604b76ee1bf19fa84a327f06ef` /
  `ca36d168b45acb15cc814ac3c1c3437894cd1050f6c51ba03f5b302efd0a31aa`。
- truth content SHA: `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`。
- SHA manifest file SHA: `ebd31a81be14637171ae991a41e3ff05307cde32f1efa8a2aae6fe33fdd244d4`。

## 未実行

- inference/submission

## 次のアクション

1. exp276の固定再検証をユーザー判断後に完了する。
2. exp276がpromotion guard FAILの場合だけ、exp303の実装・実行判断へ進む。
