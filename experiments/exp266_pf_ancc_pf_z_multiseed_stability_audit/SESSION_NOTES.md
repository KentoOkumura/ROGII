# exp266 セッションノート

## 目的

単一seedのPF ANCC / PF-Zが`11d0f5ac`と同型wellで偶然良かった可能性を、全well multiseed分布で検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: backlog / steering作成済み、実装中
- Kaggle CPU train: 未実行
- inference / submission: disabled

## 実行前コスト契約

- active PF dynamics variants: 2（PF ANCC、PF-Z）
- particles × seeds: 各600 × 64（元seed 1 + 新規63）
- 対象: 3,783,989 rows / 773 wells
- LightGBM config / fold / booster: 0 / 0 / 0
- parent/control retraining: なし
- GPU / inference / submission: なし / なし / なし
- Kaggle CPU notebook: 1本
- 予想runtime: 約5.5〜7時間。exp106のPF-Z 64 seed / 773 wells実測10,111.57秒を根拠とする。
- 2026-07-17にユーザー承認済み。

## 再現性

- `docs/06_reproducibility.md`を確認済み。
- seed 0は`stable_seed("pf_ancc", well)` / `stable_seed("pf_z", well)`。
- seed 1〜63は`stable_seed(exp266, "train", algorithm, well, seed_index)`。
- well-level thread並列の前に各seedをimmutable keyで固定し、Numba kernelへ明示seedを渡す。
- seed 0の全行PF ANCC / PF-Z exact parityが通るまで追加seed結果を採用しない。
- gzipはdecompressed content SHAを主証拠にする。
- train-side diagnosticのみでmodel / prediction submission / submission SHAは対象外。

## 変更点

- exp072のPF ANCC / PF-Z kernel、dtype、600 particles、全parameterを固定する。
- seed反復と事前固定したmean / median / 10% trimmed mean readoutだけを追加する。
- `11d0f5ac`、strong phenotype、その他wellの比較は全seed path凍結後に行う。

## 実装

- 12章 / 1,700行超のcompact self-contained Jupytext trainを実装した。
- notebook内にconfig/input解決、SHA guard、exp072 exact PF kernel、parity phase、multiseed phase、
  seed分布・nested aggregation・発生条件readout、生成物保存を展開した。
- 同じ実験ディレクトリのhelper importと`__file__`は使っていない。
- 全seed全row tensorを全well分保存せず、well処理中にseed別metrics、固定集約、strong-well path quantileへ縮約する。
- inference notebookはdisabled guardだけを持ち、`submission.csv`を生成しない。
- 親exp243正規trainは238行 / 6章で重いhelperへ委譲する構成。本実験はPF kernelと上位orchestrationを
  self-contained化したため約1,700行 / 12章となり、input、実行対象、保存先をnotebookだけで追える。

## 静的検証

    .venv/bin/python -m py_compile experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/exp266_pf_ancc_pf_z_multiseed_stability_audit_train.py experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/exp266_pf_ancc_pf_z_multiseed_stability_audit_inference.py
    .venv/bin/ruff check experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/tests/test_exp266_multiseed_stability_contract.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/exp266_pf_ancc_pf_z_multiseed_stability_audit_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/exp266_pf_ancc_pf_z_multiseed_stability_audit_inference.py
    .venv/bin/pytest -q experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/tests/test_exp266_multiseed_stability_contract.py
    make validate-exp EXP=exp266_pf_ancc_pf_z_multiseed_stability_audit
    make validate-template

- py_compile / Ruff / Jupytext train+inference round-trip / strict exp validation / template validation: PASS。
- exp266 contract test 3件: PASS。
- local venvにNumbaがないためtestではdecoratorをidentity shimに置き換え、合成pathでexp072と
  PF ANCC / PF-Zのprediction・particle stdが両方exact一致することを確認した。
- 実データのローカルnotebook実行は行っていない。最初のfull実行はKaggle CPUを正とする。
- exp209 enriched HMM cacheのraw/decompressed SHAとexp226 OOFのdecompressed SHAをローカル保存物で
  再計算し、configの期待値と一致した。

## コマンドログ

### 2026-07-17 作成

    make new-steering EXP=exp266_pf_ancc_pf_z_multiseed_stability_audit
    make new-exp EXP=exp266_pf_ancc_pf_z_multiseed_stability_audit

- steering: `docs/legacy/steering/20260717-exp266-pf-ancc-pf-z-multiseed-stability-audit/`
- `KAGGLE_DIRECTION.md`未着手バックログへ高優先・実装中として追加した。

## 次のアクション

1. Kaggle package/bootstrap SHAを確認する。
2. 承認済みKaggle CPU trainをpushし、同一versionを完了まで監視する。
3. 完了後にseed分布と発生条件を解析して全記録を更新する。

## Kaggle package / push

- 初回packageは51文字slug
  `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-audit-train`で作成した。
- config / train source / settingsは正規、loose package、bootstrap ZIPでSHA一致:
  - config: `aa8eaf3f8a14b74a7efdbf83bb537a3fc492dc3cda579054be9575a75d96a342`
  - train source: `3235f345a20cc48dd23b243e3c58e735de313e6f846f5e48f317e151ca99a060`
  - settings: `761a769732e7b3d82676053de1ad72076faaad2cfa714cfdb33959795330d478`
- metadataはprivate CPU、GPU/TPU/internet off、competition source 1、kernel source 3。
- 51文字slugのpushは初回が空response parse error、再試行が`SaveKernel 400`。pullは403、
  自分のkernel一覧は`Not found`でresource未作成を確認した。3つのkernel sourceは全てpull成功。
- Kaggle slug上限の可能性を避け、同じexp266のまま45文字のcanonical ID
  `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`へ短縮した。科学設定・source SHAは不変。
- 短縮ID version 1を正常push。URL:
  `https://www.kaggle.com/code/kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
- pull成功、id_no `127531300`。反映metadataはprivate CPU、GPU/TPU/internet off、
  competition source 1、kernel source 3。初期statusは`KernelWorkerStatus.RUNNING`、実行中logsは空。

## Kaggle train v1失敗診断

- kernel: `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
- version / id_no: 1 / `127531300`
- `kaggle kernels logs`でERRORを確認。約314.16秒、notebook `In [9]`の
  `load_reference_surface` coverage guardで停止した。
- 実測は3,783,989 rows / 776 wells、期待は3,783,989 rows / 773 wells。
- seed 0 parity phaseやmultiseed PF phaseには到達していない。
- exp072 inputのraw SHAは期待値
  `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`と一致しており、
  input version違いではない。
- 同じSHAのローカルファイルで再現すると、pandas推定読み込みは`well`を
  `str` 3,747,125件、`float` 20,480件、`int` 16,384件に分けた。事後`astype(str)`は778 unique、
  parse時`dtype={"id": "string", "well": "string"}`固定は773 uniqueとなった。
- 推定読み込みで生じた不正表現には`9053135`（先頭0消失）、`54319800000000.0`、
  `4.4441000000000005e+58`、`inf`が含まれた。Kaggle側はpandas/chunk境界差で776 uniqueとなった。

## v2修正

- exp072 `id` / `well`、exp209 `id`、exp226 `well_id`を`pd.read_csv`時点からstring dtypeへ固定。
- exp072 `id`のwell prefixと`well`列が全行一致するfail-closed guardを追加。
- 先頭0を持つ数字型well ID `01234567`をbase/HMM/exp226 join後も保持する回帰testを追加。
- py_compile / Ruff / contract tests 4件: PASS。
- 科学設定、PF kernel、seed契約、2 variants × 64 seeds × 600 particles、CPU契約は変更なし。

## Kaggle v2 package / push前確認

- kernel ID / title:
  - `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
  - `exp266 pf ancc pf z multiseed stability train`
- private CPU、GPU/TPU/internet off、competition source 1、kernel source 3、run-on-push on。
- 実行対象はPF dynamics 2 variants、各64 seeds × 600 particles。LightGBM config / fold / boosterは
  0 / 0 / 0、親control再学習なし、inference/submissionなし。
- canonical notebook cellsとpackage notebookのbootstrap後cellsは一致。
- canonical / loose package / bootstrap ZIPのSHA一致:
  - config: `e4ae09023a1973fd0a24145ea8ebe0cfc5429db8ffef410d5f79f2afae948e09`
  - train source: `1cb9e0f460982d06be827497b537a53186d1c50f4f3afcc72cf0a485b8c70418`
  - settings: `761a769732e7b3d82676053de1ad72076faaad2cfa714cfdb33959795330d478`

## Kaggle train v2 push

- `kaggle kernels push -p experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/kaggle/train`
  により`Kernel version 2 successfully pushed`。
- URL: `https://www.kaggle.com/code/kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
- latest pullで同じid_no `127531300`、private CPU、GPU/TPU/internet off、kernel source 3、
  competition source 1を確認した。
- pullしたnotebookはstring dtype guardと`id` / `well` identity guardを含む。
- run-on-pushで再実行を開始。ユーザーの完了連絡まで常時監視は停止する。

## Kaggle train v2失敗診断

- version / id_no: 2 / `127531300`
- 約286.76秒で参照3,783,989 rows / 773 wells、strict 108 / strong 53を確認。
- 約507.10秒でseed 0 parity summaryを出力し、約508.34秒でfail-closed。
- PF ANCC: mean absolute diff 0.000239、RMSE diff 0.000279、max absolute diff 0.000484、
  nonzero 3,725,183 rows。
- PF-Z: mean absolute diff 0.000239、RMSE diff 0.000279、max absolute diff 0.000484、
  nonzero 3,725,279 rows。
- multiseed phaseには到達していない。

原因:

- 親exp072の`run_pf_ancc` / `run_pf_z`はseeded kernelのfloat64内部出力をfloat32へcastして返す。
- exp266の`run_original_seed_task`とmultiseed path配列もfloat32であり、kernel/seed経路は親と一致する。
- exp072 feature cacheの`pf_ancc` / `pf_z`はfloat32由来だが、CSVには短い10進文字列で保存される。
  exp266はこれをfloat64でparseし、再生成float32値をfloat64へ拡張して差を取ったため、最大float32半ULP相当
  0.000484 ftを不一致として検出した。
- 既存exp106 v3 exact parityは参照候補列をfloat32へcastしてから比較し、max diff 0.0を達成している。

v3修正:

- exp072 CSV読み込み時に`pf_ancc` / `pf_z`を元のfloat32 dtypeへ復元。
- 両参照PF列がfloat32であることをfail-closed guardで確認。
- 数字型well ID保持testに非表現可能なPF小数のCSV round-tripを加え、期待float32 bit値との一致を確認。
- 科学設定、PF kernel、seed契約、2 variants × 64 seeds × 600 particles、CPU契約は変更なし。

## Kaggle v3 package / push前確認

- kernel ID / titleはv1/v2と同じ:
  - `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
  - `exp266 pf ancc pf z multiseed stability train`
- private CPU、GPU/TPU/internet off、competition source 1、kernel source 3、run-on-push on。
- PF dynamics 2 variants、各64 seeds × 600 particles。LightGBM config / fold / boosterは0 / 0 / 0、
  親control再学習なし、inference/submissionなし。
- canonical notebook cellsとpackage notebookのbootstrap後cellsは一致。
- canonical / loose package / bootstrap ZIPのSHA一致:
  - config: `11e4b8e588c7b431d854695a43bb9f1266a976ee16c136cc080b59f643ee1b0c`
  - train source: `f5dfa0deb61025d8345e4d977a22ddae4b0cbc34df76d96bd394b88007da042f`
  - settings: `761a769732e7b3d82676053de1ad72076faaad2cfa714cfdb33959795330d478`

## Kaggle train v3 push

- `kaggle kernels push -p experiments/exp266_pf_ancc_pf_z_multiseed_stability_audit/kaggle/train`
  により`Kernel version 3 successfully pushed`。
- URL: `https://www.kaggle.com/code/kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`
- latest pullで同じid_no `127531300`、private CPU、GPU/TPU/internet off、kernel source 3、
  competition source 1を確認した。
- pullしたnotebookにexp072 PF列のfloat32 parse、float32 dtype fail-closed guard、well identity guardが反映済み。
- run-on-pushで再実行を開始。ユーザーの完了連絡まで常時監視は停止する。

## Kaggle train v3完了監査

- ユーザーの完了連絡後、同じcanonical kernel version 3 / id_no `127531300`の成果物を取得した。
- kernel: `kentookumura/exp266-pf-ancc-pf-z-multiseed-stability-train`。
- status: `completed_train_side_multiseed_stability_audit`。
- 実行量は承認契約どおりPF dynamics 2 variants、各600 particles × 64 seeds、LightGBM 0 config、
  fold 0、booster 0、GPU/親再学習/inference/submissionなし。
- 3,783,989 rows / 773 wells、strict 108 wells、strong 53 wells。
- runtimeは合計12,482.144秒（3時間28分02秒）、parity 218.579秒、multiseed 11,958.578秒、
  8 CPU workers。
- seed 0 parityはPF ANCC / PF-Zともmean absolute diff、RMSE diff、max absolute diffが全て0、
  nonzero rows 0で全行exact一致した。

## output取得とartifact検証

- outputは`/tmp/kaggle-output/exp266_pf_ancc_pf_z_multiseed_stability_audit/train_v3`へ取得した。
- Kaggle CDNの一時的な接続切断で`detailed_strong_paths.csv.gz`だけ通常取得が中断したため、同一URLへ
  byte rangeを固定して再開し、17,274,817 bytesのraw SHAとdecompressed SHAをmanifestへ照合した。
- `artifact_manifest.csv`の必須12 filesすべてについてbytes、raw SHA、decompressed SHAを再計算しPASS。
- schema/shape contract:
  - `reference_by_well.csv`: 773 × 15。
  - `seed_by_well.csv.gz`: 98,944 × 23。各well × algorithmが64 unique seeds、index 0..63。
  - `aggregate_by_well.csv.gz`: 27,828 × 22。
  - `well_stability_summary.csv`: 1,546 × 53。
  - `detailed_strong_paths.csv.gz`: 494,204 × 15。
  - `input_manifest.csv`: headerを除き1,549 rows（参照3 + raw 2 files × 773 wells）。
- seed / aggregate primary metricsは全finite、root `metrics.json`とartifact `summary.json`はexact equality。
- input manifest SHA:
  `7ae1df7457b9ef0bd454d1a0a3620a62a1d363ecb5b45bc255fcf594ddc323d0`。
- artifact manifest SHA:
  `440a5e474b19290d667562c242ff7aec73420f6340e712d34d01a89bf45cd69c`。
- Kaggle実行時config SHA:
  `11e4b8e588c7b431d854695a43bb9f1266a976ee16c136cc080b59f643ee1b0c`。
- 完了後のroot `config.yaml`はstatusを更新したためKaggle package snapshotとは意図的に異なる。
  科学設定の再現には上記v3 config SHAとKaggle packageを正とする。
- 数字だけのwell IDを壊さないため、ローカル事後解析でもCSV読み込み時に`well`をstring dtypeへ固定した。

## `11d0f5ac`結果

- 比較RMSE: PF ANCC 2.382190、PF-Z 3.386312、exp226 2.874875、HMM 21.161156、
  likelihood-PF 24.438100。
- PF ANCC:
  - 元seed RMSE 2.382115、lower-tail percentile 0.507937で新規seed分布のほぼ中央値。
  - 新規seedRMSE mean/std 2.509247 / 0.719886、q10/median/q90 2.333788 / 2.379021 / 2.475144。
  - RMSE 5 ft以下62 / 63、Wilson下限0.915415。exp226勝率0.952381。
  - HMM / likelihood-PF勝率、strong margin再現率は全て1.0。終端符号一致0.984127。
- PF-Z:
  - 元seed RMSE 3.386387、lower-tail percentile 0.952381。元seedは良すぎず、むしろ悪い側のtail。
  - 新規seedRMSE mean/std 2.522810 / 0.625728、q10/median/q90 1.785696 / 2.725870 / 3.316659。
  - RMSE 5 ft以下63 / 63、Wilson下限0.942529。exp226勝率0.682540。
  - HMM / likelihood-PF勝率、strong margin再現率、終端符号一致は全て1.0。
- `>1000 ft`の64 seed mean path RMSEはPF ANCC 1.5700、PF-Z 1.9905。long tailまで優位を維持した。
- 判断: `11d0f5ac`の優位は単一seedの偶然ではない。PF ANCCは典型seed、PF-Zは元seedより新規seedの方が良い。

## 他wellの再現性とselection bias

- 元seedstrong 53 wellsの新規seed strong margin再現:
  - PF ANCC: 過半数38、80%以上20、90%以上17、全seed10。
  - PF-Z: 過半数27、80%以上23、90%以上20、全seed13。
  - 両手法: 過半数21、80%以上11、90%以上9、全seed4。
- 両手法で全seed再現した4 wells:
  `11d0f5ac`、`bb682ebd`、`f0188a48`、`fb0904bd`。
- 両手法で80%以上のseedがRMSE 5 ft以下だったのは`11d0f5ac`、`fb0904bd`の2 wells。
- RMSE 10 ft以下まで広げると9 wells:
  `11d0f5ac`、`1fee2b62`、`521a7819`、`647f2a41`、`833af382`、`94467f50`、
  `dd6bc5b2`、`dd7d638e`、`fb0904bd`。
- PF ANCC strong groupは元seed lower-tail percentile平均0.370 / median0.317で、nonstrongの
  0.497 / 0.492より元seedが良い側へ偏る。下位10 percentile以内はstrong 10 / 53、nonstrong 84 / 720。
- PF ANCC strong groupは元seedRMSE平均9.135に対し新規seed中央値平均12.075、64 seed meanが元seedを
  改善したのは22 / 53。元seedstrongによる抽出にはselection-on-seedが含まれる。
- したがってphenotypeは他wellにも存在するが異質であり、`11d0f5ac`は特に安定した稀な例である。

## 発生条件

- raw / target-free特徴で明確な単一triggerは得られなかった。
- eval rowsと新規seedRMSE平均のSpearmanはPF ANCC 0.218980、PF-Z 0.225526。最長tail quintileでは
  新規seed中央値RMSE平均が12.952 / 16.684、seed RMSE std平均が4.040 / 2.476へ上がった。
- 一方、eval rowsとstrong再現率は-0.061907 / -0.102533で弱く、strong groupのeval rows中央値4,657は
  nonstrong 4,847.5より長くない。長いtailは一般的不安定化条件だがstrong membershipの条件ではない。
- known rows、typewell range、GR sigma、初期rate、PF-Z beta/intercept/sigmaは大半が|rho| < 0.11。
  PF-Z終端符号一致と`pf_z_sigma`は-0.198186だが単独gateには弱い。
- strong再現率とHMM RMSEのSpearmanはPF ANCC 0.549116 / PF-Z 0.460048、likelihood-PF RMSEでは
  0.471318 / 0.409675。strong現象はraw geometryの単純regimeより、HMM/likelihood-PFが外れるrelative-method
  regimeとPF軌道のlong-tail追従が重なることで説明しやすい。

## seed集約と次の候補

- 全well pooled RMSEのmean集約収束:
  - PF ANCC seed 1/4/8/16/32/64 = 14.493051 / 13.126896 / 13.027107 / 12.924443 /
    12.885644 / 12.830319。
  - PF-Z = 17.788171 / 17.178861 / 17.201465 / 17.153356 / 17.124414 / 17.074522。
- 64 seedではmeanがmedian / 10% trimmed meanより良い。PF ANCCは4 seed meanだけで64 seed改善量の約82%を回収。
- ただしPF-Zは既存exp106/104のseedbag系より弱いため再portしない。
- 次候補はPF ANCC固定4/8 seed meanをtarget-free低コストcandidateとして再生成し、既存bankとのunique oracle、
  distance/hidden-like/worst-well、current-test runtimeを学習前に監査する。単一seed hard gateや直接submissionには進めない。

## 完了判断

- exp266はtrain-side stability auditとして完了。
- `11d0f5ac`の単一seed偶然仮説は棄却。
- strong 53 wells全体への無条件一般化、元seedphenotype gate、inference、submissionは不採用。
- 実装済みbacklogを`KAGGLE_DIRECTION.md`から削除し、少数seed PF ANCC mean候補を別の中優先backlogへ整理する。
