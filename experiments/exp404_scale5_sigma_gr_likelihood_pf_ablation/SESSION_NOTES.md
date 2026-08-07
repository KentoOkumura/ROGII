# exp404_scale5_sigma_gr_likelihood_pf_ablation セッションノート

## 目的

likelihood-PFのseed集約をtemperature 5へ固定し、全wellのGR観測scaleを
`gs×1.0`と`gs×1.3`でpaired比較する。設計確定後の追加指示により、
別名compact self-contained候補と専用testまで実装した。その後の実行指示に
より正規Notebookを採用した。version 1は予測freeze後のlate-readout契約不備、
version 2は`.bin` gzipのcompression推論、version 3はpandas間の文字列dtype
表記差でERRORとなった。いずれもtechnical failureで、version 1のfreeze済み
予測を厳密SHA固定したままversion 4を準備した。

## 現在の状態

- Route: `pf_beam`
- 状態: version 4 COMPLETE、technical PASS / scientific FAIL、閉鎖
- scientific parent: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- PF kernel parent: `exp072_exp063_full_replay_feature_cache`
- CV: x1.0 `10.914522073423171` / x1.3 `11.174615008412255`
- LB: まだなし
- steering:
  `.steering/20260726-exp404-scale5-sigma-gr-likelihood-pf-ablation/`
- 正規train / inference Notebook: compact候補を採用済み
- 別名compact self-contained train / inference候補: 実装済み
- 専用test: 12件PASS
- Kaggle kernel: version 4 / id_no `128628818`
- status: `KernelWorkerStatus.COMPLETE`
- URL:
  <https://www.kaggle.com/code/kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train>

## 2026-07-26 Kaggle private CPU train version 1開始

ユーザーの「実行してください」を、正規Notebook採用、private CPU package /
push / train実行の承認として扱った。inferenceとsubmissionは承認範囲外のまま
維持する。

### Push前の実行量再確認

- scientific variants: 2
- PF well-runs: 1,546
- seed-well trajectories: 197,888
- particle starts: 98,944,000
- reporting folds: 5
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- HMM / Beam: `0 / 0`
- 親model/control再学習: 0
- 科学controlのx1.0 PF再生成は凍結済みpaired設計の一部

### 正規Notebookとpackage

- canonical train SHA256:
  `9a26e3d3bb47bdc3bf42a7eb7b5eb183a3c9f7a2e90c4f629cf36524f760f0c5`
- canonical inference SHA256:
  `c374ae613daaa9551381584ff64ceefb30550da47ddd09acab9469ab6c384824`
- package notebook SHA256:
  `d4d0e0a29e2e9cb89edfefff3f9c86430618b4d4f0ecc7516247c4ebd4b21e1e`
- push時点のlocal / loose package / bootstrap `config.yaml` SHA256:
  `50b9770c18d6c0e66d9ea1934052ea49ae1870a7f3f6c4cfedb05f52206a21c3`
- push後のlocal `config.yaml`にはkernel version / id_no / running statusだけを
  追記したため、push済みpackageの上記SHAとは意図的に異なる。
- private / CPU / GPUなし / internet off / `run_on_push`
- competition source 1、kernel sources 4。4 sourceはpush前にpull可能と確認した。
- py_compile、Ruff、Jupytext round-trip、strict validation: PASS
- dedicated + notebook tests: `15 passed`

### Push

初回canonical案
`exp404-scale5-sigma-gr-likelihood-pf-ablation-train`はslug/titleとも51文字で、
Kaggle `SaveKernel`がHTTP 400を返した。認証と4 kernel sourceは別途pullで
正常と確認済みで、初回失敗ではkernelは作成されなかった。

科学的意味を保って`likelihood-pf`だけを`likpf`へ短縮し、slug/titleを
43文字にした。同じpackage内容を再検証後、次をpushした。

- kernel:
  `kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train`
- version: 1
- id_no: `128628818`
- started: `2026-07-26 00:18:54 UTC`
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: 空。worker開始直後のため結果はまだない。

## 2026-07-26 version 1 ERRORとversion 2 technical retry

version 1は`12,467.179 sec`（約3時間27分）後、両variantのPF生成と
prediction freezeを完了した後のlate-readoutで停止した。

```text
ValueError: hidden-like role counts mismatch for hidden_like_spatial
```

原因はexp404 `config.yaml`の`hidden_like_assignment`から、正式なexp115 /
親exp400の`allowed_roles`と`expected_role_counts`だけが実装時に欠落したこと。
コードは空dictを期待値として実role countと比較したため、spatialの最初の
検査で停止した。PF、seed、scale、prediction、truth-late順序の失敗ではない。

version 1 outputから次を回収し、technical recoveryの根拠へ固定した。

- predictions: 3,783,989 rows / 773 wells / 12 columns / 全prediction finite
- audit: 773 wells / status `ok` 773
- PF well-runs / seed-well / particle starts:
  `1,546 / 197,888 / 98,944,000`
- prediction raw gzip SHA:
  `b3699432a691229da5a6562ce74e0b84f1bee3021bd80d650526906f5aa390f8`
- decompressed SHA:
  `00fe1b90fce84bd601b4b91442d9fc698200aafadd48658f7d8c26ec1fbe0d00`
- resume logical SHA:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
- local pandas 3 raw-read schema SHA:
  `2b5a20d2a4bdd214d5135219a83dc7503647cadd418564e647aee0d10e487812`
- pandas-independent normalized schema SHA:
  `2372de95ab154ddd2b1bfe545923233bc41f96104e66c32ab93e6f0fa99bee88`
- well audit raw SHA:
  `bcd0634a760a8c3d6b5da33da878f33408cd37285633eb17498c8bb3cee0a390`
- scientific contract SHA:
  `41c1f95ca8bd7d20eef00f244ced7d4dbc4b3571cc9fd4189c08d6831ef15b57`

正式なrole件数をconfigと実行前contract testへ追加した。

- spatial: `train 573 / valid 200`
- typewell-purged:
  `train 557 / valid 200 / purged_train_excluded 16`

再計算回避のため、回収物をprivate dataset
`kentookumura/exp404-v1-frozen-predictions` version 2へ保存した。Kaggleの
tabular自動展開を避けるためgzip bytesを`.csv.gz.bin`として保持し、Kaggleから
再downloadしたraw SHAが上記と一致することを確認した。

version 2は同じscientific contract SHAを必須とし、raw / decompressed /
logical / schema / audit SHA、行・well・実行量、forbidden late columnsを
すべて検証してからpredictionをfreeze状態へ復元する。version 2自身の新規PFは
`0`、再利用するversion 1 PFは1,546 well-runs。候補値、gate、fold、truth、
temperature、multiplier、seedは変更しない。

- canonical train SHA:
  `ac36ca4310ebc0f87fd5a11ef57aa2f5739d6defa2c6564007c94251fad71d00`
- version 2 package SHA:
  `9d39add09819b9a673b9a8111bbf6dd725aaa7ae89d7869f6aeedff92401a552`
- push時config SHA:
  `4ecd21ec4fae862c17578a14f37a0bce01ace9265a37e14acdd78f1eca3fa0cc`
- dedicated tests: `12 passed`
- dedicated + notebook tests: `16 passed`
- Jupytext / py_compile / Ruff / strict exp / template validation: PASS
- version 2 start: `2026-07-26 11:51:01 UTC`
- initial status: `KernelWorkerStatus.RUNNING`

## 2026-07-26 version 2 / 3 ERRORとversion 4 technical retry

version 2は約149秒で、`.csv.gz.bin`をpandasがsuffixからplain CSVと推論し、
header readの`UnicodeDecodeError`で停止した。予測のfull parse、truth attachment、
metrics計算より前である。header/full readへ`compression="gzip"`を明示し、
同じrecovery bytesでversion 3を開始した。

version 3はraw gzip SHA、decompressed SHA、logical content SHAをすべて通過したが、
schema SHAだけで停止した。ローカルpandas 3の`dtype=str`は`str`、Kaggle pandas 2
は同じ文字列列を`object`と表現するためで、値、行順、予測、scientific contractの
差ではない。version 3もtruth attachmentとmetrics計算より前である。

version 4では読み込み直後に、`id/well_id=object`、
`row_idx/suffix_offset=int64`、`raw_gr_observed=bool`、その他の数値列を
`float64`へ固定してからnormalized schema SHAを検証する。科学設定、予測bytes、
logical SHA、audit、fold、gateは変更しない。

- version 4新規PF well-runs: `0`
- 再利用PF well-runs / seed-well / particle starts:
  `1,546 / 197,888 / 98,944,000`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- canonical train SHA:
  `56b7df38c942228b60291b6fcd5ab974a0b10c0aec22049686a22379f3bd608d`
- push準備時config SHA:
  `0f53cd30223f2338a34fa2d17e68eab5270aeb792ec6787b6cf2f65b76c52e14`
- dedicated tests: `12 passed`
- dedicated + notebook tests: `16 passed`
- Jupytext / Ruff / strict exp validation: PASS
- private / CPU / GPUなし / internet off / `run_on_push`
- dataset source 1 / kernel source 4
- version 4 start: `2026-07-26 12:20:09 UTC`
- initial status: `KernelWorkerStatus.RUNNING`

## 2026-07-26 version 4 COMPLETE

version 4は`270.98839378356934 sec`でlate readoutまで完了した。新規PFは0で、
version 1の凍結済み予測をraw / decompressed / logical / normalized schema SHA
すべて一致させて再利用した。

### Primary

- control x1.0 RMSE: `10.914522073423171`
- candidate x1.3 RMSE: `11.174615008412255`
- gain x1.0 - x1.3: `-0.2600929349890837 ft`
- improved/nonworse folds: `1/5`
- raw GR observed gain: `-0.17963169827254966 ft`
- raw GR missing gain: `-0.431145 ft`
- high missing-fraction gain: `-0.693932 ft`
- suffix 1000+ gain: `-0.286098 ft`
- hidden-like spatial / typewell-purged gain:
  `+0.068326 / +0.013773 ft`
- by-well delta p95: `+4.8264674626804425 ft`
- worst-well regression: `+37.33385058701939 ft`
- improved wells: `286/773`
- worst well: `60e37807`、missing fraction `0.025936`

technical gateと4 parity checkはすべてPASSした。truth/error/fold/hidden-likeの
pre-freeze read countは0、finite coverage 1.0、773 wells、3,783,989 rows、
common seed、scale ratio、post clip 0、実行量をすべて確認した。

scientific gateはFAIL。decisionは
`scale5_rejects_global_gs_x1p3_close_without_rescue`。同じOOFでのtemperature /
multiplier / clip / seed / particle / adaptive gate / HMM / Beam / ML / blend
救済は行わず、inferenceとsubmissionも実行しない。

### 最終SHA

- scientific contract:
  `41c1f95ca8bd7d20eef00f244ced7d4dbc4b3571cc9fd4189c08d6831ef15b57`
- prediction logical:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
- prediction raw gzip:
  `b3699432a691229da5a6562ce74e0b84f1bee3021bd80d650526906f5aa390f8`
- prediction decompressed:
  `00fe1b90fce84bd601b4b91442d9fc698200aafadd48658f7d8c26ec1fbe0d00`
- normalized schema:
  `2372de95ab154ddd2b1bfe545923233bc41f96104e66c32ab93e6f0fa99bee88`
- input manifest:
  `6672b3e9839dbf13670a327bfd45ee2f947c9839a36de42b2bc4534b0428600d`
- artifact manifest:
  `131a65c36acafc8d3cac9bdc18b2b5e296ff9aceb93cbf2702b1a79e675b58f3`
- metrics:
  `9b69317a8979cb29e899df336f218d2008504844a27af53a3de0b4ff34e3b83d`

## 2026-07-26 設計確定

### 根拠

exp400の同一x1.3 PF runでは、算術`pf_mean` RMSE
`12.221810980460939`に対し`scale_5`は`11.174614846889103`だった。
しかし保存exp072にはx1.0 scale 5列がないため、x1.3の倍率効果と
seed集約効果を分離できない。そこでx1.0/x1.3を同一seedで再実行する。

### 固定比較

```text
control   = likpf_scale_5_x1p0
candidate = likpf_scale_5_x1p3
gain      = RMSE(control) - RMSE(candidate)
```

- `scale_5`をprimaryへ事前固定
- common per-well SHA256 seed、seed indices 0--127
- 500 particles / 128 seeds / exp072 PF dynamics
- base `gs`はzero-fill population stdを`[10,60]` clip
- x1.3はbase clip後1回だけ乗算、post clipなし
- `pf_mean`はx1.0 exp072 / x1.3 exp400 parity専用
- scale 3/8/12、HMM、Beam、ML、blendなし

### 実行量

- 2 variants
- 1,546 PF well-runs
- 197,888 seed-well trajectories
- 98,944,000 particle starts
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- HMM / Beam: 0 / 0
- CPU runtime見積り: `20,992.600 sec`（約5.83時間）

### Promotion

- pooled gain `>=0.05 ft`
- 4/5 folds以上改善
- raw GR observed gain `>=0.05 ft`
- raw GR missing、高missing wells、1000+、hidden-like 2面は非悪化
- by-well delta p95 `<=0`
- worst-well regression `<=+0.25 ft`

FAIL時はtemperature、multiplier、clip、seed、particle、well gate、
blendを同じOOFで救済しない。PASSでもinference/submissionへ自動移行しない。

## 2026-07-26 implementation-only

### 実装

- exp400 compact self-contained trainの11章構成を参照し、同じ役割順で
  exp404 train候補をJupytext percent形式へ実装した。
- 親exp400は2,016行、exp404は約2,000行。薄いentrypointではなく、
  path/SHA、科学契約、input preflight、truth-free入力、Numba PF、
  paired generation/freeze、late truth、metrics/gate、生成物保存を展開した。
- 各wellで`stable_seed("likpf","train",well_id)`を1回決め、
  x1.0とx1.3の両方へ同じseed baseと0--127 indexを渡す。
- outputは`likpf_scale_5_x1p0 / x1p3`とtechnical parity用
  `likpf_mean_x1p0 / x1p3`だけ。scale 3/8/12は実装に含めない。
- exp072 mean、exp400 x1.3 mean / scale5のRMSE parityはtechnical gate専用。
- 両variant prediction / audit / logical content SHAをfreeze後だけsuffix TVT、
  exp226 fold、exp115 hidden-like roleをjoinする。
- inference候補はraw-test predictionとsubmissionを作らないfail-closed実装。

### 実装時の実行量確認

- active variants: 2
- LightGBM config / fold / booster: `0 / 0 / 0`
- PF well-runs: 1,546
- seed-well trajectories: 197,888
- particle starts: 98,944,000
- parent/control再学習: なし。ただし科学controlのx1.0 PF再生成は
  凍結済み設計の必須処置。implementation-only時点ではKaggle実行未承認
- HMM / Beam / model / inference / submission: 0

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp404_scale5_sigma_gr_likelihood_pf_ablation/*compact_selfcontained*.py \
  tests/test_exp404_scale5_sigma_gr_likelihood_pf_ablation.py
.venv/bin/ruff check \
  experiments/exp404_scale5_sigma_gr_likelihood_pf_ablation/*compact_selfcontained*.py \
  tests/test_exp404_scale5_sigma_gr_likelihood_pf_ablation.py --select F821
.venv/bin/pytest -q tests/test_exp404_scale5_sigma_gr_likelihood_pf_ablation.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp404_scale5_sigma_gr_likelihood_pf_ablation/exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp404_scale5_sigma_gr_likelihood_pf_ablation/exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py
make validate-exp EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation
```

- py_compile: PASS
- Ruff F821: PASS
- dedicated pytest: `11 passed`
- Jupytext train / inference変換とround-trip: PASS
- strict `make validate-exp`: PASS
- `task` runnerは環境に存在しなかったためMakefileの同等commandを使用
- `__file__`はcompact train / inference sourceに0件
- implementation-only時点では正規Notebookを上書きしていない

## Design-onlyコマンドログ

実行したのはscaffold作成とdesign文書編集だけ。

```bash
make new-steering EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation
make new-exp EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation \
  SOURCE=templates/experiment
make update-summary
make validate-exp EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation
.venv/bin/python \
  .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py \
  exp404_scale5_sigma_gr_likelihood_pf_ablation --root .
```

- `make validate-exp`: strict PASS
- design assertion: route、status、2 variants、scale 5、実行量、
  run disabled、metrics JSON parseを確認してPASS
- document review: core evidence categoriesは実験文書全体で充足

このdesign-only段階では学習、推論、Kaggleコマンドを実行していなかった。
当初予定した長いkernel IDは後のpushでKaggleの50文字上限を超えたため、
実行時は上記43文字の`likpf` IDへ短縮した。

```bash
make prepare-kaggle-notebooks EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train --title 'exp404 scale5 sigma gr likpf ablation train' --run-on-push --strict"
```

## 変更点

- exp400のprimary `pf_mean_x1p3`ではなく、scale 5を固定したx1.0/x1.3の
  paired A/Bを新しい科学問いにした。
- exp072 scale 5 controlが未保存のため、CPU control再実行を設計上必須にした。
- variant名をseed keyに含めず、common random-number labelを固定した。

## 再現性メモ

- seed policy:
  `SHA256("likpf::train::<well_id>")`由来base + seed index、variant間共通
- stochastic components: particle初期化、process noise、systematic resampling、
  roughening
- parallel: 8 well threads、Numba kernelへ明示seed、stable well/variant order
- CPU/GPU runtime: Kaggle CPU / GPUなし / internet off
- input / prediction SHA: 実行時にraw/decompressed、schema、logical content、
  manifestを記録
- model / submission SHA: 生成しないため対象外
- Kaggle kernel id / version:
  `kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train` / 1
- deterministic anchor: false。train-side candidate auditとしてのみ設計

## 禁止事項

- scale 3/8/12、temperature grid、meanへのprimary差替え
- multiplier / clip / particles / seeds / resamplingの探索
- HMM / Beam / ML / selector / blend / public full pipeline replay
- same-OOF rescue
- inference、submission

## 次のアクション

version 1完了後、必要な実ファイルだけを取得してinput / prediction logical
content / decompressed output / artifact manifest SHAを照合し、technical /
scientific gateを判定する。scientific gateをPASSしてもraw-test regeneration /
inference / submissionは別設計・別承認とする。
