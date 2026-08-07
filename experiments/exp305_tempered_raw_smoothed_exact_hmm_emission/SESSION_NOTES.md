# exp305_tempered_raw_smoothed_exact_hmm_emission セッションノート

## 目的

exp304選択SWTをraw emissionへbeta 0.15で弱く混ぜ、exp209互換exact-HMM posterior meanと保存済みlikPF 50/50 blendのdecoder価値を、固定controlと固定gateで判定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v3完了、全promotion gate FAILで閉鎖
- active scientific variant: 1
- HMM well-runs: 773
- model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0`
- control再実行: 0
- CV / LB / submission: train-side paired RMSE確定 / なし / なし

### 2026-07-21 実装

ユーザーの追加依頼「exp305を実装してください」を実装承認として扱い、凍結済み設計から自由度を増やさず次を実装した。

- compact self-contained train Jupytext source / Notebookと正規train Notebook。
- exp304 series/manifest/summary/scientific contract/input manifest、exp209 saved HMM/exp072 v5 cache、exp226 OOF、exp115 hidden-like assignmentのresolverとSHA hard preflight。
- exp304 6,659,300-row selected seriesのwell単位streaming、raw well-file identity再計算、raw series parity。
- `ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`、known-prefix raw sigma `[10, 60]`共有、clip 600、exp209 exact forward-backward/posterior mean。
- deterministic gzip prediction保存とraw/decompressed/content SHA freeze。その後だけunknown-suffix truthとsaved row controlを読むlate join。
- direct / fixed likPF 50/50のoverall、5 folds、1000+、hidden-like 2面、by-well p95/worst gate。
- fail-closed inference Notebook。raw-test prediction、submission、automatic promotionは生成しない。
- synthetic/contract tests 7件。

実装後の実行量は、active scientific variant 1、773 HMM well-runs、model / LightGBM config / trained fold / PF / Beam / booster `0 / 0 / 0 / 0 / 0 / 0`、control再実行0。Kaggle GPU control再学習はなく、Kaggle CPU package/push/runも行っていない。

### 2026-07-21 Kaggle CPU train実行承認

ユーザーの追加依頼「実行してください」を、凍結済みexp305 trainのKaggle CPU package / push / run承認として記録した。承認対象はactive scientific variant 1、773 exact-HMM well-runs、model / LightGBM config / trained fold / PF / Beam / booster `0 / 0 / 0 / 0 / 0 / 0`、親/control再実行0。GPU、internet、inference、submissionは無効のままとする。

- canonical kernel id: `kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train`
- canonical title: `exp305 tempered raw swt exact hmm emission train`
- kernel sources: exp304 train v1、exp209 train、exp226 train、exp115 trainの4件
- fixed emission: `ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`
- runtime: Kaggle CPU、outer workers 2、Numba threads 2、上限8.5時間

初回は実験ディレクトリ名をそのままslug化した53文字の`kentookumura/exp305-tempered-raw-smoothed-exact-hmm-emission-train`をpushしたが、Kaggle `SaveKernel` がHTTP 400を返した。id/titleのslugは完全一致していた。直後に同slugを`kaggle kernels pull -m`してHTTP 403となり、kernelが作成されていないことを確認した。Kaggle側の長さ制約に合わせ、科学的意味を残す`swt`略記で48文字のcanonical id/titleへそろえて再packageする。実験番号、Notebook、config、beta/HMM contract、実行量は変更しない。

48文字canonical packageでもbootstrap 17ファイルのSHA不一致0、埋め込みconfig一致、4 kernel sources、CPU/internet-off、approval flag、1 variant、773 HMM well-runs、beta 0.15、0 booster、inference/submission-offを再確認した。再pushは成功し、Kaggle CPU train v1を開始した。

- URL: https://www.kaggle.com/code/kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train
- Version: 1

Kaggle側metadata pullは成功し、`id_no=128079137`、CPU、GPU/internet-off、4 kernel sourcesを確認した。v1は約18秒で`KernelWorkerStatus.ERROR`となり、logsから計算開始前の`ValueError: exp304 silent fallback count mismatch`を確認した。exp304 v1の小型summary/scientific contract/manifestだけを選択取得して調べると、実値はdenoised-series manifestの`silent_fallback_count=0`であり、exp305が誤ってsummaryの`technical_gate`直下を読んでいたことが原因だった。

preflightをmanifest参照へ修正し、summary側に同fieldがなくmanifest側に0があるfixtureで回帰testを追加した。固定beta、HMM、入力SHA、gate、実行量は変更していない。exp305 testsは8 PASS、py_compile、ruff F821、Jupytext round-trip、strict experiment validationはPASS。正規train Notebookとcompact self-contained NotebookをJupytext sourceから再生成した。

修正版packageで、Notebook本体とbootstrap sourceの両方にmanifest参照修正があること、bootstrap 17ファイルのSHA不一致0、固定contract/実行量不変を確認し、同じkernelへv2をpushしてKaggle CPU実行を開始した。

v2は少なくとも2時間25分にわたり`KernelWorkerStatus.RUNNING`を維持した。ユーザーの依頼「監視は止めていいです。完了したら連絡します。」に従い、Kaggle実行は止めず、Codex側の45秒status監視プロセスだけを停止した。完了連絡後にlogs/cell outputと必要なmetrics/manifest/SHAを確認し、実験記録を確定する。

ユーザーの完了連絡「失敗しました」を受けてv2 logsを取得した。v2は773/773 HMM well-runsを完了し、9,455.6秒時点のlate readoutで`ValueError: Usecols do not match columns ... ['likpf_mean']`によりERRORとなった。exp209 v5 small outputの実schemaと`direct_hmm_comparison.py`を照合すると、保存cacheは絶対TVTの`likpf_mean`ではなく差分featureの`likpf_mean_d`を持ち、正しいcontrolは`last_known_tvt + likpf_mean_d`だった。v2のKaggle output filesは空で、凍結済みpredictionの救出はできなかった。

v3修正ではconfigを`likpf_mean_d` / `delta_from_last_known_tvt`へ合わせ、late readoutで絶対TVTへ復元する。さらに全大容量gzipのheader列をSHA preflightと同時に記録し、exp304 series、exp209 HMM、exp209 exp072 cache、exp226 truth/safe列の必須schemaをHMM開始前にfail-fast検証する。これは保存済みlikPF trajectoryの正しいmaterializationであり、beta、SWT、sigma、HMM、blend、gate、実行量、control自体は変更しない。

v3修正後は9 tests、py_compile、ruff F821、Jupytext round-trip、strict experiment validationがPASS。compact self-contained train Notebookと正規train NotebookをJupytext sourceから再生成した。同一kernelのmetadata pullも成功し、`id_no=128079137`の既存v2を確認してからv3 packageへ進む。

v3 packageでもbootstrap 17ファイルのSHA不一致0、`likpf_mean_d` delta復元、HMM前schema guard、CPU/internet-off、4 kernel sources、1 variant、773 HMM well-runs、0 booster、inference/submission-offを再照合した。同一kernelへv3をpushし、metadata pullで`id_no=128079137`、statusで`KernelWorkerStatus.RUNNING`を確認した。ユーザーの先の意向に従い、定期監視は行わず完了連絡を待つ。

### 2026-07-21 Kaggle CPU train v3完了

ユーザーの完了連絡後に一度だけstatusとlogsを取得し、`KernelWorkerStatus.COMPLETE`を確認した。ログにfold別実数が不足していたため、Kaggle outputからpredictionを除外し、summary、promotion gate、overall/fold/scope metrics、by-well metrics、input/control manifest、scientific contractの小型生成物だけを取得した。取得ファイルのraw SHAはNotebook summary記録値と一致した。

- Runtime: `15,983.839958 sec`（約4時間26分24秒）、上限30,600秒以内。
- Rows / wells / HMM runs: `3,783,989 / 773 / 773`。
- Finite coverage / ID mismatch / silent fallback: `1.0 / 0 / 0`。
- direct RMSE: candidate `13.218199372`、saved raw HMM control `11.938287235`、改善量`-1.279912137 ft`、改善1/5 folds。
- fixed 50/50 blend RMSE: candidate `10.767674449`、saved control `10.269692505`、改善量`-0.497981944 ft`、改善1/5 folds。
- direct fold candidate-control差: `+2.973011 / -0.032811 / +2.677078 / +0.381420 / +0.457403 ft`。
- blend fold candidate-control差: `+1.443477 / -0.150180 / +1.013859 / +0.096239 / +0.180397 ft`。
- 1000+ / hidden-like spatial / hidden-like typewell-purged / by-well p95 / worst-wellはdirect、blendとも全FAIL。direct差は`+1.395238 / +1.052477 / +0.993719 / +1.906563 / +56.989605 ft`、blend差は`+0.542714 / +0.442637 / +0.429642 / +1.868961 / +29.298736 ft`。

strict technical gateはraw HMM baseline parityをPASSしたが、正しい`last_known_tvt + likpf_mean_d`復元後のsaved likPFとraw-HMM/likPF 50/50が事前記録値からそれぞれ`3.2766e-6 / 3.6416e-6 ft`ずれ、固定許容値`1e-6 ft`を超えたためFAILした。この差はcandidateの科学的悪化`0.498--1.280 ft`より十分小さく、全fold/scope FAILも変わらない。positive resultとしてのstrict parityは主張しないが、候補棄却のnegative decisionは信頼できる。

主要SHA:

- scientific contract: `343084494621e1a3bb15899f6c3c441507f4dfd3af671e8de7c08f3f8867bd1b`
- input/control manifest: `48f3c68cc1d02f60871a111f0cb473cc9a67d29963cc5d4c79143c69c00d15f7`
- prediction raw gzip: `6419657d633564325ced8cabfde22532737cdabd1c7844686317d8dd7efe2552`
- prediction decompressed/content: `86b1768f18d31ba296774054c14b24e2e4650ddc74d9858d9baea8b534027302`
- promotion gate: `9edc8fa51f070d38420bc6b84ec5e329a1b5c7425a46234f4e85cf5741ee1c71`
- overall/fold/scope metrics: `efe59c08753072d658eac1138cb1702ac4579939734de68e1aec89a615f4532e`
- by-well metrics: `4b810d2620172427e6da2373a0a23813d127325f9f3beb9db4d85943ec1070b2`

事前登録の`close_without_rescue_and_keep_reserved_pf_transfer_closed`を適用する。beta、sigma、clip、HMM、blendの救済、案3/案4、inference、submissionは行わない。exp305由来の新しい救済backlogも追加せず、独立したexp307 finite-only robust sigmaと、exp305完了待ちだったexp321 Stage A/Bを既存条件のまま優先する。

```bash
.venv/bin/pytest -q tests/test_exp305_tempered_raw_smoothed_exact_hmm_emission.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_train.py experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_train.py experiments/exp305_tempered_raw_smoothed_exact_hmm_emission/exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_inference.py tests/test_exp305_tempered_raw_smoothed_exact_hmm_emission.py --select F821
make validate-exp EXP=exp305_tempered_raw_smoothed_exact_hmm_emission
make validate-template
make test
```

- 最終的にexp305の9 tests PASS。
- Jupytext round-trip、`py_compile`、ruff F821、strict experiment validation、template validationはPASS。
- 親exp304 compact trainは10章 / 2,258行。exp305 compact trainは同じ役割を保った10章 / 1,776行で、入力preflight、selected-series streaming、HMM、freeze、late readout、gate、生成物保存をすべてNotebook上で追える。
- `__file__`依存はtrain/inference Jupytext sourceに残していない。
- 全体test suiteはexp305の7 testsを含む404 PASS / 1 SKIP。既存`exp296`のstatus/run approval期待と現configがずれている2 testsだけFAILした。exp305関連test、Notebook共通test、scaffold testはすべてPASSしており、exp296ファイルは本実装で変更していない。

## コマンドログ

### 2026-07-21 設計

```bash
make new-steering EXP=exp305_tempered_raw_smoothed_exact_hmm_emission
make new-exp EXP=exp305_tempered_raw_smoothed_exact_hmm_emission
```

- `kaggle-review-exp`に従いsteeringを先に作成した。
- exp304 reserved follow-up案2、exp304 actual SHA/result、exp209 v5 control/runtime、`docs/06_reproducibility.md`を確認した。
- requirements/design/tasklist、config、README、result、metricsをdesign-onlyとして確定した。
- scaffold Notebookは誤実行でmetrics/submissionを作らないdesign-only guardとする。
- 実験ロジック、Jupytext source、Kaggle package、push、runは作成・実行していない。

## 固定した設計

- `ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`の1 variantのみ。
- selected denoiserは`swt_db4_l3`。SWT、sigma、clip、HMM、blend weightを変更しない。
- saved raw HMM `11.9382872349`、saved likPF `11.5948976722`、saved 50/50 `10.2696961466`をcontrolとする。
- direct/blend各`>=0.05 ft`、4/5 folds、1000+/hidden-like 2面/p95非悪化、worst `<=+0.25 ft`、全SHA/coverageを必須とする。
- FAIL後のbeta/sigma/filter/HMM/blend救済は禁止する。

## 再現性メモ

- seed policy: 新規RNGなし、well文字列昇順。
- stochastic components: 保存済みlikPFは上流stable SHA256 per-well seed生成物だが、本実験では再生成しない。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off、上限8.5時間。v3は15,983.840秒で完了。v2は9,455.6秒後にlate readout ERROR。
- Kaggle kernel id / version: `kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train` / v3 COMPLETE、`id_no=128079137`。v2はlate-readout列契約ミスでERROR、v1はpreflight field参照ミスでERROR。53文字の元slugはSaveKernel 400かつpull 403で未作成。
- input SHA: exp304 series content `a4acb72d...a0988`、scientific contract `8822df96...064`、raw identity `bbb687a1...b32`、exp209 HMM `8e2f4236...7ae5`、exp209 v5 exp072 cache `0503de05...536`を固定。
- prediction SHA: raw gzip `6419657d...2552`、decompressed/content `86b1768f...7302`。prediction本体は取得せずNotebook出力SHAを記録した。
- model/submission SHA: 非該当。

## 次のアクション

1. exp305をnegative resultとして閉じ、同一OOFでbeta/sigma/HMM/blend救済を行わない。
2. exp304 reserved案3/案4を閉じたまま維持する。
3. inference/submissionは行わない。
