# exp304_gr_denoiser_emission_separability_readout セッションノート

## 目的

GR平滑化をHMM/PFの尤度へ入れる前に、固定shift bank上のemission separabilityだけを監査する。
案1の設計とcompact self-contained実装を確定し、案2〜4が別セッションで変形されないよう予約契約を残す。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU v1完了、technical/quality gate PASS、`swt_db4_l3`選択
- CV / LB: 非該当（train-side separability diagnostic） / なし
- 実行量: 4 readout variants、13 shifts、7,787 blocksを完了
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam well-run: `0 / 0 / 0`
- GPU / inference / submission: なし / なし / なし

### 2026-07-20 実行承認

- ユーザーの「実行してください」を、別名compact self-contained train Notebookの正規train採用と、
  exp304 private Kaggle CPU v1を1回push/runする明示承認として記録した。
- 実行対象は`raw / robust_rts / swt_db4_l3 / l1_trend`の4 variants、固定13 shifts、保存済み5 fold strata。
- LightGBM/model config、trained fold、booster、HMM、PF、Beamはいずれも0。親/control再学習、GPU、
  inference、submissionは行わない。
- 正規採用後にpackage metadata、bootstrap/source/config SHA、CPU・internet-offを検証してからpushする。

### Kaggle CPU v1 push前preflight

- kernel id: `kentookumura/exp304-gr-denoiser-separability-train`
- title: `exp304 gr denoiser separability train`
- metadata: private、CPU、GPU/TPU/internet off、`run_on_push=true`
- kernel sources: `kentookumura/exp226-k16-kappa-repro-train`、
  `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- package config SHA256: `1b66ebb89675fd8f5129ee9ca7c0e520d2d7a55bf938f90481198751f4457e98`
- package self-contained source SHA256: `b3a038bbb769f74314570104afc8a6b34758cbdbb8b96148bd76b16e8f8f3035`
- package bootstrap Notebook SHA256: `10ea3f9e0c2c57c8bc26b2cd6321248826e08b30357970e9207560ae9ff840ea`
- kernel metadata SHA256: `6bfd4ce4f980db92822386cc0bc8f14a4753367556db660481f5bf694464f737`
- packageは`--no-src --strict`で生成し、正規train Notebook、config、必要なself-contained sourceだけを含めた。
- exp226 OOFとexp115 fold assignmentが各kernel outputに存在することを`kaggle kernels files`で確認した。
- package sourceの構文、metadata JSON、strict experiment validationはPASSした。
- 最初に予定した54文字slug/titleはKaggle `SaveKernel`がHTTP 400で拒否し、runは作成されなかった。
  科学contractを変えず、slug/titleだけを上記の短縮名へ変更してpackageを再生成した。

### Kaggle CPU v1実行

- `Kernel version 1 successfully pushed`を確認した。
- URL: `https://www.kaggle.com/code/kentookumura/exp304-gr-denoiser-separability-train`
- push後、誤再push防止のためローカル`execution.kaggle_push_approved`を`false`へ戻した。
- 実行中はstatus/logsを監視し、完了後だけactual outputを取得してSHAとfixed gateを検証する。

### 2026-07-21 Kaggle CPU v1完了・結果確定

- Kaggle status `COMPLETE`、kernel version 1、id_no `128011752`を確認した。
- runtime `4,740.758 sec`、3,783,989 rows、773 wells、7,787 blocks、13 shifts。model、LightGBM、
  trained fold、booster、HMM、PF、Beam、inference、submissionは全て0。
- raw overallはMRR `0.389625985`、top1 `0.189546680`、top3 `0.452420701`、mean rank `4.653139848`。
- `swt_db4_l3` overallはMRR `0.424724294`、top1 `0.220624117`、top3 `0.504687299`、
  mean rank `4.195582381`。raw比gainはMRR `+0.035098309`、top1 `+0.031077437`、
  top3 `+0.052266598`、truth-minus-best-decoy gap `+0.035917064`。
- SWTはMRR/top3を5/5 foldsで改善し、real-vs-shuffledも5/5 folds、必須4 scope、top1非悪化、
  decoy-gapを含む全事前登録quality checkをPASSした。唯一のpassing/selected denoiserである。
- technical gateはrawとSWTが1,546/1,546 series PASS。robust RTSは15/1,546だけが最大8反復内に収束し
  1,531 failures、L1 trendは572/1,546だけが最大500 ADMM反復内に収束し974 failuresだった。
  RTS/L1はtechnical FAILとしてquality非評価とし、同一OOFで反復・閾値を救済しない。
- PyWavelets `1.8.0`、silent fallback `0`、common freeze/row-block-fold identity/finite coverageはPASSした。

### actual output / SHA検証

- 一括`kaggle kernels output`はdenoised series download中にpartialとなり0-byte fileを残したため、
  `/tmp/kaggle-output/exp304_gr_denoiser_emission_separability_readout/train_v1_retry`へfile patternで
  summary、metrics、manifest、scientific contract、solver/fold/scope/distortionを個別取得した。
- 取得した小型生成物のSHAはsummary記録値と全件一致した。block readout本体もraw SHA
  `c312305820c2c8ebf7f5574090b2fb93dc3e8cc769f6c2c60033cae28cf3e9b8`、decompressed SHA
  `239c990260032d896894d0903ed99e7cbb39796e1e6357218acfe4869a9f6623`、15,574 data rowsで一致した。
- exp226 OOF decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp115 hidden-like SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- raw well-file identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- scientific contract content SHA: `8822df968200b74ea9969b0bc023ec127debbff01933bdc89ff3db9844d55064`
- denoised GR content SHA: `a4acb72d60b833b12b2560db1e5dc3a113ae6ecf4137efbccf78c278582a0988`
- target-free score content SHA: `6c71bdae030fee04e40988a8abdef4e26b61af733c463b6b0625e2e0cc99fa69`
- 大容量denoised series本体はリポジトリへ保存せず、Kaggle manifest/logsのraw/decompressed/content SHAを
  正とする。診断後の後続設計にはKaggle kernel outputをsourceとして参照する。
- 大容量seriesだけの単独file-pattern downloadも0 byteだった。案2の実行前にexp304 kernel source上で
  nonzero size、6,659,300 rows、manifestのraw/decompressed/content SHA一致をhard preflightし、
  不一致ならHMMを開始しない。

## コマンドログ

### 2026-07-20 設計

```bash
make new-steering EXP=exp304_gr_denoiser_emission_separability_readout
make new-exp EXP=exp304_gr_denoiser_emission_separability_readout
```

- `kaggle-strategy`で既存backlog、exp189/209/280、CV/LB anchorを確認した。
- `kaggle-review-exp`に従いsteeringを先に作成し、その後experiment scaffoldを作成した。
- `.git`はこの実行環境でrepositoryとして認識されず`git status`を取得できなかったため、既存ファイルは
  上書きせず必要箇所だけを編集した。
- `make new-exp`のtrain/inference templateは、誤実行でmetricsやsample submissionを作らないよう
  outputなしのdesign-only `RuntimeError` guardへ変更した。実験ロジックは含まない。

### 2026-07-20 実装

- ユーザーの「exp304を実装してください」をtrain-side readout実装承認として記録した。
- 既存正規Notebookは上書きせず、別名compact self-contained Jupytext train/inference sourceを作成した。
- train sourceは10章構成で、入力/SHA preflight、raw/robust RTS/SWT/L1 trend、solver status、
  deterministic gzip streaming、target-free score freeze、late truth join、scope/fold metrics、
  distortion、technical/quality gate、expected生成物保存をNotebookセル上へ展開した。
- robust RTSはStudent-t df=4、最大8 IRLS、RTS mean/posterior varianceを保存する。
- SWTはright reflection padding、db4 level 3、detail MAD universal soft threshold。PyWavelets不可時や
  short/invalid seriesではlevel低下・別filterへfallbackせず、その方式だけtechnical FAILとする。
- L1 trendはsecond-order ADMM、rho=1、最大500反復、abs/rel tolerance 1e-4を固定し、
  pentadiagonal Choleskyをseriesごとに1回だけfactorizeする。
- sharp-edge scopeはraw Type Well absolute gradientを`tvt_geop`へ補間したblock meanのpooled p90以上に固定した。
- synthetic test 7件でcontract、RTS決定性/収束、L1収束、raw +10 ft shift rank、truth freeze、
  promotion selection、inference fail-closeを確認した。
- full local notebook、Kaggle package/push/run、HMM/PF/Beam、inference/submissionは実行していない。

### 実装検証コマンド

```bash
.venv/bin/python -m py_compile <exp304 compact train.py> <exp304 compact inference.py>
.venv/bin/ruff check <exp304 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q experiments/exp304_gr_denoiser_emission_separability_readout/tests/test_exp304_gr_denoiser_emission_separability_readout.py
# 7 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 <source.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <source.py>
make validate-exp EXP=exp304_gr_denoiser_emission_separability_readout
# strict validation passed
make validate-template
# PASS
.venv/bin/pytest -q
# 383 passed, 1 skipped, 2 unrelated existing exp296 failures
```

- `task validate-exp`はこの環境に`task` commandがないため実行できず、skill指定のfallbackで
  `make validate-exp`を実行してPASSした。
- full suiteの2 FAILは既存exp296 configが完了後status/run flagへ更新されている一方、testが
  `kaggle_cpu_*` statusと旧approval分岐を期待する不整合で、exp304変更とは無関係なので修正しない。
- exp280 self-contained trainは9章 / 1,165行、exp304 compact trainは10章 / 約2,200行。
  exp304は3 solver、streaming freeze、technical/quality gateを追加しており、親相当の入力・score・
  truth join・metrics・生成物orchestration章を欠いていない。
- ローカルvenv/system PythonにはPyWaveletsがない。contractどおりSWTだけをtechnical FAILとして記録し、
  fallbackしない経路をsynthetic score testで確認した。Kaggle push時はpackage preflightで利用可否を再確認する。

## 設計判断

- exp189は平滑化後GRからsigmaも変わっており、PFではwrong-modeへの安定化が起きた可能性がある。
  exp304はsigmaをraw基準で共有し、denoiserだけを変える。
- Kalman系は通常のcausal filterではなく、offline入力を使える条件に合わせたStudent-t robust RTS smootherを採用する。
- waveletはexp218のdecimated DWT featureではなくstationary `db4` level 3、もう1本はsecond-order L1 trendに固定する。
- filter数値設定、候補bank、scope、promotion閾値をOOF結果後に変更しない。
- raw likelihoodを捨てる案は採用せず、exp304 PASS後の最初のdecodeはbeta 0.15のtempered mixtureだけにする。

## 再現性メモ

- seed policy: real denoising/scoreはRNGなし、shuffled controlだけstable SHA256 local RNG。
- stochastic components: shuffled candidate-score permutationのみ。
- CPU/GPU runtime: Kaggle CPU version 1、4,740.758秒。GPU/TPU/internet off。
- input/feature SHA: exp226、exp115、raw well-file identity、scientific contract、denoised GR、
  target-free scoreのactual SHAを上記に記録した。
- model/prediction/submission SHA: 生成しないため非該当。
- Kaggle kernel id/version/rerun: `kentookumura/exp304-gr-denoiser-separability-train` / version 1 / rerunなし。

## 次のアクション

1. exp304は完了とし、同じ実験でfilter設定変更、HMM/PF、inference、submissionへ進めない。
2. `reserved_followup_contract.md`案2は2026-07-21に`exp305_tempered_raw_smoothed_exact_hmm_emission`として
   steering/実験ディレクトリの設計を確定した。selected SWTと`ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`を
   固定し、実装・Kaggle実行はまだ行わない。
3. SWT選択のため案3は閉じる。案4は案2が全gate PASSするまで開始しない。
4. robust RTS / L1 trendの収束調整は2026-07-21に
   `exp306_robust_rts_l1_convergence_calibration_audit`としてsteering/実験ディレクトリの設計を確定した。
   低-中優先の独立候補とし、exp304のposthoc rescueには使わない。target-free technical auditでbranch別に
   全1,546 seriesが収束した場合だけ、単一設定を固定した将来の別expで科学評価する。実装・実行はまだ行わない。
