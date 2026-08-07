# exp443_mean_preserving_trapezoidal_lattice_hmm セッションノート

## 目的

exp439で不可能だったexact mean+parent variance契約を繰り返さず、
trapezoidal meanを厳密保存し格子が強制する最小分散を明示的に受け入れる。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 Kaggle kernel version 1完了、terminal fail-close
- 親: exp209
- 優先度: P3、exp441/442より後
- 実装: 2026-07-29のユーザー依頼で承認・完了
- 正規train Notebook採用 / package / Stage 0: 2026-07-30承認済み
- Stage 1 / inference / submission: 未承認
- CV / LB: なし

## コマンドログ

### 2026-07-29 design-only作成

```bash
make new-steering EXP=exp443_mean_preserving_trapezoidal_lattice_hmm
make new-exp EXP=exp443_mean_preserving_trapezoidal_lattice_hmm
```

### 2026-07-29 実装

- exp439 compact self-contained trainを親構成にし、exp209 HMM input、
  adjacent rate marginal、forward/backward、truth-late fixed32 readoutを維持した。
- 科学差分は固定5-cell、trapezoidal mean、
  `v_eff=max(v_parent,v_lattice_min)`だけに限定した。
- joint-edge table SHAへminimum/effective varianceとvariance inflationを含めた。
- variance inflation min/mean/max、floor-active edges、negative weights、
  fixed-support violationを監査表とgate diagnosticsへ追加した。
- exp439 failure edge `v_min=0.0264 > v_parent=0.01500625 ft²`を
  target-free positive contractとして追加した。
- compact train/inference `.py/.ipynb`を別名で作成し、正規Notebookは保持した。
- 専用test: `12 passed`。
- exp439/443関連test: `24 passed`。
- `py_compile`、Ruff F821、Jupytext変換/test: PASS。
- `task validate-exp`は環境に`task`がなく実行不能だったため、同等の
  `make validate-exp EXP=exp443_mean_preserving_trapezoidal_lattice_hmm`を実行し、
  strict validation PASS。
- 親compactとの構成比較: exp439 train `3,125`行、exp443 train `3,430`行。
  両方とも同じ9章の役割スロットを持ち、exp443はvariance-floor contract /
  audit / gateの追加分だけ長い。

### 2026-07-30 Stage 0実行承認・push前契約

- ユーザーの「実行してください」により、正規train Notebook採用、
  Kaggle package作成、Stage 0 fixed32実行を承認済みとした。
- 実行量:
  - active scientific variant: `1`
  - candidate HMM well-runs: `32`
  - reporting folds: `5`
  - 保存exp209 parent/control HMM rerun: `0`
  - LightGBM config / trained fold / booster / fitted model: `0 / 0 / 0 / 0`
  - PF / Beam / GPU run: `0 / 0 / 0`
- 実行環境: Kaggle private CPU、internet disabled、Numba threads `1`。
- canonical kernel（Kaggle長さ制約対応後）:
  `kentookumura/exp443-mean-pres-trapezoid-lattice-hmm-train`
- `kaggle-platform` credential check:
  OAuth credentialとlegacy CLI credentialを確認。API tokenは未設定だが、
  Kaggle CLI 2.2.3のOAuth実行経路を使用する。
- canonical train Notebook SHA:
  `d68b9bb0cfa1fcc804ad2a4072cc4382523f5a265edc88eaa458a12db45d56cf`
  （compact trainと一致）。
- Kaggle packageをcanonical id/title、private CPU、internet disabled、
  `run_on_push=true`で作成した。
- bootstrap内のconfig SHAはloose package configと一致。
  fixed32 manifest / persistent episodes / exp408 causeはそれぞれ
  `fbbc62b...` / `031067f...` / `b230ffc...`で事前固定値と一致した。
- Stage 1、inference、submissionは引き続きfail-closed。

### 2026-07-30 初回push 400とslug短縮

- 初回は実験名全体の52文字slug
  `exp443-mean-preserving-trapezoidal-lattice-hmm-train`でprepareしたが、
  `SaveKernel 400 Bad Request`となりkernel実行は開始しなかった。
- 同slugの`kaggle kernels pull -m`は403、同じ認証で既存exp439のpullは成功したため、
  認証不良ではなく未作成かつmetadata/slug長制約と判断した。
- 未作成を確認したうえで、意味要素を残した44文字slug
  `exp443-mean-pres-trapezoid-lattice-hmm-train`と同slugへ解決されるtitleに短縮する。
  旧slugのKaggle Notebookは存在せず、重複はない。

### 2026-07-30 2回目push 400とpackage縮小

- 短縮後slugでも`SaveKernel 400 Bad Request`となり、実行は開始しなかった。
- `kaggle kernels status`は同slugに404、`kaggle kernels files`は空だったため、
  Kaggle Notebookが作成されていないことを確認した。
- canonical trainはself-containedで`src` importや`sys.path`依存がない一方、
  packageには既定動作で不要な`src/`約1.2 MiBが同梱されていた。
- 科学条件、実行契約、kernel id/titleは変えず、Kaggle API requestを縮小するため
  `--no-src`で再packageして同じ短縮slugへpushする。

### 2026-07-30 Stage 0 push成功

- `--no-src` packageは約1.1 MiBとなり、strict experiment validationを再度通過した。
- `kentookumura/exp443-mean-pres-trapezoid-lattice-hmm-train`へ
  private CPU / internet disabled / `run_on_push=true`でversion 1をpushした。
- Kaggleは`KernelWorkerStatus.RUNNING`を返し、Stage 0実行開始を確認した。
- push後のローカルconfigにはversion 1を記録したが、実行中packageのconfigは
  push時点で固定されており、記録追記だけを理由に再package・再pushしない。

### 2026-07-30 Stage 0 terminal result

- Kaggle status: `KernelWorkerStatus.COMPLETE`
- kernel:
  `kentookumura/exp443-mean-pres-trapezoid-lattice-hmm-train`
- version / id_no: `1 / 129095370`
- notebook metrics created: `2026-07-29T22:39:14.394000+00:00`
- scientific variants / HMM well-runs / reporting folds: `1 / 32 / 5`
- parent HMM rerun / ML model / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0`
- Stage 0 elapsed: `5249.411124 sec`（約87分29秒）
- candidate HMM: `5191.461314 sec`
- Stage 1 runtime projection: `125406.237362 sec`
  （上限`30600 sec`を`94806.237362 sec`超過）
- peak RSS: `1.234661 GiB`

technical gate:

- `runtime_projection`だけFAIL。
- conditional mean最大誤差`5.718655e-13 ft`、effective variance最大誤差
  `1.911128e-13 ft²`、rate marginal誤差`0.0`、negative weight `0`、
  support違反`0`、posterior normalization誤差`3.330669e-15`。
- exp439 failure edge、brute-force reference、forward/backward table identity、
  one-step grid mean bias 95%削減、truth-late、readback SHA、finite coverage、
  variance inflation audit、RSSを含む残りのtechnical contractはPASS。
- variance-floor active edges: `9665508`
- variance inflation min / mean / max:
  `0.0 / 0.003905098479 / 0.01561875 ft²`
- exp209 one-step grid mean bias reduction:
  `0.9999999999993123`

mechanism gate:

- 6項目中2項目PASS。
- forward-cause episode SSE reduction:
  `0.0551674072`、必要`>=0.10`でFAIL。
- persistent episode SSE reduction:
  `-0.0576564021`、5.766%悪化してFAIL。
- persistent improved wells:
  `10 / 16`でPASS。
- persistent improving folds:
  `4 / 5`でPASS。
- matched control pooled RMSE:
  parent `3.428436286 ft`、candidate `3.522133884 ft`、
  delta `+0.093697598 ft`でFAIL。
- control by-well delta p95:
  `+1.394367634 ft`でFAIL。
- persistent fold 0だけはSSEが`896156.141`から`3491884.152`へ大きく悪化し、
  他4 foldsの改善を上回った。

判定:

- `stage0_all_gates_pass=false`
- `stage1_eligible_for_separate_approval=false`
- `stage0_fail_closed`
- grid/support/variance floor/noise/rate/emission/gate、blend/selectorの
  same-fixed32 rescueを行わない。
- Stage 1 full OOF、rerun、inference、submissionへ進まない。
- fixed32はmechanism preflightであり、CVやpromotion evidenceではない。

artifact SHA:

- Kaggle `metrics.json`:
  `a1a773a2f967c1ac48e4d041e2df1251e469b991a86e8cdbd963cdcb3e806bfe`
- gate report:
  `38d2d8032f5a52bc419b33f233ceff7f2b3f1ea3d909e64423d6de5143df3ac9`
- numerical contract:
  `e26dbaddc6477e9315e6f59a7a0b9d4da574deda466fc28c518b5b7f8af9c394`
- prediction logical:
  `ef0f61b8c2adc42a52bf1e6c50c1b69f296b774754480a03f8ae11562b0643d7`
- moment audit logical:
  `d0c6cba874437fcf8115b5b144b3d9445c13981e04d97afa7d25d3b222f110ae`
- rate readout logical:
  `3585ca26e5bfc14af73200649da00bd345607c5c5d48a2ffadd34f50a36b339d`

再現性:

- 初回runをdeterministic anchorにはしない。
- runtimeとmechanismが独立にFAILしたため、同一設定rerunによるanchor確立は不要。
- 保存生成物はnegative mechanism evidenceと、0-HMM原因分解候補だけに使う。

post-run local package lock:

- remote Kaggle version 1は変更していない。
- ローカル`kaggle/train`だけを`--no-src`で再生成し、
  `execution.run_hmm=false`、`execution.stage0_run_authorized=false`、
  `runtime.run_approved=false`、`run_on_push=false`へロックした。
- locked packaged notebook SHA:
  `ed64a895b9f67145afc21631de1c328b8442c9aad2f534c0370e30dae517fed5`
- locked `kernel-metadata.json` SHA:
  `e724ae1e7524ddb8d5d3ce75872a4e1c4fa9f10f383c6fa5c96aa8b3f7faaf04`
- locked source / packaged `config.yaml` SHA:
  `cbdb4a4226d76326ef33e6c09f20e700d5229af08eebc2e4334116f2b4c85433`
  （byte-identical）

## 設計契約

- legal rate edgeごとに`mu=0.5*(r_source+r_destination)*dMD-dZ`。
- `v_min=(mu-x_lo)*(x_hi-mu)`。
- `v_eff=max(parent_target_variance,v_min)`。
- 5-cell非負maximum-entropyでsum/mean/v_effを保存する1候補だけ。
- rate marginal、state、emission、prior、grid、readoutは親固定。
- Stage 0 / Stage 1 HMM runs=`32 / 773`、parent rerun 0。
- ML / booster / PF / Beam / GPUは0。
- grid、support、variance floor、sig_p、rate、gateのsame-OOF rescueは禁止。

## 再現性メモ

- RNGなし。edge tableとsolver順を固定する。
- rate marginal、joint edge、variance-floor distribution、prediction、diagnostic SHAを保存する。
- truth-late順序を固定する。
- 初回runをdeterministic anchorとしない。

## 次のアクション

1. ローカルconfig / packageをfail-closed状態へロックする。
2. `experiment_summary.md`と`KAGGLE_DIRECTION.md`へterminal結果を反映する。
3. 必要性と別承認がある場合だけ、保存済みvariance auditを使う0-HMM原因分解を
   別実験として設計する。
