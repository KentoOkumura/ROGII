# exp233_adaptive_outlier_mixture_likelihood_pf セッションノート

## 目的

`backlog/KAGGLE_DIRECTION.md` の backlog `adaptive_outlier_mixture_likelihood_pf` を、
temperature-only の exp232 と独立した mixture-only PF/Beam train-side audit として
実装する。目的は、target-free gate が発火した row のみで state-neutral な outlier
mixture を使い、誤った GR motif に起因する particle collapse を抑えることである。

## 現在の状態

- Route: `pf_beam`
- 状態: `kaggle_train_variant_split_prepared`
- 親: `exp072_exp063_full_replay_feature_cache`
- 比較: `exp232_adaptive_robust_likelihood_pf`、exp214 public raw PF diagnostic
- CV / LB: 未実行
- 推論 / 提出: 明示的に無効

## 実装した内容

- Gaussian likelihood の gate-on update のみを、`L=(1-epsilon)L_gaussian + epsilon
  L_uniform` に置き換えた。`L_uniform=sqrt(2*pi)*gr_sigma/500` は一様 GR support
  `[0,500]` から定まり、row 内の全粒子で同一である。
- gate-off 分岐は exp072 の `exp(-0.5*r^2)` Gaussian update をそのまま使用する。
- exp232 の gate threshold、500 particles、128 seeds、raw GR/typewell GR、transition、
  resampling、seed mean aggregation、score rows を保持した。
- epsilon はユーザー承認済みの `0.02` / `0.05` の 2 variant に固定した。current train
  の raw GR finite range `13.8802..487.0329` を覆う `[0,500]` 外の評価 GR は fail-fast
  する。
- gate rate と mixture application rate を別 diagnostics として保存し、完全一致を
  runtime assertion する。
- exp232 temperature row candidates が利用可能なら id / well / row_idx を one-to-one
  検証して結合し、同一 metrics / bucket / hidden-like / by-well / interval readout に
  含める。並行初回 run では pending を明示し、採用判定を禁止する。

## 実行予定

- active variants: 2 (`mix_eps_0p02`, `mix_eps_0p05`)
- LightGBM configs: 0
- folds: 0
- boosters: 0
- PF budget: 773 eligible wells x 2 variants x 128 seeds x 500 particles
- control / parent 再学習: なし。保存済み exp072 `likpf_mean` を参照する。
- exp232 comparison: user-approved parallel executionのため initial train では pending
  を許容するが、acceptance 前には必須。
- Runtime: CPU-only、GPU disabled、internet disabled、`num_workers=1`

## 再現性

- stochastic components: particle propagation / systematic resampling
- seed policy: well id、variant name、`public_likpf`、seed index から stable SHA256
  seed base を生成する。variant ごとの trajectory は独立だが deterministic rerun が可能。
- output: row candidates gzip は decompressed content SHA を主証拠として記録する。
- deterministic submission anchor ではない。raw-test regeneration、inference、submission
  を生成しない。

## 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp233_adaptive_outlier_mixture_likelihood_pf/exp233_adaptive_outlier_mixture_likelihood_pf_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp233_adaptive_outlier_mixture_likelihood_pf/exp233_adaptive_outlier_mixture_likelihood_pf_train.py
.venv/bin/python -m py_compile experiments/exp233_adaptive_outlier_mixture_likelihood_pf/adaptive_outlier_mixture_likelihood_pf.py experiments/exp233_adaptive_outlier_mixture_likelihood_pf/exp233_adaptive_outlier_mixture_likelihood_pf_train.py
.venv/bin/ruff check experiments/exp233_adaptive_outlier_mixture_likelihood_pf --select F821
make validate-exp EXP=exp233_adaptive_outlier_mixture_likelihood_pf
```

## 次のアクション

ユーザー承認済みの runtime-only variant split として、全 eligible well を対象にした次の
2 CPU kernel を並列 push / 実行する。どちらも control、gate、500 particles、128 seeds、
transition、resampling、Uniform support、epsilon の定義を変更せず、親 / control は再学習しない。

- `mix_eps_0p02`: `train_variant0` →
  `kentookumura/exp233-outlier-mixture-pf-e02`、title `exp233 outlier mixture pf e02`
- `mix_eps_0p05`: `train_variant1` →
  `kentookumura/exp233-outlier-mixture-pf-e05`、title `exp233 outlier mixture pf e05`

各 kernel は full eligible-well surface を単独で評価する。Kaggle output は kernel ごとに
標準の exp233 artifact 名で保存されるため、download 後に同じ directory へ上書きせず、
id / well / row_idx を確認してから集約する。exp232 temperature artifact の接続前に、
mixture variant を採用しない。

## 2026-07-11 Kaggle push preflight

- Jupytext conversion / `--test`、`py_compile`、`ruff --select F821`、
  `make validate-exp EXP=exp233_adaptive_outlier_mixture_likelihood_pf`: PASS。
- package preflight: bootstrap manifest に mixture helper、config、train/inference scripts、
  settings、project.yml を確認。metadata は CPU / internet disabled、competition source と
  exp072 / exp115 kernel source を確認した。
- 最初の full title `exp233 adaptive outlier mixture likelihood pf train` は 51 characters
  で Kaggle `SaveKernel` 400 になった。canonical slug を意味を保った
  `exp233-adaptive-outlier-mixture-likpf-train`、title を
  `exp233 adaptive outlier mixture likpf train` に短縮し、同一実験として再 prepare /
  push する。元 slug は Kaggle 側に存在しないことを pull 403 で確認した。

## 2026-07-11 Kaggle train v1 push

- canonical kernel: `kentookumura/exp233-adaptive-outlier-mixture-likpf-train`
- title: `exp233 adaptive outlier mixture likpf train`
- push: success、Kaggle kernel version `1`
- URL: https://www.kaggle.com/code/kentookumura/exp233-adaptive-outlier-mixture-likpf-train
- metadata pull: success (`/tmp/kaggle-pull/exp233-adaptive-outlier-mixture-likpf-train-v1`)
- initial logs: 空。Kaggle 実行中に CLI logs が空を返す既知挙動として扱い、これだけで
  failure / re-push と判断しない。
- status check: `KernelWorkerStatus.RUNNING`。同一 canonical kernel id のまま監視する。

## 2026-07-11 Kaggle train v1 ERROR

- final status: `KernelWorkerStatus.ERROR`。bootstrap 後、約217秒で PF generation 前に停止した。
- first meaningful traceback: notebook `In [4]` の control contract check。
  `RuntimeError: The saved exp072 likpf_mean control is required for this experiment.`
- 原因: exp072 ML feature cache は row-level `likpf_mean` を保存しない。これは exp232 v1 と
  同じ input contract error であり、mixture kernel、Numba、memory、runtime timeout の失敗ではない。
- ユーザー承認により、exp209 enriched cache の `hmm_mean_tvt - hmm_minus_likpf_mean` から
  reconstructed control を作る。id / well / target / last_known_tvt / md_since の strict
  one-to-one alignment と finite guard を実装し、exp072 full artifact exact parity が未証明
  であることを result / metrics に残す。
- v2 は同じ canonical kernel `kentookumura/exp233-adaptive-outlier-mixture-likpf-train` に
  version を追加する。active variants、particle / seed / transition / resampling、CPU-only
  runtime、control non-regeneration は変更しない。
- local PF kernel unit test は local `.venv` に `numba` がないため import 前に停止した。
  Jupytext / syntax / F821 / strict experiment validation は PASS。Numba 実行は Kaggle
  CPU runtime の kernel output で確認する。

## 2026-07-11 Kaggle train v2 起動

- package preflight: canonical id/title slug 一致、CPU / internet disabled、competition source、exp072 / exp115 / exp209 kernel source、bootstrap 内の reconstructed-control helper と mixture config を確認した。
- same canonical kernel `kentookumura/exp233-adaptive-outlier-mixture-likpf-train` に Kaggle version `2` を追加した。
- current status: `KernelWorkerStatus.RUNNING`。`mix_eps_0p02` / `mix_eps_0p05` の2 variants、LightGBM config / fold / booster `0 / 0 / 0`、CPU-only、control / parent 再学習なし。
- exp232 と並行実行中だが、exp232 artifact の id-aligned temperature comparison が完了するまで mixture output を採用しない。

## 2026-07-11 Kaggle train v2 中断（timeout 報告）

- ユーザーから timeout の報告を受け、canonical kernel
  `kentookumura/exp233-adaptive-outlier-mixture-likpf-train` の最終 status を確認した。
  結果は `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。
- 最終 `kaggle kernels logs` は debugger warning と `Prepared 8 Kaggle support files from zip bootstrap.`
  までで、notebook cell の出力、mixture PF progress、traceback はない。したがって利用可能な
  証拠は「実行が中断された」ことまでであり、Kaggle の時間上限か手動 cancel かは識別できない。
- PF metrics、row artifact、SHA は得られていない。control、gate、particle / seed、Uniform support、
  epsilon variants は変更していない。
- 再実行は runtime 対策として variant 別または deterministic target-well shard 別に分ける必要が
  ある。どちらも比較意味を変えないが、実行構成の変更なのでユーザー判断後に実装する。

## 2026-07-11 variant 別 full-surface retry を実装

- ユーザー承認: `mix_eps_0p02` と `mix_eps_0p05` をそれぞれ別の Kaggle CPU kernel へ
  分け、full eligible-well surface のまま並列再実行する。
- `exp233_adaptive_outlier_mixture_likelihood_pf_train_variant0.py` は `mix_eps_0p02` のみ、
  `..._train_variant1.py` は `mix_eps_0p05` のみを実行する。両方とも base config の
  2 variant 定義を先に検証し、deep copy に
  `execution.active_outlier_mixture_variants: [selected]` を設定してから input / PF contract
  check と実行を行う。したがって並列 prepare / push 中に共有 config を書き換えない。
- 実行数: 2 PF variants（各 kernel は 1）、LightGBM configs / folds / boosters は各々
  `0 / 0 / 0`、CPU-only、parent / Gaussian control 再学習なし。各 run の PF budget は
  `773 eligible wells x 1 variant x 128 seeds x 500 particles`。
- summary / metrics は `execution.active_outlier_mixture_variants` と full-well execution を
  記録する。完走判定は両 kernel の output と exp232 temperature artifacts の id-aligned
  comparison 後にのみ行う。

### 実装・package preflight

- PASS: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb` と `--test` を
  `train_variant0.py` / `train_variant1.py` に実行し、対応する `.ipynb` を生成した。
- PASS: `py_compile`（mixture helper と両 variant source）、`ruff --select F821`、
  `make validate-exp EXP=exp233_adaptive_outlier_mixture_likelihood_pf`。
- PASS: `prepare_kaggle_notebooks.py --strict --run-on-push` により次を生成した。metadata は
  両方とも private CPU、GPU / internet disabled、exp072 / exp115 / exp209 kernel source を持つ。
  - `kaggle/train_variant0`: code file
    `exp233_adaptive_outlier_mixture_likelihood_pf_train_variant0.ipynb`、kernel
    `kentookumura/exp233-outlier-mixture-pf-e02`、title `exp233 outlier mixture pf e02`
  - `kaggle/train_variant1`: code file
    `exp233_adaptive_outlier_mixture_likelihood_pf_train_variant1.ipynb`、kernel
    `kentookumura/exp233-outlier-mixture-pf-e05`、title `exp233 outlier mixture pf e05`
- `train_variant0` / `train_variant1` は独立 package のため、並列 push / run 時にも
  `config.yaml` を相互に書き換えず、各 notebook のメモリ内 deep copy だけを選択する。

## 2026-07-11 parallel split kernel 起動待ち

- `mix_eps_0p02` / `mix_eps_0p05` を同時 push したが、Kaggle は account-level
  `Maximum batch CPU session count of 5 reached` を返し、いずれも kernel を作成・起動しなかった。
- 空き枠確認後の同一 ID 再 push は `Notebook not found`、metadata pull は一時的な
  `GetKernel` 500、その後 status は 404 だった。slug / title は変更せず、Kaggle API / CPU slot が
  安定したら同じ canonical ID を再試行する。
- ユーザーの他の実行を停止していない。exp232 `temp_t2` / `temp_t4` が RUNNING のままなので、
  mixture 2 variant は CPU slot 待ち。実装・package preflight は完了済みである。

## 2026-07-12 canonical kernel v3 再実行

- `temp_t2` の完了で CPU slot が1つ空いた後も、専用 kernel ID
  `kentookumura/exp233-outlier-mixture-pf-e02` / `...-e05` は `Notebook not found` のまま作成できなかった。
  title/slug は変更していない。
- 既存 canonical kernel `kentookumura/exp233-adaptive-outlier-mixture-likpf-train` を pull して
  id/title、private CPU、internet disabled、competition と exp072/exp115/exp209 sources を照合した。
- `train_variant0`（`mix_eps_0p02` だけを選択する notebook）を同 kernel の version `3` として
  push した。Kaggle status は `KernelWorkerStatus.RUNNING`。PF/control/gate/particle/seed/eligible-well
  surface は変えていない。
- `mix_eps_0p05` は同じ canonical kernel の次 version として v3 complete 後に実行する。専用 ID package
  は維持するが、Kaggle backend が作成できるまで slug を増やさない。

## 2026-07-12 mix_eps_0p02 v3 complete / mix_eps_0p05 v4 start

- canonical v3 `mix_eps_0p02` は COMPLETE。3,783,989 rows / 773 wells、runtime
  32,369.848s。exp209 reconstructed control RMSE 11.594898 に対し RMSE 13.519963
  （+1.925065）、`1000_plus` RMSE 14.759210、最大 well regression +44.485364 で、
  overall / long-tail / worst-well guard をすべて破った。
- gate / mixture は同じ target-free rule で gate-any-seed 4,641 rows、gate-all-seed 659 rows。
  sampled interval coverage は 0.226543 で、control の interval がないため coverage 改善は主張しない。
- dedicated e05 kernel は status/log API が 404 のまま再 push も `Notebook not found` だった。
  canonical kernel v3 の完了後に同じ ID/titleを pull して `train_variant1` を v4 として push し、
  `mix_eps_0p05` は RUNNING。e02 v3 output は保持しており、v4完了後にだけ id-aligned 比較を行う。

## 2026-07-13 mix_eps_0p05 v4 complete・train-side 不採用

- canonical kernel `kentookumura/exp233-adaptive-outlier-mixture-likpf-train` v4 は
  `KernelWorkerStatus.COMPLETE`。`mix_eps_0p05` は 773 wells / 3,783,989 rows を完走し、
  final metrics、row candidate、by-well、gate、interval、SHA を取得した。runtime は 24,345.640s。
- exp209 reconstructed control RMSE 11.594897672 に対し、ε=0.05 は RMSE 13.550173069
  （+1.955275396）、`1000_plus` 14.799418368、max well regression +46.711288333。
  gate all-seed は 677 rows、any-seed は 4,662 rows。overall / long-tail / worst-well guard をすべて破った。
- ε=0.02 v3 も RMSE 13.519962657（+1.925064985）、`1000_plus` 14.759209729、
  max well regression +44.485363908 で同じ guard を破っている。mixture 内での最小悪化であっても、
  baseline または採用候補ではない。
- checkpoint-free exp232 `temp_t2` v2 output と e05 v4 は exp072 input、exp209 reconstructed
  control、schema content SHA が一致する。ID-aligned 比較で e05 は T=2 の RMSE 13.529887109 より
  0.020285960 悪い。したがって temperature を mixture で置き換える根拠もない。
- 結論: `adaptive_outlier_mixture_likelihood_pf` は train-side 不採用。epsilon 再 grid、raw-test
  regeneration、inference、submission は行わず、次は `adaptive_likelihood_pf_trajectory_containment_audit`
  を優先する。
