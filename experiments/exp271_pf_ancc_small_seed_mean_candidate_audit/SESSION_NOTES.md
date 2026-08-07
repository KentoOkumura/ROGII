# exp271 PF ANCC small-seed mean candidate audit セッションノート

## 目的

exp266の固定先頭4/8 seed PF ANCC meanを全train pseudo-tailへ保存し、exp263 core 12へ
追加する価値と4/8の計算量差を、学習なしのcandidate auditで確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train version 2完了、candidate audit支持
- Kaggle package: canonical kernel version 2 `COMPLETE`
- inference / submission: disabled
- CV / LB: 対象外 / 未提出

## 実行量契約

- active PF dynamics variant: 1（PF ANCC）
- particles × generated seeds: 600 × 8（exp266固定seed順の先頭8）
- 固定集約: mean4 / mean8
- 対象: 3,783,989 rows / 773 wells
- LightGBM config / fold / booster: 0 / 0 / 0
- parent/control retraining: なし
- GPU / inference / submission: なし / なし / なし
- 実測runtime: total 1,386.570秒、PF generation 953.056秒。

## 再現性

- `docs/06_reproducibility.md`を確認済み。
- seed 0は`stable_seed("pf_ancc", well)`、seed 1〜7は
  `stable_seed(exp266 namespace, "train", "pf_ancc", well, seed_index)`。
- well threadへ渡す前に整数seedを固定し、Numba kernelへ明示する。
- seed 0のexp072全行exact parity、mean4/mean8のexp266 per-well RMSE parityをfail-closedにした。
- targetはcandidate path生成・gzip保存後にだけjoinする。
- exp263 manifest SHAとcore 12 Parquet 60 partitionsのfile SHAを照合する。
- candidate gzipはraw SHAとdecompressed content SHAを分けて保存する。
- deterministic submission anchorではない。model / submission SHAは非該当。

## 実装

- steering: `.steering/20260717-exp271-pf-ancc-small-seed-mean-candidate-audit/`
- 12章 / 1,500行超のcompact self-contained Jupytext trainを実装した。
- exp266 exact PF ANCC kernel、raw input、固定seed生成、exp072/266 parity、exp263 core bank loader、
  row/block/well oracle、distance/hidden-like/worst-well、seed disagreement、SHA保存を展開した。
- target-free candidate pathにはseed0、mean4/8、seed std4/8、particle std4/8、mean8-mean4を保存する。
- hidden-like assignmentはbootstrap dependencyで既存exp251 inputをSHA固定して同梱する。
- inference notebookはdisabled guardだけで、`submission.csv`を生成しない。
- 親exp266 trainは1,823行 / 12章、本実験も12章を維持し、PF-Z/64-seed occurrence auditを
  省いてcandidate bank auditへ置き換えた。

## 変更点

- exp266のPF ANCC scientific contractは変えず、生成seed数を固定先頭8へ限定した。
- exp266では保存しなかった全row mean4/mean8 pathとseed disagreementを保存対象にした。
- exp263 core 12へのrow/block/well oracleとstress readoutを追加し、selector学習は追加していない。

## コマンドログ

### 2026-07-18 version 2完了確認・生成物監査

- ユーザーから完了連絡を受け、canonical kernel version 2が
  `KernelWorkerStatus.COMPLETE`であることと完全ログを確認した。
- 実行契約は3,783,989 rows / 773 wells、PF dynamics 1、600 particles × 8 seeds、
  LightGBM config / fold / booster 0 / 0 / 0、GPU / inference / submissionなし。
- total runtime 1,386.569822秒、PF generation 953.055951秒、workers 8。
- parity:
  - seed0 max abs diff vs exp072: 0.0 ft
  - mean4 max per-well RMSE diff vs exp266: `7.105427357601002e-15 ft`
  - mean8 max per-well RMSE diff vs exp266: `7.105427357601002e-15 ft`
  - exp263 manifestと60 candidate partition SHA、raw horizontal/typewell各773件を含む
    input manifest 1,611 recordsをfail-closedで通過した。
- standalone RMSEはseed0 14.493051、mean4 13.126896、mean8 13.027107。
- core12追加oracle delta（mean4 / mean8 / both）:
  - row: `-0.046543 / -0.049720 / -0.065252`
  - block128: `-0.045224 / -0.048093 / -0.063204`
  - block256: `-0.043186 / -0.047303 / -0.061414`
  - block512: `-0.041472 / -0.045046 / -0.058807`
  - whole-well: `-0.028392 / -0.036973 / -0.050751`
- row unique-bestはmean4単独252,772（6.6800%）、mean8単独251,635（6.6500%）。
  両方追加時はmean4 165,283 + mean8 175,404 = 340,687 rows（9.0034%）。
- mean8-mean4 path差はmean absolute 1.257438 ft / RMSE 2.400394 ft、
  seed std8とmean8 absolute errorのSpearmanは0.501475。
- Kaggle outputはcandidate gzipを除くsummary、metrics、parity、standalone、oracle、by-well、
  disagreement、input/artifact manifestだけを`kaggle/output/train_v2`へ選択取得した。
- 取得した全CSV/gzipはartifact manifestのraw/decompressed SHAと一致した。summaryと
  `metrics.json`もbyte一致した。candidate gzipは134,426,437 bytesのためKaggleに保持した。
- candidate path SHA:
  - raw: `a01f2082717c17c5c22ef26dc91f7f87cc98cb48e2d4e0c92dc0a9a0b922590a`
  - decompressed: `a7c48204d6782e62941e433b5d47ba5e03f6e441b8e601461be1c63ebcdca336`
  - schema: `9037c3e40cd7a4ad8535479dcad7ee16885c2214940a6c357915e8ec8b2a5ba9`
- input manifest SHA: `faea0fdeac2017a4f98456f3f57515442caa2d00e21d2cafd275a394950e2ce4`
- artifact manifest SHA: `3c697683083cb921250dd2e9a00003c85054be5e05087e13ee6db88b92aa7632`
- 判断: branchを維持する。単一candidateならmean4へ縮約するが、保存済みpathで行う次の
  add-only selector監査では相補性のあるmean4/mean8とdisagreementを両方残す。
- 結果記録後のtargeted tests 6件、repository全124件、strict experiment validation: PASS。

### 2026-07-17 Kaggle CPU実行承認

- ユーザーから本実験の実行依頼を受領した。
- push前に実行量を再確認した。PF dynamics variant 1、600 particles × 固定8 seeds、
  mean4 / mean8、LightGBM config / fold / booster 0 / 0 / 0、親/control再学習なし。
- 同じcanonical kernelを`run_on_push=true`で1回実行する。GPU、inference、submissionは使わない。
- 初回フル実行はKaggle CPUを正とし、seed0 / exp266 / exp263 SHA guardの通過を必須とする。
- canonical kernel version 1をpushし、実行開始を確認した。

### 2026-07-17 kernel version 1失敗と修正

- version 1は約11分で773 wells / 8 seedsのPF生成を完了後、exp266 mean4 per-well RMSE
  parity guardで停止した。最大差は`0.00045923159395044877 ft`、設定許容値は`2e-6 ft`。
- seed0 exact parityは通過しており、PF kernel、seed、演算順の差ではない。
- exp266 aggregateはraw horizontalのevaluation `TVT`をfloat64で評価する一方、version 1は
  exp072 cacheのfloat32 `target + last_known_tvt`から復元した近似TVTを評価truthに使っていた。
- 許容値は緩和せず、target-free candidate gzipを書き終えた後にraw horizontal `TVT`を
  float64で別読みし、parity / standalone / oracle / disagreementの評価だけへjoinするよう修正した。
- exp072 cacheはidentity、md_since、seed0 pathの固定参照として継続利用する。target truthには使わない。
- version 1はfail-closedで終了し、監査metrics / submissionは生成していない。
- 修正後のpy_compile、Ruff、対象test 6件、Jupytext round-trip、strict validationを通し、
  同じcanonical kernelへversion 2をpushした。
- version 2 package SHA:
  - config（push時）: `9b0d28eca9a67dba5f1e0aa454db5b3228f3a1e178383674144d0af7351761f4`
  - train source: `86973bcf245d410a950e2925251007ba72a5236a83f6586462276703425c64d2`
  - canonical notebook: `16c719db3ca6294bda05bdafd7183f959d6c264718fea217b24a03261366accd`
  - bootstrap付きpackage notebook: `20f9842bb3f9427e970c83fabccb3379cc93bdf1ef2bc81a584e76685e46734c`
- ユーザー依頼によりこちらの定期監視を停止した。最後の確認は2026-07-17 13:37 UTCで
  `KernelWorkerStatus.RUNNING`。Kaggle kernel version 2自体は停止していない。

### 2026-07-17 作成・実装

```bash
make new-steering EXP=exp271_pf_ancc_small_seed_mean_candidate_audit
make new-exp EXP=exp271_pf_ancc_small_seed_mean_candidate_audit
.venv/bin/python -m py_compile <train.py> <inference.py>
.venv/bin/ruff check <train.py> <inference.py> tests/test_exp271_pf_ancc_small_seed_mean_candidate_audit.py
.venv/bin/pytest -q tests/test_exp271_pf_ancc_small_seed_mean_candidate_audit.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py/inference.py>
make validate-exp EXP=exp271_pf_ancc_small_seed_mean_candidate_audit
make validate-template
```

- 初回scaffoldは作業開始時点の次番`exp268`で作成したが、同じworkspaceへ並行して
  `exp268_multi_scale_initial_rate_candidates`、続いてexp269/270が追加されたことを全体検証で検出した。
  既存実験を保持し、本実験は未実行の段階で`exp271`へ改番した。仮説・科学設定・source内容は不変。

- targeted tests 5件、repository全98件: PASS。
- py_compile、Ruff、Jupytext train/inference round-trip、strict experiment validation、
  template validation: PASS。
- 全体testで既存exp266 contract testのNumba shimが`sys.modules`へ残り、並行追加されたexp268
  contract testを汚染する問題を検出した。exp266 test内でshimを必ず解除するようtest isolationだけを修正し、
  exp266科学コード・設定・生成物には触れていない。
- local notebook実行と実データPF生成は行っていない。初回フル実行はKaggle CPUを正とする。

## 初回Kaggle package監査

- kernel id / title:
  - `kentookumura/exp271-pf-ancc-small-seed-mean-audit-train`
  - `exp271 pf ancc small seed mean audit train`
- metadata: private CPU、GPU/TPU/internet off、competition source 1、kernel source 3、
  `run_on_push=false`。
- canonical / loose package SHA一致:
  - config: `f9055b11bdd733e8a030c2d595da5751b31610286f0b24ecb5e3ee89b3f613ae`
  - train source: `f9709495587a6afd4c17c7412cda48aafc7d0dc22fd51dfdb2e90d1905bcae34`
  - canonical notebook: `c25c35c3a3f4562961f59b2306127be6d06ebfa7f73229bff32ca8a942f85f33`
  - bootstrap付きpackage notebook: `c9925bfdb0339a13acc61057cfd44f653e3e41b370d630581826dea4ec737bac`
- bootstrap manifestはhidden-like assignmentを
  `inputs/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv`として含み、
  SHA `5f9ac9fa...6597`が正規inputと一致した。
- この節は初回`run_on_push=false` package時点の記録。後続でversion 1/2をpushし、
  successful version 2を完了した。competition submissionは行っていない。

## 次のアクション

1. 保存済みmean4/mean8 pathとseed/particle disagreementを、exp263/exp264系selectorへ
   add-onlyで加えるfold-safe OOF監査を別候補とする。PFは再生成しない。
2. 単一candidateのraw-test計算量契約はmean4へ縮約する。次監査でmean8依存のsame-run gainが
   確認できた場合だけ8 seed生成を再検討する。
3. oracle routing、raw-test inference、submissionはsame-run control、hidden-like、worst-well guardを
   通るまで行わない。
