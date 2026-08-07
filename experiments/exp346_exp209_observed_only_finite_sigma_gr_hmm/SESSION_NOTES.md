# exp346_exp209_observed_only_finite_sigma_gr_hmm セッションノート

## 目的

exp209のraw missing行を一切変えず、元からfiniteなGR行だけfinite-only scaleへ狭めることで、GR識別力と過信防止を両立できるか検証する設計を固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train version 1完了 / scientific gate FAIL / branch closed
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV: candidate `13.295027` / exp209 control `11.938287` / 改善`-1.356739 ft`
- LB / submission: なし
- 実行実績: 1 variant / 773 HMM well-runs / model・LightGBM・fold・booster・PF・Beam・control再実行すべて0
- compact self-contained train / fail-closed inference: 実装済み
- Kaggle CPU train package/push/run: version 1 COMPLETE
- raw-test inference、submission: 未承認・未実施

## 2026-07-22 設計確定

```bash
make new-steering EXP=exp346_exp209_observed_only_finite_sigma_gr_hmm
make new-exp EXP=exp346_exp209_observed_only_finite_sigma_gr_hmm
```

- `kaggle-strategy`で現行バックログ、exp307/308/337、exp339/341を照合した。
- `kaggle-review-exp`に従いsteeringを先に作り、次にexperiment scaffoldを作成した。
- 「確実」を補間前raw `GR`がfiniteの行だけと定義した。learned confidenceやthresholdは使わない。
- `sigma_obs`はknown-prefix finite residualのpopulation std、`sigma_base`はexp209 zero-fill std。両方`[10,60]`へclipする。
- raw finite evaluation行だけ`sigma_obs`、raw missing行は`sigma_base`を使う。finite pair 20未満/nonfinite時は`sigma_base`へno-op fallbackする。
- exp209のGR interpolation、Type Well、Gaussian `-0.5*min(z²,600)`、state grammar、transition、prior、posterior meanを固定した。
- 保存済みexp209 HMM/LikPFをcontrolに使い、再実行しない。

## 既存実験との関係

- exp307はfinite scaleを全行に適用して悪化した。本実験はraw missing行をexp209へ戻す。
- exp308はfinite-MAD親からmissing行をdownweightする依存実験で、親FAILにより閉鎖済み。本実験はexp209直系で再開ではない。
- exp337は追加構造分散を全wellへ適用してFAILしたが、finite-only forward NLLが良いという独立根拠を残した。
- exp339/341はmissing補間分散を測る/広げる枝。本実験はobserved行を狭めるため依存しない。

## 実行量ガード

- schedule audit: 1
- active variants: 1 (`observed_finite_std_missing_exp209_sigma`)
- HMM well-runs: 773
- model / LightGBM configs / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- 親control再実行: 0
- Kaggle GPU: 0、CPU予定、internet off

## 2026-07-22 実装

- ユーザーの`exp346を実装してください`という依頼を、科学実装とNotebook候補作成の承認として記録した。Kaggle実行承認には拡張していない。
- exp307のself-contained exact-HMM kernelを親実装参照として、exp346固有の単一row scheduleへ限定したtrain sourceを実装した。
- 補間前raw GR mask、`sigma_base`、`sigma_obs`、row scheduleをtruth attachment前にgzipへ保存し、schema SHAとdecompressed content SHAを記録する。
- raw missing行では`sigma_row == sigma_base`をassertし、Gaussian emissionのexp209との差が厳密に0であることをwellごとに記録する。
- direct、5 folds、raw observed/missing、missing-fraction 3 bucket、距離3 bucket、hidden-like 2面、by-well p95/worst、fixed LikPF 50:50を読む固定gateを実装した。
- finite pair 20未満/nonfiniteではobserved側も`sigma_base`へ戻し、well全体がscale上no-opになる。
- inference notebook候補は、train-side gate PASSと別承認までは常に例外停止するfail-closed実装とした。
- 実装時点の予定実行量は1 variant / 773 HMM well-runs / schedule audit 1 / model・LightGBM・trained fold・booster・PF・Beam・control再実行各0のまま。

## 実装検証

実行済み:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py> <inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <inference.py>
.venv/bin/python -m py_compile <train.py> <inference.py>
.venv/bin/ruff check <train.py> <inference.py> tests/test_exp346_*.py
.venv/bin/pytest -q tests/test_exp346_exp209_observed_only_finite_sigma_gr_hmm.py
make validate-exp EXP=exp346_exp209_observed_only_finite_sigma_gr_hmm
make validate-template
make test
```

- Jupytext round-trip: train / inferenceともPASS。
- py_compile / Ruff: PASS。
- 専用contract test: `11 passed`。
- strict experiment validation / template validation: PASS。
- 全体test: 610 collected、`605 passed / 3 skipped / 2 failed`。2 failureは既存`exp296`の完了済みconfig（statusとrun flag）に対してtestがKaggle実行前状態を期待している不一致で、exp346のファイル・契約とは無関係。exp346 testは全11件PASS。
- 親exp209にはcompact self-contained sourceがないため、同じexact-HMM実装系のexp307 compactを構成比較に使用した。exp307 train 1,717行に対しexp346 train 1,901行で、10個の役割章をすべて維持し、raw mask scheduleと追加scope/gate分が増えている。
- Notebookは変換とround-trip検証のみで、ローカル科学実行、Kaggle package/push/runは行っていない。

## 再現性メモ

- `docs/06_reproducibility.md`を確認した。
- RNGなし。well ID、row、single variant、aggregation順を固定する。
- exp209 HMM cache content SHA: `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- exp072 LikPF cache content SHA: `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`。
- exp226 fold/truth content SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- raw mask、scale audit、row schedule、prediction、metricsはschema SHAとdecompressed content SHAを保存する設計。
- model/submission SHAは非該当。本実験をdeterministic submission anchorとは扱わない。
- Kaggle package時はmetadataとbootstrap内config/source SHAを照合する。

## 次のアクション

## 2026-07-22 Kaggle CPU実行承認

- ユーザーの`実行してください`という依頼を、exp346 train-side Kaggle CPU package/push/runの承認として記録した。
- 実行対象: `observed_finite_std_missing_exp209_sigma` 1 variant。
- 実行量: schedule audit 1、773 HMM well-runs、model config 0、LightGBM config 0、trained fold 0、booster 0、PF/Beam run 0、親control再実行0。
- 保存済みexp209 HMM/LikPFを比較基準として読み、controlを再生成しない。
- canonical kernel: `kentookumura/exp346-observed-only-finite-sigma-gr-hmm-train`。
- CPU、internet off、`outer_workers=2`、Numba threads 2。raw-test inferenceとsubmissionは未承認のまま。
- compact trainを正規`exp346_exp209_observed_only_finite_sigma_gr_hmm_train.ipynb`へ採用し、compact/canonical SHA一致`97aaf8e3...afbfb`を確認した。
- package metadataはcanonical id/title一致、private、CPU、internet off、run-on-push true、competition source 1件、kernel source 3件を確認した。
- package notebook SHA: `ac800eab3bc6ac95ffa86af4021f2d35bef140f450f7e326b8c8d437fbbf5819`。
- bootstrap config SHA: `fbeb8c60ad6ac619ec05dbcc22c764505c0e8c9ade58a4c2acf92c073226a577`。local / manifest / ZIP内configの3者一致を確認した。
- bootstrap内でも1 variant、773 HMM runs、booster 0、control再実行false、run schedule/HMM true、inference/submission falseを確認した。
- 初回候補`kentookumura/exp346-exp209-observed-only-finite-sigma-gr-hmm-train`はslug/titleが53文字で、Kaggle `SaveKernel`が400を返した。同IDへのmetadata pullは403で、kernelが作成されていないことを確認した。
- Kaggle CLI公式changelogにtitle/slug長のvalidationが明記され、手元の成功済みkernelが50文字以内であることから、長さ制約による失敗と判断した。仮説名を保ち、冗長な親exp番号だけを落とした46文字のcanonical IDへ修正した。実験仮説・入力・実行量は変更していない。
- 短縮後packageでlocal/package/ZIP/manifestのconfig SHA `aef66471a83c858d901efe8ce2be04bce6aea2f4ce944d916b7254d1a0f374c7`が一致し、train source SHAもpackage/manifestで`bc7534c3463f83fdaef0318f0e896a6cd98cbab74c5d1ac6dd355e24ced88f7a`一致を確認した。
- kernel version: `1`、id_no: `128227279`、URL: `https://www.kaggle.com/code/kentookumura/exp346-observed-only-finite-sigma-gr-hmm-train`。
- push後のmetadata pullで正しいcanonical ID、private、CPU、GPU/TPU/internet off、competition source 1件、保存済み親kernel source 3件を再確認した。

## 次のアクション

同じversion 1 kernelを完了まで監視し、固定gate結果とSHAを記録する。空logsや一時的status失敗を理由にre-pushしない。raw-test inferenceとsubmissionはtrain-side gate PASS後も別承認まで行わない。

### 監視停止

- ユーザーの依頼により、2026-07-22にローカルの60秒間隔status監視だけを停止した。
- Kaggle上のversion 1実行は停止していない。完了連絡後に同じkernelのlogs/outputを確認し、固定gate判定と記録を再開する。

## 2026-07-23 Kaggle CPU version 1完了

- ユーザーの完了連絡後、同じcanonical kernel version 1のstatusが`COMPLETE`であることと、完了logsを確認した。再pushは行っていない。
- 実行は1 variant / 773 HMM well-runs / 3,783,989 rows、runtime `17,757.849174 sec`。model / LightGBM config / trained fold / booster / PF / Beam / control再実行はすべて0。
- technical gateはPASS。finite coverage 1.0、ID mismatch 0、fallback 0%、raw missing emission schedule差0.0、posterior正規化最大誤差`2.8866e-15`、exp209 HMM/LikPF/fixed blend parity差はいずれも`1e-5 ft`以内だった。
- direct overallはexp209 control `11.938287235`に対しcandidate `13.295026644`、改善`-1.356739409 ft`。改善foldは1/5で固定条件4/5をFAILした。
- raw observed / raw missingはそれぞれ`-1.647067 / -0.710366 ft`。high missing fraction `-0.450547`、1000+ `-1.416738`、hidden-like spatial / typewell-purged `-1.729533 / -1.502915 ft`で、必須non-regression scopeをすべてFAILした。
- fixed LikPF 50:50もcontrol `10.269692505`に対しcandidate `10.531117870`、`+0.261425365 ft`悪化した。
- well別は380改善 / 393悪化。p95差`+1.162673 ft`、worst `be83e781`は`+70.766418 ft`で上限`+0.25 ft`を大幅超過した。
- promotion decisionは`observed_only_finite_sigma_failed_close_without_rescue`。technical PASSかつscientific FAILのため、仮説棄却として信頼できる。

### 原因解釈

- exp209 zero-fill sigma中央値`38.641808`に対しobserved finite std中央値は`13.895676`、比中央値`0.369701`。raw observed evidenceを強めすぎた過信が主因である。
- raw missing行のemissionはexp209と厳密一致したが、同一wellのobserved行で変えたposterior evidenceがexact-HMM smoothingを通じてmissing行へ波及し、missing scopeも悪化した。row-local scale固定はseries-global posteriorの安全性を保証しなかった。
- well別sigma比とRMSE deltaのSpearmanは`0.1441`、縮小幅との相関は`-0.1895`と弱く、post-hocな単純threshold救済の根拠もない。

### 再現性とoutput確認

- small metrics/gate/manifestのみ`--file-pattern`で一時取得し、large prediction 86,976,146 bytesとschedule 21,003,806 bytesは取得していない。
- promotion gate SHA: `eaad34eb1889e67d9b623a67d15094349fa756180b91ba2c503e78d6bb0f4c96`。
- metrics SHA: `ada0d67a8c5b7facbd171b3b14a090dd83247da45a0dcb5a73cf99db911671db`。
- prediction decompressed SHA: `f6888ff8755d64fa72a3d8b23f0949a72b9b448b98eea898f9d9b028326fc74b`。
- schedule decompressed SHA: `79473f9e56ebc5aa6b61b54f219603c2f175c9d1bfc296ad081844b40ec62796`。
- scale audit raw / decompressed SHAはローカル再計算とKaggle summaryで`d01f02f8...b584 / 892b49f2...b067`一致。

## 次のアクション

branchを閉じ、sigma/confidence/emission/HMM/blend救済、raw-test inference、submissionを行わない。完了済みexp346をバックログから削除する。同familyの新規救済は追加せず、独立0-boosterの既存exp340をP1--P2として維持する。GR evidenceの重複過信を再訪する場合も、低優先の既存exp343 Stage 0でACF安定性を先に検証し、gate PASSと別承認なしにHMMへ進めない。
