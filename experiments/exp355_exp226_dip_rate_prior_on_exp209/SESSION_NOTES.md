# exp355 セッションノート

## 目的

旧exp323のrate-change仮説をfailed chainから分離し、exp209直結の0-HMM Stage 0として固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 1 version 2完了、scientific gate FAIL、branch closed
- CV: direct `11.291976616`、Stage 1 5/8 scientific checks PASS
- LB: なし

## 2026-07-23 設計

- ユーザー依頼によりexp355を採番し、steeringとscaffoldを作成した。
- parentはtrusted exact-HMMのexp209に固定した。
- Stage 0はdiagnostic 1 / 5 reporting folds / HMM・model・trained fold・booster各0。
- Stage 1予約は1 variant / 773 HMM runs / parent-control再実行0。
- 設計時点では実装、Notebook採用、Kaggle package/push/run、inference、
  submissionは行っていなかった。

## 2026-07-23 Stage 0実装

- ユーザーの「exp355を実装してください」をStage 0実装承認として記録した。
- compact self-contained train候補をJupytext percent形式で実装し、同名の
  compact `.ipynb`へ変換した。既存の正規train/inference Notebookは上書きしていない。
- fail-closed inference候補を実装し、Stage 1、raw-test生成、prediction、
  submissionを明示停止した。
- exp226 OOF、exp209 trusted control cache、raw well identity、exp115 hidden-like
  assignmentのSHA/row/well/fold preflightを追加した。
- pre-truthでK16 geometry ledger、fallback、row-wise rate-prior schedule、
  cumulative diagnostic pathをfreezeし、content SHA一致後だけraw TVTを結合する。
- K16境界はexp226互換row-position `linspace` 16分割、区間rateはfiniteかつ
  正の`ΔMD` stepの中央値、先頭geometry区間invalid時はwell全体fallbackとした。
- fold gateはsegment rate-changeとcumulative pathの双方で4/5改善を要求する。
- Stage 0実行量は1 diagnostic / 5 reporting folds / HMM 0 / model 0 /
  trained fold 0 / booster 0。parent/control再実行0。
- Stage 1予約は1 variant / 773 HMM well-runsのまま未実装・未承認。

## Notebook構成比較

- 親exp209にはcompact self-contained版が存在しないため、親compactとの直接比較は
  非該当。
- exp355 train候補は8章で、runtime/config/SHA、contract、input preflight、
  K16 schedule、truth late-join、readout、gate、orchestrationをNotebook上に展開した。
  同一exp helper importと`__file__`は使用していない。

## 静的検証

- `py_compile`: train / inferenceともpass。
- `ruff --select F821`: train / inference /専用testともpass。
- `pytest -q experiments/exp355_exp226_dip_rate_prior_on_exp209/tests/test_exp355_exp226_dip_rate_prior_on_exp209.py`: 8 passed。
- `jupytext --to ipynb --test`: train / inferenceともpass。
- `make validate-exp EXP=exp355_exp226_dip_rate_prior_on_exp209`: strict pass。
- `make validate-template`: pass。`task`と`jq`は環境に存在しなかったため、
  MakefileとPython標準`json`で同等確認した。
- 初回専用testで16分割不能な35-row例の境界がexp226原式より1行ずれることを検出し、
  `searchsorted(edges[1:], step_idx, side="left")`へ修正後に8 tests passを再確認した。

## 2026-07-23 Kaggle Stage 0実行承認

- ユーザーの「実行してください」を、compact train候補の正規Notebook採用、
  Kaggle CPU package/push/run、Stage 0完了監視までの承認として受領した。
- push前実行量はdiagnostic 1 / reporting folds 5 / HMM well-runs 0 /
  model config 0 / trained fold 0 / booster 0 / parent-control再実行0。
- Stage 1予約の1 variant / 773 HMM well-runs、inference、submissionは未承認のまま。
- canonical kernel id:
  `kentookumura/exp355-exp226-dip-rate-prior-on-exp209-train`。
- title: `exp355 exp226 dip rate prior on exp209 train`。
- `make prepare-kaggle-notebooks ... --run-on-push --no-src --strict`: PASS。
- loose / package / bootstrap内config SHA:
  `2f914152983dcb5240e8574996da60ee98250314d6e9f6fbacbceec9244a83ab`で一致。
- canonical 19 cellsとpackage bootstrap後19 cellsのsource content SHA:
  `7780ea1b763c438369ef613c7ab69a82d95b83f4c3b7b20c4c7cad7ae936b82d`
  で一致。
- 実装SHA:
  - compact train source:
    `a6aab54379754a8c766bc1491cd8f39231697dc2232f5ad3233344d4d2780139`
  - canonical train Notebook:
    `acd792dddbd275d2adf59abfedd7019f244f243315000c86de1b1431a6a4f20e`
  - contract test:
    `da37851827df56e96474e98f89ae79e9fe52e60e19d08c8a362af4e2ad87e55d`

## 2026-07-23 Kaggle Stage 0 version 1

- `make push-kaggle-train EXP=exp355_exp226_dip_rate_prior_on_exp209`: push成功。
- canonical kernel:
  `kentookumura/exp355-exp226-dip-rate-prior-on-exp209-train`。
- version / id_no: `1 / 128366148`。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp355-exp226-dip-rate-prior-on-exp209-train`
- pull metadataでprivate、CPU、internet off、exp115/209/226 kernel sourcesを確認した。
- runtime: `460.872765 sec`。3,783,989 rows / 773 wells / 12,368 K16 segments。
- segment rate-change: baseline `0.018237982`、candidate `0.016710597`、
  gain `8.374744%`、4/5 folds改善。
- cumulative path: baseline `49.493155005`、candidate `46.977325325`、
  gain `2.515829680 ft`、5/5 folds改善。
- 1000+ / hidden-like spatial / hidden-like typewell-purged delta:
  `-2.828214214 / -1.830574021 / -2.056030635 ft`。
- by-well: 439改善 / 334悪化、median `-1.941905 ft`、
  mean `-2.442515 ft`、p95 `+20.585463 ft`。
- worst `071d7b45`: baseline `4.445532`、candidate `73.463200`、
  delta `+69.017669 ft`。上限`+0.25 ft`をFAILした。
- 8 checks中7 PASS。worst-well guardだけFAILし、
  `stage_0_failed_close_without_parameter_rescue`。
- fallback well / segmentは`0 / 0`。truthはschedule freeze後にだけ結合した。
- outputを`kaggle/output/train_v1`へ取得した。13 artifact raw SHAと
  3 gzip decompressed SHAはmanifestと全一致した。
- 実行後は科学式を変更せず、status contractを
  `stage0_completed_failed_worst_well_guard_closed`へ同期し、再実行flagをfalseへ戻した。
  archival compact train source / canonical Notebook / contract test SHAは
  `91a400a666f5acf6b81a072de525871e38807c579cf532e239d6d0d37232eb65` /
  `7556655392b60ca93be08597d72766861b28e8811e0b4eeddcabedf2d4fb80c8` /
  `350fb25bb154aa3933b4aecb222784949b9d8dfa060a588af17664629d893f22`。

## Kaggle生成物SHA

- scientific contract:
  `8972158b5f0e248df0a67241dbb5af5a5cb0ee4d733ad95682590cfe21149e60`
- input manifest:
  `00d351b23a49b843ea0737ba41b0cd523908dc3af4a786080403f12a9a40c1c3`
- frozen schedule logical content:
  `53f9d42bcca0f5596568971b5da6c440114922d0a25b5622592e1b7b50774c85`
- geometry segment ledger:
  `b527d3401e2d730ec883681051c476c929a428e7fc28ed88fff3091045915a39`
- segment rate readout:
  `4d28d0088268bad8818d042abf92a60db6884418ca2c1a2d22b339a68b77d9af`
- path readout:
  `a09c77ae6ed060a63eebdaaafd23649fe46f4062093bbe49ff0781e9eb548107`
- ローカルでfull Stage 0やNotebook実行は行っていない。

## 再現性メモ

- RNGなし。outer-fold、well、segment、row、reduction順を固定する。
- exp226 OOFとexp209 controlのdecompressed SHAをhard guardする。
- geometry ledger、schedule、fallback、readoutのcontent SHAを記録する。
- Stage 1時だけdecoder contractとprediction SHAを記録する。
- deterministic anchorとは扱わない。

## 2026-07-23 Stage 1 user override

- ユーザーの「平均で改善しているのなら次に進んでください」を、Stage 0の
  worst-well guard FAILをoverrideし、Stage 1のtrain-side実装、正規Notebook採用、
  Kaggle CPU package/push/run、完了監視を行う明示承認として記録した。
- overrideは実行許可だけで、Stage 1 promotion gateは変更しない。
- push前実行量: scientific candidate `1`、exact-HMM well-runs `773`、
  reporting folds `5`、LightGBM config `0`、trained fold `0`、booster `0`、
  parent/control再実行 `0`。
- Stage 0 schedule / geometry ledger logical SHA:
  `53f9d42bcca0f5596568971b5da6c440114922d0a25b5622592e1b7b50774c85` /
  `b527d3401e2d730ec883681051c476c929a428e7fc28ed88fff3091045915a39`。
- residual-rate座標を使い、`effective_dz = dz - mu_rate * dMD`としてexp209
  exact-HMMへ入れる。実rateは`mu_rate + q`であり、観測、diffusion、grid、
  momentum、posterior meanを固定する。
- schedule SHA一致後に全well予測をfreezeし、その後だけsuffix truth、saved exp209
  control、fixed LikPF 50:50 diagnostic、hidden-like roleを結合する。
- parameter rescue、raw-test inference、submissionは承認範囲外。

## 2026-07-23 Stage 1 push前検証

- Stage 1 trainは12章、26 cells（markdown 14 / code 12）で、親exp209の
  forward-backward kernelと数値演算が一致し、残差rate座標だけを追加した。
- 合成HMM smokeで4予測すべてfinite、posterior row-sum最大誤差
  `1.0335648e-7`を確認した。
- `py_compile`、`ruff --select F821,F811`、Stage 0/1専用pytest
  `13 passed`、3候補の`jupytext --to ipynb --test`、
  `make validate-exp`、`make validate-template`はすべてPASS。
- canonical trainとStage 1候補の全26 cell source content SHAは
  `684f35798c4523e734960146914e9334a8de32a87370b9be24c10c4834d9d517`
  で一致し、output 0、`__file__`依存なし。
- source / canonical Notebook / Stage 1 test SHA:
  `27c6fe93300e48049f0444668cb6310f0774528469d5bdd6fb1c81d6734fac7d` /
  `3fa0c167bf6ce14f8001772cc98bd5b4d43618336aced168c8c16bf6e886df1f` /
  `3d30e65e27cfb7bb8ba10c8b64fec842a96bff2662b2d9d711cbb93ca792e1ee`。
- `make prepare-kaggle-notebooks ... --run-on-push --no-src --strict`: PASS。
- loose / package / bootstrap内config SHAは
  `cd5a8a8810ef2f99eeef07df48e215f5c6ddf65a7ba3a0966d95ef04e27debce`
  で一致。package bootstrap後の26 cell source SHAもcanonicalと一致した。
- metadataはprivate、CPU、internet/TPU off、exp226/209/115の3 kernel source。
- Kaggle CLI credential preflightはOAuthとlegacy keyを確認した。API Tokenは未設定だが、
  同じOAuth CLIでexp209 output一覧を取得し、parent HMM cacheとenriched LikPF
  controlの両ファイルがcanonical入力に存在することを確認した。

## 次のアクション

parameter/blend/selector救済、再実行、inference、submissionなしでbranchを閉じる。
独立仮説は別実験として扱い、exp355固有の追加backlogは作らない。

## 2026-07-24 Kaggle Stage 1 version 2

- ユーザーの完了連絡後、canonical logとstatusを取得し、
  `KernelWorkerStatus.COMPLETE`を確認した。
- canonical kernel / version / id_no:
  `kentookumura/exp355-exp226-dip-rate-prior-on-exp209-train / 2 / 128366148`。
- private、CPU、internet off。runtime `18,161.789478 sec`、prediction freeze
  `18,052.169336 sec`。
- 3,783,989 rows / 773 wells / 773 HMM runs。candidate 1、reporting folds 5、
  model config / trained fold / booster / parent-control rerunは各0。
- technical gateはPASS。finite coverage 1.0、duplicate 0、truth-before-freeze 0、
  posterior normalization最大誤差`3.9968e-15`、parent direct/blend parity差
  `1.8211e-7 / 3.6401e-6`、schedule/ledger/dependency SHA一致。
- direct: exp209 `11.938287235 -> 11.291976616`、
  `0.646310619 ft`（`5.4138%`）改善、5/5 folds改善。
- direct fold改善量:
  `1.818538117 / 0.383520261 / 0.523300847 / 0.320849660 /
  0.420460031 ft`。
- fixed LikPF 50:50:
  `10.269696317 -> 10.053143746`、`0.216552571 ft`改善、4/5 folds。
  fold 4だけ`+0.238414157 ft`悪化。
- direct 1000+は`-0.730160080 ft`改善したが、100--250 / 250--500は
  `+0.008066 / +0.004951 ft`の微悪化。
- hidden-like spatial / typewell-purgedは
  `+0.414943459 / +0.371719953 ft`悪化。fixed LikPF 50:50でも
  `+0.607564 / +0.590972 ft`悪化した。
- well別は360改善 / 413悪化、paired delta mean / median / p95は
  `-0.488746 / +0.012873 / +5.663043 ft`。candidate/parentのwell-RMSE
  分布p95差は`-2.541374 ft`で実装済みgateをPASSした。
- worst `86454a6f`: parent `5.098001`、candidate `57.841755`、
  delta `+52.743754 ft`。Stage 0 worstの`071d7b45`とは別well。
- scientific gateは8件中5 PASS。FAILはhidden-like spatial、
  hidden-like typewell-purged、worst-well regression guard。
  decision=`stage_1_exact_hmm_failed_no_automatic_inference`。
- scientific contract / input manifest / prediction logical / SHA manifest content:
  `08f5d08576d2bc9ffeca80e6c7c9a8cf752751ba62ed5fd27c49736b6f7ac1ef` /
  `458871d2a1cb80fdd9d4acbefaf35661ae16487b2cf83baafd0a07017c9b3349` /
  `634303f022bced6685367094304da6182fee42815302344469b5919a36cd5e21` /
  `d721f74334ab82f0fd4d95faca6c83002fe5520bf4928b96ab4059091f09371f`。
- output archive全体と3.78M-row OOFは取得せず、summary、gate、fold、
  distance、hidden-like、by-well、fallback、SHA manifest、metricsだけを
  `/tmp/exp355_stage1_v2_metrics`へ選択取得した。manifest対象の取得6ファイルは
  raw SHA全一致。
- inference / submissionは実行・生成していない。

## 2026-07-24 Stage 1実行後同期

- config statusを`stage1_completed_scientific_gate_failed_closed`、
  `run_stage_1=false`、kernel version `2`へ同期した。
- Stage 0/1 sourceのarchival status guardを同じ最終状態へ更新し、Stage 1候補を
  正規train Notebookへ再同期した。candidate/canonical 26-cell source SHAは
  `5b1bad4e4c959cbf8da90168592dabb0a5d0227e4228036be2c1434d30e3d3ba`
  で一致した。
- archival config / metrics / Stage 1 source / canonical Notebook / Stage 1 test SHA:
  `584c7584db48f96a28f895439495e78e1fe05e210e46fef4c5f018926ef8eb8b` /
  `891e56f26e5611243dfe570c96b0c30dbf66f3af5f5e540c4c7d6592f51b3934` /
  `1e56e7ce8c7385010b3bb6ef397aaad96cde5bf83f1cc0097d02335574805eca` /
  `ac1beb7a283967ecf78907d2ba0e587c802faa52d8182a8156e3a9535719e9ac` /
  `a4dbbaa9128844ee71c809fb111ffe3572f5bc7e891eadcdca75f9a78cdf90d4`。
- 実行済み`kaggle/train` packageはversion 2の再現証拠としてpush時config
  SHA `cd5a8a8810ef2f99eeef07df48e215f5c6ddf65a7ba3a0966d95ef04e27debce`
  を保持し、最終archival configで再package/pushしていない。
- final `py_compile`、Ruff F821/F811、Stage 0/1 pytest `13 passed`、
  Jupytext 3候補、strict experiment validation、template validationは全PASS。
- ローカルfull notebook実行は行っていない。
