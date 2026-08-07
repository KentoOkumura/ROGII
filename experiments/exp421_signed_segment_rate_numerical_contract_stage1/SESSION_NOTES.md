# exp421_signed_segment_rate_numerical_contract_stage1 セッションノート

## 目的

exp418のtechnical FAILを変更せず、truth-freeな`1e-10 ft` numerical contractを
事前固定した後継実験で、同一signed K16 rate Stage 1を実行する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 1 v5完了、scientific FAILでbranch閉鎖
- 親: exp418
- base: exp226
- CV / LB: まだなし
- inference / submission: 未承認

## 実行量

| variant | config | folds | boosters | exp226 fit | control再学習 | GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 5 | 5 | 0 | 0 | 0 |

- PF/HMM/Beam再生成: 0
- 親exp418 Stage 0再実行: 0
- 保存exp333 nested predictionを再利用する。
- 診断v2: variant 0 / config 0 / fold 0 / booster 0 / truth column 0。

## 固定した後継契約

- exp418 summary file SHA:
  `07c719e0f174b1712650563620f6331504dbd1333969c8777f41ce46419dc412`
- exp418 rate-target content SHA:
  `5c936b03e86e7250afdfef551e796e0beead22d50715d025b84acb9b13a9e2ff`
- exp418 decisionは`FAIL_CLOSE_BRANCH`のまま。
- 唯一のtechnical failureは`integration_parity`。
- 観測差`6.295408638834488e-12 ft`を、固定上限`1.0e-10 ft`で判定する。
- fixed lengths `[1, 2, 17, 257, 4097]`とfixed symmetric 16-rate vectorによる
  truth-free synthetic auditを学習前に実施する。
- exp418のtarget / feature / fold / model / sample weight / Stage 1 gateは不変。

## 再現性

- Kaggle private CPU、internet off、GPU off
- LightGBM `random_state=0`、`deterministic=true`、`force_col_wise=true`
- `n_jobs=num_threads=8`
- exp333 nested / fold / feature schema、exp226 OOF、exp072 cacheをSHA検証する。
- feature freeze、5 model SHA、segment / OOF content SHA、summary SHAを保存する。
- current-test rerun parityはないためdeterministic submission anchorとは呼ばない。

## 承認

2026-07-28:

- ユーザーが、exp418をPASSへ書き換えず`1e-10 ft`数値契約を持つ後継実験で
  Stage 1を実行する方針を明示承認した。
- 承認範囲: scaffold、実装、正規train Notebook、package、Kaggle Stage 1 push/run。
- inferenceとsubmissionは承認範囲外。

## コマンドログ

- `make new-steering EXP=exp421_signed_segment_rate_numerical_contract_stage1`
- `make new-exp EXP=exp421_signed_segment_rate_numerical_contract_stage1 SOURCE=experiments/exp418_exp226_signed_segment_rate_residual`
- exp418 compact sourceを後継self-contained sourceとしてコピーし、数値eligibility
  とsynthetic auditを追加した。
- 親compact比較: exp418 2,349行 / 12章、exp421 2,502行 / 12章。
  既存の入力freeze、basis、feature、LightGBM、metrics/SHA orchestrationを保持し、
  parent eligibilityとtruth-free numerical auditを追加した。
- `__file__`参照は0件。
- `.venv/bin/pytest -q tests/test_exp421_signed_segment_rate_numerical_contract_stage1.py`
  → `15 passed`
- exp418 + exp421専用test: `29 passed`
- Jupytext round-trip、`py_compile`、Ruff full: PASS
- `make validate-exp EXP=exp421_signed_segment_rate_numerical_contract_stage1`:
  strict PASS
- `make validate-template`: PASS
- `make prepare-kaggle-notebooks ... --strict`でtrain packageを生成した。
- metadataはprivate / CPU / internet off / run-on-push。
- kernel sourcesはexp072 / exp333 / exp226 / exp418の4件。
- package configとbootstrap manifestのconfig SHAはともに
  `a203de3921073c0f394470084a6827dcd67839bec06daeb3a9d863c2fb16629b`。
- embedded configは`selected_stage=stage_1`、5 boosters、parent summary SHA、
  numerical upper bound`1e-10 ft`を保持する。
- 初回canonical候補
  `kentookumura/exp421-signed-segment-rate-numerical-contract-stage1-train`
  はid/title slugを一致させたが58文字で、`SaveKernel 400`となった。学習は0
  boosterのまま開始されていない。
- 同slugを`kaggle kernels pull -m`で確認すると403で、kernel作成は確認できなかった。
- Kaggle slug長制約へ収めるため、同じexp421内で意味を保つ43文字のcanonical名
  `kentookumura/exp421-rate-numerical-contract-stage1-train` /
  `exp421 rate numerical contract stage1 train`へ短縮して再packageする。
- short canonical packageのconfig / embedded config SHAは
  `fbd2d9b60390d38926e201f95da17e073287e2d8ebf8989533bdf3c5db8079c5`
  で一致した。
- `make push-kaggle-train EXP=exp421_signed_segment_rate_numerical_contract_stage1`
  → short canonical kernel version 1をpushし、Stage 1を開始した。
- Kaggle version 1は約578秒後、numerical contractとexp418 eligibilityをPASSし、
  LightGBM開始前の`reconstructed exp333 row-feature content SHA mismatch`で停止した。
  booster / exp226 fit / control再学習はいずれも0で、科学CVは生成されていない。
- v1 kernel id_noは`128915070`。exp333 v1 id_no `128116592`とdocker image digest
  `gcr.io/kaggle-images/python@sha256:dafd4ce...40b9`が一致した。
- exp333実行時sourceと現行exp228 sourceのfile SHAはともに
  `8cf163688daa8a6ed4cf4d9169b66b039a04d167df4d45ac881335bb5ee6e155`。
- exp333実行時と現行exp072 cacheは、file SHA
  `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`、
  decompressed SHA
  `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`、
  schema file SHA
  `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
  が一致した。
- exp333 runtime configとexp421のu-projection / GRWR / allowlistも一致した。
- 同じkernelのversion 2は、真値・保存exp226予測・hidden-like assignmentを読まず、
  feature source/cache/schema、projection/GRWR summary、unsorted/sorted row SHAを
  保存する0-booster診断として実行する。
- 診断実装後の専用pytest: `16 passed`、py_compile / Ruff: PASS。
- v2 package configとembedded bootstrap config SHAはともに
  `9eb7f1cc28e939077205ebaf824b66dbd918cd2bfa25bd64048f2eb38a8018ee`。
- embedded診断source SHAは
  `c404b8291c713f7dcc442aa18bb8c114a23e4561f58315fd2d02087bc80f675a`。
- v2 metadataはshort canonical id/title、private / CPU / internet off /
  run-on-push、exp072 / exp333 / exp226 / exp418の4 kernel sourceを維持する。
- `make push-kaggle-train EXP=exp421_signed_segment_rate_numerical_contract_stage1`
  → same canonical kernel version 2のpush成功。診断を実行中。
- Kaggle version 2は`COMPLETE`、notebook log終端は約354.9秒、診断処理のsummary
  出力は約346.9秒。variant/config/fold/booster/truth/exp226 fit/control再学習は全0。
- exp072 file/decompressed/schema、exp228 source、exp333 feature schema、
  projection summary、GRWR summaryのSHAはすべてcanonical期待値と一致した。
- unsorted / sorted row SHAはともに
  `d8e9320cfb6fce9f51b277b2fcb19380b9dbbb3f08fd53c5ba35308e1b5b3048`。
  exp333 canonical full-train v1ログも同じ値を記録している。
- v1で期待していた
  `9475721131bfd93a036d0a636d473a8cf6cc8d7d46eaf203b3879ccba6272a79`
  はexp333 candidate inference v2の3 current-test wellsに対するrow SHAであり、
  773 train wellsのSHAではなかった。
- target/features/folds/model/gateを変えず、exp421のexpected train row SHAだけを
  exp333 canonical full-trainログ値へ訂正した。
- v2小規模成果物とlogは`artifacts/feature_sha_debug_v2/`へ保存した。
- Stage 1 v3 package configとembedded bootstrap config SHAはともに
  `d39fbcb95d9fcc276d5739f4924c0146579a7860b5fe4b64b0cad074debc3628`。
- embedded v3 source SHAは
  `15f9231c1a2084609570c9a9d67e3f5bf69a9bf2ab3759bb29ccedf3056f60e2`。
- embedded configは`selected_stage=stage_1`、1 variant / 1 config / 5 folds /
  5 CPU boosters、exp226 fit/control再学習/PF/HMM/Beam/GPU各0を保持する。
- `make push-kaggle-train EXP=exp421_signed_segment_rate_numerical_contract_stage1`
  → same canonical kernel version 3をpushし、固定Stage 1を実行中。
- Kaggle version 3は約651.4秒で
  `reconstructed exp333 feature-freeze SHA mismatch`となった。row feature SHA
  guardは通過し、停止はLightGBM開始前なのでbooster / exp226 fit / control再学習は0。
- v2でfeature surface全体はcanonical一致済みのため、v4はsaved exp333
  fold manifest / segment assignment / nested exp226 predictionとouter-valid parity
  だけを確認する。feature surface再生成 / truth / model / boosterは全0。
- v4 package / bootstrap config SHAは
  `acfc81880b378e7c72772afd0ac78aa13da7602fe983f706fab15c6516886102`、
  embedded source SHAは
  `d504b79e74ed235cb09a29c7e7f763a057ea14e867527f7b15781e1bc3c63e88`。
- Kaggle version 4は`COMPLETE`、診断summary出力約110.4秒、notebook終端約120.4秒。
  fold manifest SHA、segment assignment SHA、outer-valid parent parityはexp333
  canonical実行時と完全一致した。
- nested predictionだけは、exp333実行時のin-memory SHA
  `575a2aae9991a229b80432b056cb6e5fdd21734629d03943ed131b1ac97df216`
  に対し、manifestでfile/decompressed SHAを固定した保存CSV再読込後は
  `8140e7ad165eacb0234e2ccd034d725f1b74273dc284a97d42a32401728188bc`
  だった。
- exp421が再利用するのは保存CSVなので、expected nested SHAを保存境界の値へ訂正。
  feature surfaceと他componentを固定したaggregate feature-freeze SHAは
  `13dd89dc9d438e190b9d781bbd1fdf2c919329f2bf3f59436b917e6d1690db28`。
- v4成果物とlogは`artifacts/frozen_input_sha_debug_v4/`へ保存した。
- Stage 1 v5 package / bootstrap config SHAは
  `04404d1985ae6a9a536106494b0948dac05db89a59c04c009cace5730e140644`、
  embedded source SHAは
  `c9940264c66688dc99765c5b238c9695047176a9da04d7655addda408bb51651`。
- same canonical kernel version 5をpushし、約1,176秒で`COMPLETE`。
- 5 modelを完走。best iterationはfold 0..4で`951 / 686 / 573 / 715 / 720`。
- pooled RMSEはexp226 `9.427109597`から`9.405572476`へ
  `0.021537121 ft`改善したが、固定上限`8.894085501`に届かなかった。
- fold別deltaは`+0.205410 / +0.035848 / -0.283577 / +0.096284 /
  -0.142552 ft`で、改善2/5 < 必須4/5。
- near `-0.056097`、hidden spatial `-0.099659`、hidden typewell
  `-0.091574`、boundary `-0.108806 ft`は改善。
- 1000+ `+0.003414`、by-well p95 `+0.513310`、worst well
  `+10.467233 ft`は悪化。
- rate targetは5/5 foldsでzero priorより改善し、rate sign balanced accuracy
  `0.5822–0.6005`、first-row anchor、integration parityもPASSした。
- gateは8 PASS / 7 FAIL、decisionは`FAIL_CLOSE_BRANCH`。
- summary / model manifest / OOF content / segment prediction SHAはそれぞれ
  `6ac237ce...eaf26` / `cde79f53...697e2` / `0d0b35cd...7ca3f` /
  `7f97bf0a...f11005`。
- 小規模metrics / evaluation tables / manifests / logは`artifacts/stage1_v5/`
  へ保存。大容量OOF archiveは取得していない。

## 次

同一OOF上の救済、inference、submissionは行わない。再訪時は保存predictionだけを
使う累積drift attributionを独立実験・別承認で設計する。
