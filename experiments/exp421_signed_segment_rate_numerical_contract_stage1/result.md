# exp421_signed_segment_rate_numerical_contract_stage1 結果

## 状態

Kaggle Stage 1 version 5を5 CPU boosterで完走した。CVは`9.405572476`で
exp226比`-0.021537121 ft`改善したが、固定gateは8 PASS / 7 FAILとなり
`FAIL_CLOSE_BRANCH`。inference / submissionへ進まない。

## 仮説

exp418の`6.3e-12 ft` integration parity差はfloat64演算順による無害な差である。
truth-freeに`1e-10 ft`を固定すれば、exp418を再分類せずにsigned K16 rateの
学習可能性を正しく評価できる。

## 一因子差分

- 親exp418のStage 0判定はtechnical FAILのまま維持する。
- fixed synthetic numerical auditと`1e-10 ft` eligibilityを追加する。
- signed-rate target、K16 basis、136特徴、fold、LightGBM、sample weight、
  Stage 1 scientific gateは変更しない。

## Kaggle v1

- kernel: `kentookumura/exp421-rate-numerical-contract-stage1-train`
- version 1 / private CPU / internet off / GPU off
- 約578秒でERROR
- exp418 eligibilityとtruth-free `1e-10 ft` numerical audit: PASS
- 停止点: `reconstructed exp333 row-feature content SHA mismatch`
- 実行済みbooster / exp226 fit / control再学習: 0 / 0 / 0
- 科学CVは未生成

## 診断v2結果

- 真値、保存exp226予測、hidden-like assignmentを読まない。
- exp072 cache file/decompressed/schema SHA、exp228 source SHA、exp333 feature
  schema SHA、projection/GRWR summary SHA、unsorted/sorted row SHAを保存する。
- 0 variant ×0 config ×0 fold = 0 booster
- Kaggle version 2: `COMPLETE`、約354.9秒
- exp072 file / decompressed / schema SHA: 全一致
- exp228 source SHA: 一致
- exp333 feature schema / projection summary / GRWR summary SHA: 全一致
- 実row SHA: `d8e9320c...b3048`
- exp333 canonical train v1ログのrow SHA: `d8e9320c...b3048`
- v1期待値`94757211...a79`は3井戸current-test inferenceのSHAだった。
- 結論: 特徴生成差ではなく、trainとtestのSHA参照範囲取り違え。期待train SHAだけを
  canonicalログ値へ訂正し、target/features/folds/model/gateは変更しない。

## 診断v4結果

- version 3はcanonical row SHA通過後、feature-freeze aggregate SHAで0 booster停止。
- version 4は0 feature / 0 truth / 0 boosterでfold、segment assignment、parent
  parityの一致を確認した。
- 唯一の差はexp333 in-memory nested SHAとmanifest検証済み保存CSV再読込SHAの境界。
- 保存境界SHA `8140e7...`とaggregate freeze SHA `13dd89...`を固定してv5へ進んだ。

## Stage 1 version 5

- kernel: `kentookumura/exp421-rate-numerical-contract-stage1-train`
- version 5 / private CPU / internet off / GPU off / 約1,176秒
- 1 variant ×1 config ×5 folds = 5 boosters、exp226 fit / control再学習: 0 / 0
- pooled RMSE: exp226 `9.427109597` → exp421 `9.405572476`
  （`-0.021537121 ft`）
- 改善fold: 2/5（fold 2: `-0.283577 ft`、fold 4: `-0.142552 ft`）
- 悪化fold: 3/5（fold 0: `+0.205410 ft`、fold 1: `+0.035848 ft`、
  fold 3: `+0.096284 ft`）
- near / hidden spatial / hidden typewell / boundaryは改善した。
- 1000+は`+0.003414 ft`、by-well p95は`+0.513310 ft`、
  worst wellは`+10.467233 ft`悪化した。
- rate target RMSEは5/5 foldsでzero priorより改善し、rate sign gateもPASSしたが、
  累積TVT改善へ安定転移しなかった。
- numerical contractはPASS。実積分差最大`1.268319e-12 ft < 1e-10 ft`。
- failed gates: pooled、exp228比gain、exp333比gain、4/5 fold改善、1000+、
  by-well p95、worst-well。

## 検証

- 専用pytest: `17 passed`
- Jupytext / py_compile / Ruff: PASS
- Stage 1 summary SHA: `6ac237ce...eaf26`
- model manifest SHA: `cde79f53...697e2`
- OOF prediction content SHA: `0d0b35cd...7ca3f`
- rate target SHA: `5c936b03...a9e2ff`
- 小規模metrics / manifest / logは`artifacts/stage1_v5/`へ保存した。

## 次

同一OOF上のclip/shrink/taper/gate救済は行わずbranchを閉じる。再訪するなら、
保存segment/row OOFだけを使う0-boosterの累積drift attributionを独立実験として
設計し、rate-space改善が長距離TVTとwell-tailへ転移しない原因だけを確認する。
