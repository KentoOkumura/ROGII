# exp239_distribution_matched_multicut_pseudotail 結果

## 仮説

固定3 quantileではなくofficial-start分布へ合わせたearly-start multi-cutoffを使うことで、
現行exp218系のofficial-start OOFとhidden-like頑健性を改善できるか検証する。

## 設定

- 親: exp218 CV 8.475793752 / Public LB 7.843
- historical augmentation: exp023 control 13.494554 / best 12.942938
- 検証: source-well GroupKFold、official-start OOF primary
- cutoff: quantile、fixed hidden rows、GR change、GR missing boundary、trajectory curvature
- matching: deterministic marginal-deficit quota
- 最終学習: official 3,783,989 rows weight 1.0 + pseudo 799,961 rows weight 0.5、LightGBM 3 config x 5 folds = 15 boosters

## 結果

Kaggle CPU audit v1 `kentookumura/exp239-distribution-matched-multicut-train` はCOMPLETE。
約48秒、CPU、LightGBM 0 config、booster 0、親/control再学習なし。

- input: 773 wells / official evaluation 3,783,989 rows
- candidate cutoffs: 11,123
- selected cutoffs / replay requests: 1,546 / 1,546、全773 wells
- estimated augmentation: 1,545,558 rows / ratio 0.408447、cap 0.45以内
- hidden-like: metadata valid 204 wells / selected valid 204 wells
- leakage contract: 全項目pass
- maximum marginal share delta: 0.210220 (`prefix_rows`)
- mean absolute marginal share delta: 0.041871
- source: fixed 704、quantile 607、GR change 181、missing boundary 43、curvature 11

Kaggle CPU audit v2は同じcanonical kernelのversion 2でCOMPLETE。global quotaへ修正後は
800 cutoffs / 617 wells、hidden-like 204/204、augmentation ratio 0.211407、max marginal差
0.030344 / mean 0.004408となり、事前定義したdistribution/leakage guardをすべてpassした。

## 再現性

- deterministic prediction anchor: false。予測を生成しないmanifest段階。
- seed policy: sorted wellとSHA256 stable tie key。global RNGなし。
- cutoff/fold/distribution/replay manifestのcontent SHAとschema SHAを保存する実装。
- gzipはdecompressed content SHAを主証拠とする。
- kernel: `kentookumura/exp239-distribution-matched-multicut-train` v2。
- selected cutoff SHA: `3eb8e1776387cc73596fd5faa53c2e76bd1dce14306570df41eb4e8e2625d6c6`
- replay request SHA: `710bc9e694ccb67c896a3f82657495dd61f2ed7999d336f5f91696f9fbfde26f`
- model / prediction / submission SHA: 対象外。

## 解釈

manifest coverage、fold、leakage、runtime、SHAは成立した。しかし、全wellへ2 cutoffを必ず
割り当てるround-robin制約により、official targetのprefix rows最下位binが10.09%から31.11%へ
過剰化した。分布matchedと呼ぶには差が大きいため、このmanifestでexp218特徴再生成やGPU学習へ
進まない。

## 次

v2 global quotaを実装した。v1候補に対するread-only preflightは800 cutoffs、617 wells、
hidden-like 204/204、augmentation ratio 0.211407、max marginal差0.030344で事前guardをpass。
Kaggle CPU audit v2でも同値を再現し、全distribution/leakage guardをpassした。selected cutoff SHAは
`3eb8e1776387cc73596fd5faa53c2e76bd1dce14306570df41eb4e8e2625d6c6`、replay request SHAは
`710bc9e694ccb67c896a3f82657495dd61f2ed7999d336f5f91696f9fbfde26f`。次はprefix依存特徴の
段階的再生成としてv3 prefix materializationを実装した。各request最大1,000行の決定的sampling、
anchor/prefix統計、target分離、request/fold/row-cap guard、feature/schema SHA保存を含む。静的検証はpass、
Kaggle CPU v3はversion 3でCOMPLETE、約96秒。800 requestsから799,961行・50列を生成し、
推定memory 528,026,392 bytes、gzip 68,757,754 bytes。全materialization guardをpassした。
feature decompressed content SHAは`cb6c7f401d88ecb9ac133d0ea035bbe626c54c2a7260aad62c7d0b4a989afa89`で、
download後の再hashとも一致した。full exp218 GPU学習にはまだ進まない。

v4 CPU residual probeはversion 4でCOMPLETE、約212秒。799,961 rows / 44 features、
1 config x 5 folds = 5 boostersを学習した。overall RMSEはanchor hold 69.526871、
anchor + delta-z 156.505994に対しresidual probe 24.349143で、全distance bucketでも改善した。
一方、617 wells中210 wellsが悪化し、worst `86454a6f`は42.424906から105.840567へ悪化、
max regression +63.415661で事前guard +20を超えた。したがってdirect residual routeは不採用とし、
親exp218学習、推論、提出へ進まない。残す場合はcross-fitted confidence / anchor shrinkage材料に限定する。

## v11 full exp218 augmentation 最終結果

CPUでpseudo 32 shards / 799,961 rowsとofficial 16 shards / 3,783,989 rowsを生成し、
共通380-feature schemaを確認した。GPUでは両cacheをdisk-backed memmapへstreamingし、
official weight 1.0、pseudo weight 0.5、official-only validation、outer-valid source-well由来
pseudo除外で3 configs x 5 folds = 15 boostersを学習した。親exp218 controlは再学習していない。

- Kaggle kernel: `kentookumura/exp239-pseudotail-dual-cache-streaming-train` version 1 / COMPLETE
- official-start OOF RMSE: 8.697380066
- saved exp218 OOF RMSE: 8.475793752
- delta: +0.221586314（約+2.61%悪化）
- runtime: 15,371.68秒（約4時間16分）
- peak RSS: 19,509.44 MB
- prediction decompressed content SHA: `16f77eccfb66d6c702ed2b70b33dfedb7544a51e7b00877bff8093371fe17ce9`
- pseudo / official manifest SHA: `ed9e5da6...501d` / `a2c89e0a...27ef`
- feature/schema SHA: `197c7ee8...09b5`

cache row数、schema、SHA、valid-well pseudo除外、booster数は契約どおりであり、OOMも解消した。
したがって悪化は実装失敗ではなく、この比率・weightでdistribution-matched early pseudo-tailを
exp218通常学習へ加える仮説のnegative resultと解釈する。exp239は不採用で完了し、inference、
competition submit、weight微調整には進まない。

## v12 trial submission

ユーザー明示依頼により、negative CVを承知で保存済みv11モデルを1回だけ提出した。再学習は
行わず、exp218のtarget-free raw-test replayへv11の15 boostersを差し替えた。

- inference kernel: `kentookumura/exp239-distribution-matched-multicut-inference` v1 / COMPLETE
- input verification: 15/15 model SHA、ordered 380-feature schemaともにPASS
- submission: 14,151 rows、`id,tvt`、ID順一致、重複0、NaN/Inf 0、fallback 0
- submission SHA256: `81d16997fe50f5e89186906d1fa5f1d70d255b4abc452ff91950fa1b59d5ccee`
- submit-check: PASS
- submission ref: `54720769`
- status: COMPLETE
- Public LB: 7.944
- delta vs exp218 Public LB 7.843: +0.101
- delta vs exp238 ML anchor Public LB 7.775: +0.169
- delta vs exp082 ensemble anchor Public LB 7.601: +0.343

LBも親および現行anchorより悪化し、official-start OOFの悪化方向と一致した。今回の直接混合型
pseudo-tail augmentationは採用せず、weight微調整や同方式の追加提出には進まない。
