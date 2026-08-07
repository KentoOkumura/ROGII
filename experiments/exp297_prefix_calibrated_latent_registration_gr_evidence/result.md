# exp297_prefix_calibrated_latent_registration_gr_evidence 結果

## 結論

固定Stage 2判定は **`FAIL_STOP_NO_STAGE4`**。known prefixで校正したType Well/horizontal GRの
latent-registration posteriorは、exp293固定12候補のtruth-good candidateを識別できなかった。
Stage 3、Stage 4、inference、submissionには進まない。

## 仮説

known prefixで校正したType Well/horizontal GR evidenceをlatent registrationとreliabilityで周辺化すれば、
exp293固定candidate bankのoracle headroomのうちH256で35%以上を回収できる。

## 設定

- 親: `exp293_physics_only_candidate_bank_headroom_contract` version 2
- 検証: exp293 outer foldとH128/H256/H512 non-overlap blockを固定、fitなし
- メトリック: H256 expected candidate SSE headroom recovery
- シード: 42（stable shuffle controlのみ）
- model / trained fold / booster / PF-Beam再生成: `0 / 0 / 0 / 0`
- Kaggle: private CPU version 2、id_no `127897451`

## 変更点

exp293の候補、fold、block、TVT値は固定し、prefix robust-affine calibration、21-state observation
registration、residual/NCC/derivative evidence、reliable/unreliable posterior、matched shuffle controlだけを追加した。
posterior freeze後のtruthはexpected candidate SSEのreadoutにだけ使用した。

## 主要結果

| メトリック | 値 | 判定 |
| --- | ---: | --- |
| H256 anchor RMSE | 8.238332 | 参照 |
| H256 oracle RMSE | 3.552829 | 参照 |
| H256 real posterior expected RMSE | 8.620041 | anchorより+0.381709悪化 |
| H256 real headroom recovery | -0.116476 | `>=0.35`をFAIL |
| H256 shuffle expected RMSE | 8.571583 | realより0.048458良い |
| H256 shuffle headroom recovery | -0.101397 | realがshuffleを下回る |
| H512 real headroom recovery | -0.119000 | H256との差0.002524でcontinuityだけPASS |
| 1000+ expected / anchor RMSE | 9.459663 / 9.042324 | 非劣化FAIL |
| hidden-like spatial expected / anchor RMSE | 9.034820 / 8.748108 | 非劣化FAIL |
| hidden-like typewell-purged expected / anchor RMSE | 8.975787 / 8.694132 | 非劣化FAIL |

H256 real recoveryはfold 0..4で
`-0.089589 / -0.069222 / -0.181848 / -0.158431 / -0.092005`となり、5/5 foldsで負だった。
対応するshuffle recoveryは
`-0.073011 / -0.060739 / -0.164503 / -0.130028 / -0.085741`で、realは5/5 foldsでshuffleより悪かった。

## Evidenceの状態

- 773 wells中704 wellsでprefix calibrationがvalid。69 wellsは
  `prefix_typewell_gr_std_below_minimum`でunreliable-safeへfallbackした。
- H256 15,174 blocksのうちeligible stateを持つ割合は29.5044%。reliable probabilityは平均0.118509、
  中央値0であり、多くのblockがunreliable側へ退避した。
- fallback自体は安全に停止したが、evidenceが使えるblockでもpooled expected SSEを改善できず、
  real GRはmatched shuffleを上回らなかった。

## 技術・再現性確認

- 3,783,989 rows / 773 wells / 105,818 block-controlを完走。runtimeは1,070.800秒。
- truth accessはtarget-free freeze前0回、freeze後773回。
- target-free 8件とpost-freeze readout 4件、合計12ファイルのSHAを取得outputで再計算し全一致。
- candidate content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- truth content SHA: `b0a1bebf24ec925728c40a690147d1820b88d0bf3f403333d9452a79ef179c8d`
- joint posterior SHA: `2f4f443a93491e3dd1d5b87ac239d1802dcff1ddef5ad9584aedb982e5376060`
- selected/corrected TVT prediction、raw-test inference、submissionは生成されていない。
- `All-NaN slice` warningはType Well範囲外などeligible stateなしの箇所で発生したが、
  unreliable-safe fallbackを通って完走しておりfatal errorではない。

## 解釈

exp293は固定12候補内に十分なoracle headroomを持つが、今回のprefix affine校正、residual scale、
raw residual、NCC、chain-rule derivative、21-state registrationの組み合わせは、そのheadroomを
target-freeに識別する観測モデルにはならなかった。負のrecoveryが全foldで再現し、realがshuffleにも負けたため、
単なるcoverage不足だけでなく、利用可能なposterior質量のcandidate順位付けも誤っている。

したがって同じtruth上でregistration grid、component weight、prior、thresholdを救済調整したり、
posteriorをTVTへ直接補正したりする根拠はない。exp297 branchはここで閉じる。

## 次

exp297からStage 3/4は開かず、新しい救済backlogも追加しない。物理routeの次候補は、この観測posteriorとは
独立して事前設計済みのexp298 local-shape source監査とexp295 candidate-free SSMであり、各実験の既存guardと
別承認に従う。
