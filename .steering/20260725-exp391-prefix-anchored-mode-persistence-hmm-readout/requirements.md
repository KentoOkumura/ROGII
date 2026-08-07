# 要件

## 依頼

異なる posterior mode の間で、Type Well TVT は不連続に切り替わり得る一方、
物理モデルの出力が徐々に正解から外れて一定offsetへ収束する現象を切り分ける。
同じwell・同じrow identityで次を重ね、どの段階がrampを作るかを判定する。

- exp209 exact HMM の posterior mean
- marginal MAP
- global Viterbi
- top-2 marginal mode の TVT、mass、basin conditional mean
- exp226 K16 の projection 前後
- exp263 の固定物理候補 `exp226_w500_50_50`

切り分けでHMM内のmode averaging / transitionが支持された場合だけ、prefixから
引き継いだmode identityを保持し、別modeへ移ったpathを最終候補から除外する
anchor-mode persistence readoutへ進む。

## 制約

- Routeは`pf_beam`とする。
- 親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とし、
  exp209のabsolute-TVT grid、emission、transition、prior、sigma、posterior-mean controlを固定する。
- exp236はtop-2抽出規則の参照、exp270は保存済みmean / MAP / Viterbiの参照とする。
  exp236はexp221 posterior、exp270はexp209 posteriorであるため、両者を同一posteriorとして
  joinしてはならない。同じHMMのtop-2 massはexp209を再生した同一passから抽出する。
- mode identityはrowごとのmass rankでは定義しない。top1/top2のmass順位が入れ替わっても、
  transition overlapと前rowのmode IDで同一modeを追跡する。
- `jump_used`だけでは小さいstepの累積による別mode移動を検出できないため、
  `anchor_mode_id`、`current_mode_id`、`mode_switch_count`を保持する。
- 最終候補に残せるのは`current_mode_id == anchor_mode_id`かつ
  `mode_switch_count == 0`のpathだけとする。別mode laneは診断用で、平均・blendしない。
- Stage Aのevent抽出、mode追跡、候補生成、logical SHA freezeまではunknown-suffix truth、
  error、hidden-like roleを参照しない。truthはfreeze後のscoreだけにlate joinする。
- Stage A0は保存済みartifactだけ、Stage A1は固定16 wellsのCPU preflight、
  full 773-well HMMはpreflight全PASSと別承認後だけ実行する。
- LightGBM / fitted model / PF / Beam / boosterは0、GPUは使用しない。
- exp226、exp263、exp209のcontrolを再学習・再生成しない。full HMM passで再計算する
  posterior meanは必要なposterior抽出の内部parity列であり、別control variantではない。
- PF / Beamへのmode flag移植、K16変更、ML feature化、current-test inference、
  submissionは本実験の対象外とする。
- threshold、mode数、jump penalty、blend weight、対象wellを同一OOF truthで救済しない。
- 再現性は`docs/06_reproducibility.md`に従い、入力、decoder contract、mode ledger、
  predictionのlogical/decompressed content SHAとKaggle kernel versionを記録する。
- 初回設計ターンでは設計文書と実験scaffoldだけを作った。2026-07-25の後続ユーザー指示
  `exp391を実装してください`をimplementation-only承認とし、compact self-contained
  notebook候補と専用testを実装する。A0 / A1 / BのKaggle実行承認には拡張しない。

## 受け入れ基準

- 原因切り分けが同一well・同一row・同一exp209 posteriorで比較でき、
  exp236とexp270を誤って同一posteriorとして扱わない。
- posterior averaging、transition kernel、K16 projection、fixed blendの各原因について、
  target-freeな分類規則と重複時の扱いが事前固定されている。
- prefix anchorからのmode ID、row間mode matching、cross-mode edge、
  no-switch candidate、fail-closed fallbackが一意に定義されている。
- Stage A0 / A1 / Bの実行量、停止条件、科学gate、禁止事項が事前固定されている。
- 16-well preflightは5 reporting foldsからtarget-freeに固定選択し、選択manifest SHAを
  truth join前に保存する。
- full runへ進む場合も、1 HMM variant / 773 wells / LightGBM config 0 /
  trained fold 0 / booster 0 / PF 0 / Beam 0 / parent replay 0である。
- gzip生成物はraw archive SHAではなくdecompressed content SHAを主証拠にする。
- deterministic anchorを主張する場合は、同一設定rerunでmode ledgerとpredictionの
  logical SHA一致を確認する。
- inference / submissionは無効であり、別承認なしに有効化できない。
