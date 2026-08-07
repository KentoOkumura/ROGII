# exp391_prefix_anchored_mode_persistence_hmm_readout 結果

## 状態

Kaggle private CPU Stage A1 version 3（id_no `128527913`）を固定16 wellsで完了した。
technical gateとmechanism gateがともにFAILしたため`fail_closed`とし、Stage B、
inference、submissionへ進まない。

## 仮説

posterior modeのmass crossoverまたはtransitionがramp-to-offsetを作っている場合、
prefixから継承したstable mode identityを保持し、cross-mode edgeを通ったpathを
除外すれば、posterior meanより物理的に一貫した候補を得られる。

ただしK16 projectionまたはfixed blendが原因なら、HMMのmode persistenceでは
解消しない。そのためStage A0で対象をtruth-freeに固定し、Stage A1で同じposteriorを
再生して原因を切り分けた。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: pf_beam
- Stage A1: 1 diagnostic pass / 16 HMM wells
- LightGBM config / trained fold / booster / PF / Beam / GPU: 0
- parent-control retraining / replay: 0
- candidate: `prefix_anchor_no_switch_conditional_mean`
- gate: HMM内原因60%以上かつ4/5 folds、same-pass parity `1e-5 ft`以下、
  posterior normalization誤差`1e-8`以下、full換算30,600秒以下

## 結果

| メトリック | 値 |
| --- | --- |
| Stage A0 | PASS |
| Stage A1 | FAIL_CLOSED |
| Kaggle kernel | version 3 / id_no 128527913 |
| Stage A1 HMM wells | 16 / 16 |
| kernel runtime | 18,105.382秒（約5.03時間） |
| HMM runtime合計 | 18,008.710秒 |
| projected 773-well runtime | 870,045.814秒（約241.68時間） |
| peak RSS | 4.132145 GB |
| selected events | 19 |
| HMM-supported events | 1 / 19（5.2632%） |
| HMM-supported folds | 1 / 5 |
| cause counts | posterior averaging 1 / transition 0 / K16 0 / fixed blend 3 / unresolved 15 |
| max same-pass parity diff | 0.350000000000364 ft |
| max posterior-mean diff | 0.26953125 ft |
| max posterior normalization error | 2.4567824e-05 |
| candidate active rows | 0 / 78,866 |
| forbidden reads before freeze | 0 |

PASSしたgateはdecoder event数、mode-ledger key重複なし、mode identity衝突なし、
projected RSS。FAILしたgateはsame-pass parity、posterior normalization、
projected runtime、HMM-supported event比率、HMM-supported fold数。

## 再現性

- deterministic anchor: false（成功rerunによるSHA再現確認は行わない）
- input manifest SHA:
  `935d7fc6178279d846d83133a24c43d5abde90b0020f56b27683ca131ba29a6a`
- event manifest logical SHA:
  `30dae154edf9cb5bdc353378649c3fdd38bf3592000e0feabfbdc2083565cd09`
- preflight manifest logical SHA:
  `f02d5cc034b7d313fe9f3d33d1ef516f33e2d382a33679cfa7ad00164b5868ab`
- decoder contract manifest SHA:
  `486370c0912ec4569b50aefb14e2c1cfcd9c1705a4d1189c178367426cfd1de6`
- mode ledger logical / decompressed SHA:
  `a15b1a88eab1dfce0fbb9fbd23fdaccdb443d3e7666e27b3cb960f1a0afbe334` /
  `ee4a56f32c2ae541e54a7798fa64d0b5d2faef1605c05e8bbe5563edb3b61a5d`
- candidate prediction logical / decompressed SHA:
  `111b91597c512afb318df919bf80c62d415ed42ba4f93e11c920d1c86b57916a` /
  `bcf8b522094fd979f3e6f134309f95b6d3fa40e58d52174e3edbb647e43f807b`
- posterior row summary logical / decompressed SHA:
  `46b55f4a5a8d4fffca9f88c3a86d02b3d2d78f40700a8c7650004ef8eec7e2ca` /
  `dd86c3c96189d772b0e84efbb59f11d78b9e77bdab39c9be997c2942ca468c8c`
- path ledger logical / decompressed SHA:
  `232706a2f14e2d3ef30b062a7259abe959929190e0112c519b47098358c67c16` /
  `6a89626d744d562f0bc2d93672c431f381eb56206cf4359a531b06e230de3220`

## 解釈

primary仮説は支持されなかった。HMM内部原因は19 events中1件、1 foldのみで、
事前条件の60%・4 foldsから大きく不足した。15 eventsは未解決、3 eventsは
fixed blend支持であり、prefix-anchor no-switch decoderを773 wellsへ広げる根拠がない。

さらに16 wellsすべてで保存exp209とのsame-pass parityがfail closedし、candidate
78,866行は全行saved exp209へfallbackした。full換算も約10.07日で上限8.5時間を
大幅に超える。threshold、matching、normalization tolerance、decoder実装、fallback、
blendを同じOOFで救済せず、本branchを閉じる。

## 次

Stage B、inference、submissionは実行しない。exp391 Stage A1 PASSを先行条件としていた
exp395 left/right mode consensusも閉じる。再開には、今回のgateを緩和しない独立した
HMM内部原因の証拠と、新しい実験設計・実行承認が必要。
