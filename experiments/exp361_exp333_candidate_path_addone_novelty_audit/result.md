# exp361_exp333_candidate_path_addone_novelty_audit 結果

## 状態

Kaggle private CPU version 2を完了した。technical guardとcandidate novelty guardは
ともにPASSし、判定は`exp333_candidate_path_novelty_supported`となった。

## 固定設定

- 親: `exp333_exp226_k16_segment_residual_offset_target`
- 追加候補: 保存済み `exp333_segment_offset` OOF 1本
- 比較bank: exp293 fixed deployable12
- novelty契約: exp302 と同じ H512 / whole-well / strict unique-best / 4-of-5
- 実行量: 1 candidate、5 fold readouts、0 booster、再学習0
- Kernel: `kentookumura/exp361-exp333-candidate-novelty-train` version 2
- 実行時間: audit完了 `234.280 sec`、最終log `242.407 sec`
- inference / submission: 未実行

## Direct参考値

direct scoreはhard gateに使わず、保存結果のparityだけを確認した。

| candidate | pooled RMSE | exp226差 | 改善fold | 用途 |
| --- | ---: | ---: | ---: | --- |
| exp333 segment offset | 9.076677 | -0.350433 ft | 5/5 | context / parity |

exp228 `8.944086`、exp263 `8.238332`は参考値であり、この監査の合否条件ではない。

## Candidate novelty結果

exp293 fixed deployable12へexp333をadd-oneした結果:

| 指標 | fixed12 | exp333 add-one | 改善 / 比率 | 条件 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| H512 oracle RMSE | 3.683763 | 3.550659 | +0.133104 ft | >=0.03 | PASS |
| whole-well oracle RMSE | 4.784904 | 4.682772 | +0.102132 ft | >=0.02 | PASS |
| H512 strict unique-best | - | - | 11.5064% | >=2% | PASS |
| H512改善fold | - | - | 5/5 | >=4/5 | PASS |

H512 fold別改善は`+0.247155 / +0.186671 / +0.081279 / +0.072752 /
+0.093295 ft`で、全fold正方向だった。fixed12との予測相関は高く
（最大は`exp226_w500_50_50`との`0.9999669`）ても、局所block単位では追加価値がある。

## Technical・再現性

- 3,783,989 rows / 773 wells、finite coverage 1.0、duplicate 0。
- evaluation truth access before freeze: 0。
- exp333 source file SHA:
  `70b623d4c839c4f7eb11fb2134aa214ca8f0ce8d6ebe65e723d2fffa95dcc2dc`。
- exp333 source decompressed SHA:
  `f2ebc6f6ea243b45fdb785342b8815b3b04947f96d787d3017e5e2be7ff92e5a`。
- exp361 prediction content / decompressed SHA:
  `e9bb5e9f0689facf7d7aa468dda89ca776c11bbec1c97cc02f74ab992a016450` /
  `ed7b7a7f281b5d8d0b43b7007a5f640408312f4ed08abd1e417a3372bccd6bff`。
- candidate bank content SHA:
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`。
- block assignment decompressed SHA:
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`。
- truth content SHA:
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`。
- SHA manifest file SHA:
  `4dadbd76efbdf6c2af86fe9e9fb13b8b132c211afd7c5c03b900dca222ae0bef`。
- 取得した小容量artifactはmanifest対象10件すべてfile SHA一致。
- stochastic処理、model、booster、inference、submission、oracle prediction保存はない。

## 解釈

exp333をexp228/exp263の単体置換として見ると不採用だが、候補パス改善という目的では
fixed12に対する相補性が明確に確認できた。したがって、exp333のcurrent-test候補生成を
実装する価値はある。ただしoracle headroomはdeployable selector性能ではなく、exp333を
単独採用・平均blend・提出してよいことを意味しない。

## 次

別承認があれば、保存済み5 fold modelと同一feature contractを使うcurrent-test inferenceを
新しい実験番号ではなく`exp333`内に実装する。まずmodel/prediction parityと14,151行の
candidate artifactを作り、submissionは作らない。固定12への組み込み方はその後の
target-free selector/blend readoutとして別途設計する。
