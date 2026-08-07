# exp391_prefix_anchored_mode_persistence_hmm_readout

## 状態

- ルート: pf_beam
- 状態: Kaggle private CPU Stage A1 version 3 FAIL_CLOSED・branch閉鎖
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-25
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

異なるposterior modeのmassが交差する区間で、posterior averagingまたはmode間の
transitionが徐々にoffsetを作る場合、prefixから継承したstable mode identityを保持し、
一度でも別modeへ移ったpathを除外すれば、ramp-to-persistent-offsetを抑えられる。

ただし、raw HMMではなくexp226 K16 projectionやexp263 fixed blendがrampを作っている
可能性もあるため、同一well上の切り分けを先に行う。

## 変更点

- Stage A0: 保存済みexp270 mean / MAP / Viterbi、exp226 projection前後、
  exp263 fixed candidateを同一rowで重ねる。
- Stage A1: 固定16 wellsだけexp209 posteriorを再生し、同じposteriorからtop-2 TVT /
  mass / conditional meanを抽出して原因を分類する。
- Stage B: Stage A1全PASSと別承認後だけ、transition-overlapでmode IDを追跡し、
  prefix-anchorかつ`mode_switch_count=0`のconditional meanを読む。
- top1/top2のmass順位はmode identityに使わない。
- compact train候補とfail-closed inference候補、専用testは実装済み。
- 正規train Notebookを採用し、Stage A0 target-free censusと固定16-well
  Stage A1を実行済み。Stage B、inference、submissionは未実行。

## 検証方針

- Fold: exp226の保存済みouter 5 reporting folds
- Group: well_id
- Stratification: 5 folds、distance、Stage A cause、hidden-like、by-well
- Leakage Check: event選択、16-well選択、mode追跡、candidate freeze前の
  suffix truth / error / hidden-like role readを0にする
- Gate: Stage A1でHMM内原因がeligible eventの60%以上かつ4/5 folds、
  Stage Bはexp209比0.25 ft以上改善、4/5 folds、tail / hidden-like /
  by-well safetyを全ANDで判定する

## 実行入口

- 正規学習 notebook: `exp391_prefix_anchored_mode_persistence_hmm_readout_train.ipynb`
- compact train候補:
  `exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.ipynb`
- compact inference候補:
  `exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.ipynb`
- 正規train Notebookはcompact self-contained候補を採用済み。正規inferenceは
  templateのまま保持する。
- Stage A0は`kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
  version 2で完了。Stage A1は同kernel version 3で完了しFAIL_CLOSED。
  Stage Bはgate failureにより実行しない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Stage A0 | PASS（1,234 events / 730 wells、固定16 wells選択） |
| Stage A1 | FAIL_CLOSED（HMM支持1/19 events・1/5 folds） |
| technical / mechanism gate | FAIL / FAIL |
| runtime / peak RSS | 18,105.382秒 / 4.132145 GB |
| full 773-well projected runtime | 870,045.814秒 |
| candidate active rows | 0 / 78,866 |

## 所見

### 良かった点

- 原因切り分けを対策実装より先に固定した。
- 3,783,989 rows / 773 wellsのstrict join、全technical gate、
  truth/error/hidden-like事前read 0をPASSした。
- exp236とexp270が別posteriorであることを明示し、誤joinを禁止した。
- 大きなjumpだけでなく、小さなstepの累積によるmode移動も
  `mode_switch_count`で除外する設計にした。

### 悪かった点

- exp236 / exp270ではMAP、dominant mode、Viterbiの直接置換が既にnegativeである。
- mode identityがmerge / splitで未解決になるwellはparentへfail closedするため、
  candidate coverageが不足する可能性がある。
- Stage A1では15/19 eventsがunresolved、fixed blend支持3、HMM内原因支持1のみだった。
- same-pass parity、posterior normalization、projected runtimeもgateをFAILし、
  固定16 wellsの78,866 candidate rowsが全てparentへfail closedした。

### リスク / 注意

- Stage A1がHMM内部原因を支持しなかったため、Stage B HMM decoderは実行しない。
- parity tolerance、normalization tolerance、matching、fallback、blendの救済は禁止。
- threshold、matching、fallback、blendの同一OOF rescueは禁止。

## 次

本branchは閉鎖。Stage B、inference、submissionへ進まない。再開には今回のgateを
緩和しない独立した根拠と、新しい実験設計・実行承認が必要。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
