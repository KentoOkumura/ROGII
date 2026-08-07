# exp439_continuous_kinematic_joint_transition_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical FAIL、no-rescueで完了・閉鎖
- CV / Public LB / Private LB: なし
- 親/control: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- scientific variant: 1
- Stage 0: 予定32 HMM well-runs、最初のwellで0 runs完了のままfail-close
- parent rerun / ML / booster / PF / Beam / GPU: すべて0
- Stage 1 / inference / submission: 実施しない

## 仮説

exp209の持続`(TVT, U-rate)` state、41 rate support、隣接3状態rate
marginal、noise、prior、GR emission、TVT grid、readoutを固定したまま、各legal
rate edgeの位置変位を次へ置き換える。

```text
delta_TVT = 0.5 * (r_source + r_destination) * delta_MD
          - delta_Z + eta_p
```

各edgeは5、7、9セルの順で最小の実行可能supportを選び、非負
maximum-entropy projectionで確率和、条件付き平均、条件付き分散を保存する。
全supportで不可能ならparameterを変更せずcandidateをfail-closeする。

## 実行結果

Kaggle private CPU kernel
`kentookumura/exp439-continuous-kinematic-joint-hmm-train` version 1
（id_no `129058811`）を実行した。約`33.181 sec`、最初のwell
`060ab2b8`のrow 0で次を検出して停止した。

```text
source_rate=0
destination_rate=0
mean_shift=-0.11000000000021828 ft
target_variance=0.015006249999999999 ft^2
minimum_nonnegative_lattice_variance=0.026400000000028373 ft^2
```

固定0.35 ft latticeでは平均を挟む`-0.35 / 0.0 ft`への非負2点分布が最小分散を
与える。target varianceはその最小値より`0.011393750000028374 ft^2`小さく、
5/7/9-cellのどのsupportでも指定momentを保存できない。

これは事前登録したtechnical contractの失敗であり、実装やpackageの不具合ではない。
HMM message、prediction、truth-late evaluationへ入る前に停止したため、HMM well-run
完了数、prediction artifact、truth/role/fold/episode readはすべて0。

## Notebook

- 正規train:
  `exp439_continuous_kinematic_joint_transition_exact_hmm_train.ipynb`
- 正規inference guard:
  `exp439_continuous_kinematic_joint_transition_exact_hmm_inference.ipynb`
- Jupytext実装元:
  `exp439_continuous_kinematic_joint_transition_exact_hmm_compact_selfcontained_train.py`

正規notebookは2026-07-29の実行承認によりcompact self-contained版を採用した。

## 検証方針

fixed32の32 wellsをtruth-freeでfreezeし、全prediction/SHA生成後だけ
role/fold/truth/episodeを結合する計画だった。ただしjoint-edge table自体の
nonnegative moment feasibilityを最初のtechnical gateとし、1 edgeでも不可能なら
predictionやmechanism評価へ進まずcandidate全体を閉じる。

## 検証

- contract test: 12件PASS
- `py_compile`: PASS
- Ruff: PASS
- Jupytext round-trip test: PASS
- strict `validate-exp`: PASS
- Kaggle package: private / CPU / internet off / strict / `--no-src`

## 利用判断

`completed_stage0_technical_failed_closed_moment_projection_infeasible`。
exp209の固定grid/noiseを維持したexact moment-preserving lattice projection branchは
閉じる。support、moment、noise、grid、rate、emission、prior、gateのsame-exp救済、
再実行、Stage 1、inference、submissionは行わない。

## 所見

continuous-kinematic meanは妥当でも、粗い固定latticeと小さい固定varianceを同時に
維持するとexact moment projectionが存在しない。exp439の失敗はjoint-state仮説より
手前の表現可能性で決まったため、mechanism scoreやCVのnegative evidenceとしては
扱わない。
