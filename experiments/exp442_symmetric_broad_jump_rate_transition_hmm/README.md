# exp442_symmetric_broad_jump_rate_transition_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU Stage 0完了、`stage0_fail_closed`
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Stage 1 / inference / submission: 不可

## 仮説

exp209の安定な局所rate遷移を99%残し、1%だけ全rate binへ届く対称Gaussian branchを
混ぜれば、通常区間を壊さずに急なrate変化へ追従するescape pathを持てる。

```text
P_candidate = 0.99 * P_exp209 + 0.01 * P_broad
P_broad sigma = 0.02 rate units
```

方向triggerは使わず、HMM尤度に両方向を比較させた。exp441はnegative contextであり、
exp442の実行前提やpositive evidenceには使っていない。

## 固定変更

- 親exp209のlocal kernelを99%維持する。
- `jump_weight=0.01`、`broad_sigma_rate=0.02`の対称branchを1つだけ加える。
- position、emission、prior、state、readoutは変更しない。
- 保存exp209 controlを使い、親controlは再実行しない。

## 検証方針

Stage 0はfixed32のmechanism preflightとし、technical / mechanism全gateを
AND判定する。predictionとtarget-free diagnosticのSHAをfreezeしてから
truth / role / fold / episodeを読み、1件でもFAILならStage 1へ進まない。

## Stage 0結果

Kaggle version 1（id_no `129101211`）で1候補×fixed32を完走した。
technicalは14/15、mechanismは4/9 PASSだった。

- broad branch responsibility: `0.00976695`
- non-adjacent edge mass: `0.00684557`、PASS
- 将来rate方向一致: `0.529732 < 0.60`、FAIL
- persistent SSE削減: `-4.4385% < +5%`、FAIL
- persistent改善well / fold: `9/16` / `2/5`、FAIL
- matched-control pooled / p95 delta:
  `-0.155414 / +0.069364 ft`、PASS
- full-773 runtime投影:
  `222,019.844 > 30,600 sec`、FAIL

branchは実際に使われ、control safetyも保ったが、必要な方向性とpersistent改善がなく、
full実行も非現実的だった。Stage 0はmechanism-onlyであり、CV/LB evidenceではない。

## 所見

対称broad supportを低確率で足す方法は、support不足そのものは緩和できるが、
正しい方向と持続区間を選ぶ情報を与えない。今回は「branchが使われなかった」のではなく、
使われてもpersistent lagの改善へ結び付かなかったnegative resultである。

## 結論

固定契約どおり`stage0_fail_closed`とする。`0.01` / `0.02`のgrid、
asymmetric/GR trigger、duration、reset/re-anchor、emission/grid/gate救済、
rerun、Stage 1、inference、submissionは行わない。

## 次

完了済みexp442をbacklogから削除する。exp444などtrend/persistenceを明示する
既存案は独立仮説として扱い、exp442のFAILをpositive evidenceや実行条件には使わない。
