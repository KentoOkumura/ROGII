# exp321_z_only_residual_gr_correction_ladder 結果

## 状態

Kaggle CPU Run AB version 1を完了した。Stage AはPASS、Stage Bは固定bank range / quantization gateをFAILしたため、Stage C、inference、submissionを救済なしで閉じた。

## 実行

- kernel: `kentookumura/exp321-z-only-residual-gr-ladder-train` version 1
- URL: `https://www.kaggle.com/code/kentookumura/exp321-z-only-residual-gr-ladder-train`
- runtime: `611.963350 sec`
- data: 3,783,989 suffix rows / 773 wells / exp226固定5 folds
- compute: 1 diagnostic contract / 5 fold strata / 0 model config / 0 trained fold / 0 booster / 0 HMM / 0 window decoder
- parent/control再実行: 0
- runtime: Kaggle CPU、GPU/TPU/internet off

## Stage A: residual structure

Stage Aは全technical checkと科学gateをPASSした。

| H512メトリック | 値 |
| --- | ---: |
| Z-only direct RMSE | 107.494824 |
| exp226 `tvt_geop` direct RMSE | 10.077950 |
| Z-only affine-quotient RMSE | 0.609237 |
| exp226 affine-quotient RMSE | 0.669091 |
| Z-only / exp226 affine-quotient比 | 0.910543 |
| affine SSE説明率 | 0.999968 |
| `±4 ft` oracle RMSE gain | 3.205124 ft |
| block mean residual絶対値 | 90.628894 ft |
| block slope絶対値 | 0.033651 ft/row |
| lag-1 residual correlation | 0.999827 |

H256/H512のrelative-shape gateはともに5/5 foldsで成立した。Z-onlyは局所block内ではほぼaffineな残差構造を持つ一方、direct pathの絶対offset/driftは非常に大きい。

## Stage B: GR shift separability

Stage Bは順位信号のgateを通過したが、固定shift bankのcoverage gateをFAILした。

| メトリック | real | shuffle | lift |
| --- | ---: | ---: | ---: |
| top1 | 0.332991 | 0.109028 | 0.223963 |
| top3 | 0.587903 | 0.277257 | 0.310646 |
| MRR | 0.503399 | 0.278098 | 0.225301 |
| sign | 0.685887 | 0.486709 | 0.199178 |

- top1/top3/MRR/signは全5 foldsでreal > shuffle。
- pooled値はexp280保存値を4指標すべてstrictに上回った。
- 1000+、hidden-like spatial、hidden-like typewell-purgedでも4指標すべて正方向。
- 固定`[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft` bankのrange coverageは`0.494029`でFAIL。
- quantization coverageは`0.604212`、最大量子化誤差は`384.734576 ft`でFAIL。
- 7,787 blocksのtruth-nearest slotは`-80 ft`が25.65%、`+80 ft`が36.32%を占め、bank端への集中も確認した。

GR likelihoodはbank内の方向・順位を識別できるが、Z-only residual scaleの約半分を固定bankが覆えない。事前登録したStage Bは全check必須なので総合FAILとする。

## 判定

- Stage A: PASS
- Stage B: FAIL
- Stage C: `blocked_by_stage_ab_gate`
- next action: `close_stage_c_branch_without_rescue`

shift bank拡張、sigma変更、threshold緩和、decoder変更を同一truth上で試さない。予約案4 exact-HMMと案5 sparse candidateも開始条件不成立のため閉じる。Stage C、inference、submissionは実装・実行していない。

## 再現性

- target-free contract SHA: `8ab762faff47b5d402064c88cfd0b6cdbc271c92b18e4ff1de1342b6c37186c4`
- decision manifest SHA: `00eb8e81d8b82ff5ae5774b5cced0f5655b14c935f7c8be5a9c53eb6827b309c`
- target-free path: 3,783,989 rows、logical SHA `a0154e64808e2085c32eb0a99e9826a4caadd5ad940e579e0dea33edcb0f38af`
- target-free shift scores: 101,231 rows、logical SHA `af4194f087d459a588077278116ecfca614288694402f2ec0d91e2c40b075e57`
- Stage A blocks: 52,909 rows、logical SHA `68864f754b6989587c993183554db9e8016d3410807952233d0bb6347c26fd96`
- Stage B readout: 7,787 rows、logical SHA `80786980bc8c37bc25101bf80392f024e0257c938f7adeef09d878573fa0fbcc`
- truthはtarget-free path/score freezeとcontract SHA確定後にだけ結合した。

## 実装検証

専用synthetic/contract test 10件、Notebook/scaffold共通testを含む対象21件、Jupytext round-trip、構文、ruff F821、strict experiment validation、template validationはすべてPASSした。全体suiteは433 PASS / 1 SKIPで、既存exp296の完了後configと実行前期待がずれている2件だけFAILした。exp321関連は全PASS。

## 解釈と次

Stage Aは「Z-only残差が局所的に低次元」という仮説を支持したが、平均offset scaleが大きく、Stage Bの固定小shift bankを直接の補正候補にする前提は成立しなかった。GR順位信号だけを根拠にbankや補正幅を事後拡張するとexp280/281/298と同じ救済探索になるため行わない。同系の新規backlogは追加せず、独立した既存優先実験へ戻る。
