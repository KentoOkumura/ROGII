# 設計

## 1. 目的

exp445は改善候補ではなく、exp209の離散exact HMMをTVT座標からU座標へ
純粋に再パラメータ化したとき、全結果が一致することを確認するtechnical
parity auditである。

位置を`p_t=TVT_t`、既知の坑井座標を`Z_t`、rateを
`r_t=d(TVT+Z)/dMD`とすると、

```text
p_t - p_(t-1) = r_t * delta_MD - delta_Z
u_t = p_t + Z_t
u_t - u_(t-1) = r_t * delta_MD
```

は連続空間で同値である。本実験では離散格子でも同値性を保つ。

## 2. 離散座標contract

親の全row共通TVT index格子を`P_j`、stepを`h=0.35 ft`とする。
candidateではcell数、index、phaseを変えず、各rowの状態値だけを

```text
U_t,j = P_j + Z_t
TVT_t,j = U_t,j - Z_t = P_j
```

と定義する。これは既知Zによるrow-wise translationであり、状態supportを
data-dependentに作り直すrow-adaptive regridではない。

source index `j`からdestination index `k`へのcandidate physical U edgeは、

```text
U_t,k - U_(t-1),j
  = (P_k - P_j) + delta_Z
```

である。desired U displacementを`r_t*delta_MD`とするとresidualは、

```text
(P_k-P_j) + delta_Z - r_t*delta_MD
  = (P_k-P_j) - (r_t*delta_MD-delta_Z)
```

となる。したがってcandidateのindex-space position kernel meanは親と同じ
`r_t*delta_MD-delta_Z`でなければならない。`r_t*delta_MD`だけをindex shiftへ
渡すと、moving U gridの`delta_Z`を二重に扱えず、exp438型の別モデルになる。

## 3. emission・prior・readout

- emission:
  `GR_typewell(U_t,j-Z_t)=GR_typewell(P_j)`。親のemission配列と一致する。
- initial position prior:
  last-known rowでも`U_last,j=P_j+Z_last`なので、中心index、sigma、massは親と
  一致する。
- rate prior / transition:
  exp209の41 rate states、rate span、`sig_r`、momentum、arrival-rate方式を
  そのまま使う。
- posterior:
  同じstate index `(j, rate_index)`上のalpha、beta、filtered / smoothed
  posteriorは親と一致する。
- numerical readout:
  exp209と同じ周辺化順でposterior / log-likelihoodを保持する。座標期待値は
  parent/candidate双方のposition posteriorを明示正規化して比較し、
  exp209互換raw matrix-product readoutはreport-only ledgerへ併記する。
- readout:

```text
E[U_t] - Z_t
  = sum_j posterior_t,j * (P_j+Z_t) - Z_t
  = sum_j posterior_t,j * P_j
  = E[TVT_t]
```

- U readout:
  `E[U_t]=E_parent[TVT_t]+Z_t`も同時に監査する。

## 4. exp438との差

| 項目 | exp438 | exp445 |
| --- | --- | --- |
| U格子 | `P_j+Z_last`を全rowで固定 | `P_j+Z_t`へrowごとに既知量だけ平行移動 |
| TVTとして見た格子 | `P_j+Z_last-Z_t` | 常に`P_j` |
| index transition mean | `r_t*delta_MD` | `r_t*delta_MD-delta_Z` |
| 科学差分 | 固定離散格子を置く座標が変わる | なし。state labelだけを変える |
| 期待結果 | 親と異なり得る | 親と一致しなければ実装不備 |

exp438のnegative resultを再分類しない。exp445 PASSはfixed absolute-U格子を
支持せず、座標変換そのものが情報を追加しないことだけを確認する。

## 5. 実験範囲

- 対象実験: `exp445_tvt_to_u_coordinate_parity_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 比較参照: `exp438_u_state_fixed_lattice_exact_hmm`
- 変更する変数: position stateの数値表現を`P_j`から`P_j+Z_t`へ変える。
- 固定する変数: state index、grid、rate、全transition確率、emission、
  prior、forward-backward順、readout後のTVT、入力well順。
- 評価対象: technical parityだけ。CV / RMSE / LBは対象外。

## 6. 実装方針

2026-07-29に実装承認を得て、Jupytext percent形式のcompact
self-contained train候補、inference禁止guard、専用testを実装した。
2026-07-30の実行承認後に正規notebookへ採用し、Kaggle Stage 0 v2を完了した。

candidateはU値を明示的に構築し、次の2経路を分ける。

1. exp209互換parent TVT path。
2. row-shifted U座標からphysical edge、emission、readoutを組み立てるcandidate
   path。

両経路が最初から同じposteriorを共有するだけの形式にはせず、少なくとも
synthetic dense referenceとreal fixed32で独立に組み立てた配列を比較する。
ただし高速実装の最終kernelが代数簡約後に親と同じindex kernelを使うことは、
座標contractの帰結として許可する。

## 7. technical parity gate

### synthetic

- variable-Zとconstant-Zの両方を含む。
- `U_t,j-Z_t=P_j` max abs `<=1e-12 ft`。
- candidate physical-edge residualとparent index-edge residual max abs
  `<=1e-12 ft`。
- emission / initial prior / rate kernel / position kernel max abs
  `<=1e-12`。
- tiny joint HMMを全path列挙したbrute-force referenceと比較し、
  log-likelihood / posterior max abs `<=1e-6`。

### real fixed32

- SHA固定fixed32 manifestを使用する。
- candidate 32 + paired parent 32 = 64 HMM well-runs。
- suffix truth、fold、role、episode、error readは0。
- finite coverage `1.0`。
- per-row emission max abs `<=1e-12`。
- transition probability max abs `<=1e-12`。
- log-likelihood max abs `<=1e-6`。
- smoothed position / rate posterior max abs `<=1e-8`。
- TVT posterior mean / std max abs `<=1e-6 ft`。
- candidate `E[U]-Z`とparent `E[TVT]` max abs `<=1e-6 ft`。
- prediction / posterior / transition ledgerのreadback SHAを照合する。

全条件PASSで`coordinate_parity_verified`として完了する。1条件でもFAILなら
`technical_parity_failed`とし、CV、inference、submissionへ進まず、
原因をimplementation / numerical contractとして調査する。threshold、grid、
noise、rate、emissionを変更して一致させない。

## 8. 実行量と承認境界

- scientific candidate: 1。
- manifest wells: 32。
- candidate HMM well-runs: 32。
- paired parent HMM well-runs: 32。
- total HMM well-runs: 64。
- reporting folds: 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- Stage 1 full OOF、inference、submission: なし。

2026-07-30のユーザー依頼により、正規Notebook採用、Kaggle package /
push、fixed32 Stage 0の1回実行まで承認済みである。inference、submission、
独立rerunは承認範囲に含めない。

## 9. 再現性設計

- `docs/06_reproducibility.md`確認済み。
- seed policy: RNGなし。well / row / position / rate / edge / message順を固定する。
- stochastic処理: なし。
- PF / Beam / likelihood-PF / seed bagging: なし。
- 並列処理: 初回parityはCPU 1 worker / Numba 1 threadを固定する。
- runtime: Kaggle private CPU、internet disabledを予定。GPUは使わない。
- 入力: manifest、horizontal、typewell、親contractのfile/content SHAと
  row / well / schemaを記録する。
- 生成物: coordinate ledger、transition/emission arrays、posterior、
  predictionのlogical/content SHAを記録する。
- gzip: decompressed content SHAを主証拠にする。
- model / submission SHA: modelもsubmissionも作らないため非該当。
- Kaggle package: prepareが承認された場合だけ、source、loose config、
  bootstrap展開後config、metadataのbyte/SHA parityを確認する。
- deterministic anchor: 初回runだけでは呼ばず、独立rerunでinput /
  posterior / prediction SHAが一致した後にparity anchorとして扱う。

## 10. リスク

- リークリスク: truthを不要とする。decoder入力から未知suffix TVTを除外し、
  truth/fold/role read countを0として監査する。
- 誤設計リスク: `U_t,j=P_j+Z_t`を「row-adaptive regrid」と誤解して固定Uへ
  戻すとexp438を再実行してしまう。
- 二重補正リスク: moving U gridに加えてindex meanを`r*delta_MD`へ変えると
  `delta_Z`の扱いが崩れ、parityでなくなる。
- 偽陽性リスク: parentとcandidateが同じ中間配列を無条件共有すると独立確認に
  ならない。synthetic physical-edge契約と別組立て配列を必須とする。
- ランタイムリスク: paired 64 HMM runsはparent保存predictionのload-onlyより
  重いが、posterior / likelihood / kernel parityには直接parent rerunが必要。
- 解釈リスク: PASSは性能向上でもexp438救済でもなく、同値変換の確認だけである。
