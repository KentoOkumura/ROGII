# exp365_bounded_gr_registration_offset_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical PASS / scientific FAIL、fail-close完了
- CV / LB / Submit: なし
- 作成日: 2026-07-23
- 実装・実行日: 2026-07-25
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Kaggle kernel:
  `kentookumura/exp365-bounded-gr-registration-offset-hmm-train` version 2

## 仮説

物理TVTとGR/typewellの登録位置に小さな持続ずれがあるなら、出力位置とは別のbounded
offset stateを周辺化することで、物理pathを直接ずらさずにGR emissionの整合を改善できる。

## 固定契約

- `delta=[-6,-3,0,3,6] ft`。出力はphysical positionのまま、emission参照だけ
  `p+delta`とする。
- Stage 0はvisible prefix内の128行history / 64行held-out / stride 64 rolling
  originだけを使う。
- delta=0比のheld-out GR NLL、circular control、非縮退、境界mass、
  隣接window符号安定性、runtime / RSSをAND評価する。
- unknown suffixの`TVT`、物理prediction、保存済みexp209 controlは読まない。
- exact HMMの5倍状態空間は親exp209 runtimeと16-well shapeから固定投影する。

## 検証方針

technical gateを全て満たしたうえで、real NLL gain、circular control差、4/5 folds、
posterior非縮退、境界mass、隣接window符号安定性、runtime、RSSの固定条件をAND評価する。
1項目でも不合格ならStage 1へ進めない。

## 実行入口

- `exp365_bounded_gr_registration_offset_hmm_train.ipynb`:
  実行済みStage 0 compact self-contained Notebook。再実行フラグはfalse。
- `exp365_bounded_gr_registration_offset_hmm_inference.ipynb`:
  Stage 1未実装を明示して停止するfail-closed Notebook。

## 結果

Kaggle private CPU version 2はtechnical gateを全てPASSしたが、scientific gateはFAILした。

- real NLL gain: `5.430399%`（PASS）
- circular NLL gain: `15.311425%`
- real-minus-circular: `-9.881025%`（FAIL）
- passing folds: `0 / 5`（FAIL）
- adjacent-window sign agreement: `0.580771`（FAIL）
- projected runtime: `56,429.34 sec`（FAIL）
- projected peak RSS: `7.358320 GB`（PASS）

詳細とSHAは`result.md`、実行履歴は`SESSION_NOTES.md`を参照する。

## 所見

`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`。offset/transition/grid、sigma、runtime係数、
gate、controlを救済変更しない。Stage 1 exact HMM、inference、submissionは実施しない。

## 次

本branchは完了として閉じる。同じscientific contractの閾値調整ではなく、
別の独立仮説を次候補として選ぶ。
