# exp306_robust_rts_l1_convergence_calibration_audit 結果

## 状態

Kaggle CPU Stage 0 version 1完了。L1はfull audit適格、RTS A/Bは不適格。full auditは未実装・未承認。

## 仮説

exp304のRTS/L1 failureが反復予算と停止許容差の不整合なら、truth-freeな固定設定で全series収束を回復できる。

## 設定

- 親: `exp304_gr_denoiser_emission_separability_readout`
- 検証: fixed 64-well Stage 0、8-well deterministic parity、eligible branch別full 773-well technical audit
- メトリック: solver technical coverage、finite/order/fallback、content SHA、runtime
- シード: RNGなし。固定salt SHA256 sample。
- 実行量: Stage 0 core 384 + L1 parity 16 series-runs、model/HMM/PF/Beam/booster 0
- Kaggle kernel: `kentookumura/exp306-rts-l1-convergence-calibration-audit-train` version 1、id_no `128231380`、CPU / internet off

## 結果

- L1 `max_admm=2000, rho=1, tol=1e-4`は`128/128` convergence/technical PASS。iterationsはmin/mean/max=`264/656.758/1993`、実測`25.161 sec`、773-well外挿`303.896 sec`、8-well x 2 seriesのoutput/status/iteration SHA parityも完全一致した。
- RTS A `max_irls=32, tol=1e-6`は`7/128` PASS、`121/128` FAIL。iterations mean/max=`31.844/32`、実測`999.044 sec`、外挿`12,066.577 sec`。FAILはhorizontal 59、typewell 62。
- 条件付きRTS B `max_irls=32, tol=1e-4`は`108/128` PASS、`20/128` FAIL。iterations mean/max=`23.219/32`、実測`695.615 sec`、外挿`8,401.723 sec`。FAILはhorizontal 7、typewell 13。
- 全branchでfinite input/output、length/order/status identity、silent fallback 0、runtime上限をPASSしたが、RTSは全series convergenceを満たさずfull不適格。唯一のfull eligible branchはL1。
- Notebook summary到達は`1774.422 sec`（約29分34秒）。scientific score、CV、LB、prediction、submissionは生成していない。

## 再現性

- deterministic anchor: 予測・提出anchorではない。L1 Stage 0のsolver technical parityはexact PASS。
- seed policy: RNGなし、固定SHA256 sample、固定single worker/thread。
- kernel version: Kaggle CPU version 1、id_no `128231380`。
- raw well identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
- Stage 0 input raw/decompressed SHA: `308d09d90dc13ba29db6b0a5e7c5930833fa1d3833ef63fe7376b4ef074126ec` / `ee4b26f34177d3367e4c3e84900727bc115e497bd076268c1e62a1f5276ce50b`。
- Stage 0 output raw/decompressed SHA: `649a98cdd5591bdac35582e69ca5c347c7b66809376e0c2a261f441fa1d0284b` / `8a6f7e38bcea659f5ab7d0fd0cf37475c6d5d84bfed1826b5febfcb7ecf67df7`。
- Solver status raw/decompressed SHA: `fcb2fe6a658cec66be314353475c475b7e19c7335145cda89c49df2850147592` / `bef261e1b905dd59e05f91c9966d481ede3fb063566db0f5fca0fe829fb665e9`。
- Sample manifest raw SHA: `67508cba8dab2de14e13d77edec6b8faadab8fdacd44334ca2ce029b6ddcf691`。取得した実ファイルを再計算し、gate記録値との一致を確認した。
- model/prediction/submission SHA: 非該当。

## 解釈

L1のexp304 failureは反復予算不足で説明可能で、固定2000 ADMMによりStage 0全件収束を回復した。RTSは許容差緩和で`7→108/128`へ改善したが、残る20 seriesは32 iterationsで停止しており、事前固定gate上は不合格。追加iterations/tolerance gridによる救済はexp306の範囲外とする。technical-only結果なのでdenoiser品質は判断せず、exp304のselected SWT、exp305、案4の状態を変更しない。

## 次

L1全件technical auditはdesign-only後続`exp351_exp306_l1_full_convergence_audit`へ切り出した。実装・Kaggle実行は別承認待ち。RTS再調整を検討するなら、まず残る20 FAILのtarget-free failure profileを別steeringで固定し、exp306内では再試行しない。科学評価、inference、submissionへ自動進行しない。
