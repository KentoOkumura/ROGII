# 要件

## 依頼

- `robust_rts_l1_convergence_calibration_audit`を`exp306`として新規作成し、steeringと実験ディレクトリで設計を確定する。
- exp304で反復上限内に収束しなかったrobust RTS / L1 trendについて、truthやseparability scoreを使わずsolver収束だけを調整可能か判定する。
- 今回は設計とdesign-only scaffoldまでとし、実装、Kaggle package/push/run、科学評価、inference、submissionは行わない。

## 仮説

exp304の未収束がsolver objectiveの破綻ではなく反復予算と停止条件の不整合なら、事前固定した最小変更だけで全series technical coverageを回復できる。

## 制約

- Routeは`pf_beam`とする。
- exp304のcommon input、missing policy、coordinate normalization、RTS objective/Q/R/Student-t df、L1 objective/lambda/rho/toleranceを固定する。
- RTSはA=`max_irls 32, tol 1e-6`、AがStage 0で1 seriesでも未収束の場合だけB=`max_irls 32, tol 1e-4`を許可する。
- L1は`max_admm 2000`だけを変更し、`rho=1`, abs/rel tolerance `1e-4`、lambdaを固定する。
- Stage 0の64 wellsは`SHA256("exp306-stage0-v1|" + well_id)`昇順の先頭64に固定する。各wellのhorizontal/typewell、計128 seriesを監査する。
- RTSとL1は独立sub-branchとして判定する。一方のFAILで他方を無効化しない。
- truth、error、formation、exp304 MRR/top3、hidden-like roleをsolver設定選択に使わない。
- adaptive rho、lambda/rho/tolerance/iteration grid、同じOOFでのscore救済を行わない。
- exp304の`selected_denoiser=swt_db4_l3`、quality gate、artifact、後続exp305の順序を変更しない。
- model / LightGBM config / trained fold / HMM / PF / Beam / boosterはすべて0とする。

## 受け入れ基準

- Stage 0で候補branchの128/128 seriesがconverged、finite、length/order一致、silent fallback 0を満たす。
- Stage 0の先頭8 wellsを同じ設定で再実行し、denoised output、solver status、iteration countのcontent SHAが完全一致する。
- 候補branchごとのfull 773-well runtime外挿が8.5時間以内である。
- Stage 0 PASS branchだけを全773 wells / 1,546 seriesで別々にfull technical auditし、1,546/1,546収束、finite、deterministic identity、silent fallback 0、input/output/status SHA、実測runtime 8.5時間以内を必須とする。
- full auditで1 seriesでもFAILしたbranchはtechnical negativeとして閉じる。他branchの判定は維持する。
- 全件PASSしてもexp306内でMRR/top3、truth-nearest rank、HMM/PF、RMSEを評価しない。科学評価は単一設定を固定した将来の別expに限定する。
- gzip生成物を保存する場合はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠にする。

## 次のアクション

Stage 0 version 1は完了し、L1だけがfull-eligibleになった。2026-07-23のユーザー指定によりfull 773-well / 1,546-series technical auditは後続`exp351_exp306_l1_full_convergence_audit`へdesign-onlyで切り出した。exp306はStage 0 evidence anchorとして固定し、RTS救済、科学評価、inference、submissionへは自動進行しない。

## 2026-07-23 後続設計

- L1 full auditの実装先を新規`exp351_exp306_l1_full_convergence_audit`へ変更した。
- exp351のsteering/scaffold/backlog作成だけを承認済みとし、実装とKaggle実行は未承認。
- exp306のcode/config/Stage 0 artifactをfull audit用に変更せず、parent SHA anchorとして維持する。

## 2026-07-22 承認範囲

- compact self-contained train、fail-closed inference、synthetic contract test、正規Notebook採用、実験記録更新を許可する。
- 実データStage 0、Kaggle package/push/run、full 773-well audit、科学評価、inference、submissionは許可範囲に含めない。

## 2026-07-22 Stage 0実行承認と結果

- ユーザーの`実行してください`により、Kaggle CPU Stage 0のpackage/push/runを追加承認した。
- version 1はL1 `128/128` convergenceとexact parityをPASSし、RTS A/Bは`7/128` / `108/128`でFAILした。
- full audit、RTS追加調整、科学評価、inference、submissionはこの追加承認に含めず、未承認のまま維持する。
