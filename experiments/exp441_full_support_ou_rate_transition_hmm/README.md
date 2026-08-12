# exp441_full_support_ou_rate_transition_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle Stage 0 v1完走、`stage0_fail_closed`
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209の追従遅れの一部は、rate確率質量が1行で隣接binにしか移れない
tri-diagonal近似による。親の`momentum=0.998`と`sig_r=0.002`から定まる
連続OU過程を全41 rate binへ直接積分すれば、物理パラメータを増やさずに
人工的な有限伝播速度を除去できる。

## 変更点

- rate kernelだけを隣接3状態Euler近似から全support exact OUへ置換する。
- OU平均・分散は親の`momentum`と`sig_r`から一意に決める。
- TVT遷移、GR emission、prior、grid、境界mass切捨て、readoutは固定する。
- trigger、jump mixture、acceleration、reset、re-anchorは入れない。

## 検証方針

- Stage 0: exp411 fixed32の1候補×32 HMM well-runs。保存exp209 controlを使い、
  control HMMは再実行しない。
- Technical gate後にtruthをjoinし、exp408のzero-directed under-response、
  forward-cause、persistent episode、matched-control safetyをAND判定する。
- 全PASSと別承認がある場合だけ773 wellsのStage 1へ進む。
- model / LightGBM / booster / PF / Beam / GPUはすべて0。

## 実行入口

以下を実装済みである。

- `*_compact_selfcontained_train.py` / `.ipynb`
- `*_compact_selfcontained_inference.py` / `.ipynb`
- `experiments/exp441_full_support_ou_rate_transition_hmm/tests/test_exp441_full_support_ou_rate_transition_hmm.py`

正規`*_train.ipynb`はcompact候補から採用し、Kaggle private CPU Stage 0
version 1を実行済みである。正規inference Notebookは変更していない。
Stage 1、inference、submissionはfail-closedを維持する。

## 実装内容

- `kappa=-log(momentum)`のexact OU平均・分散を使う。
- 各rate centerの有限Voronoi区間へGaussian CDF差を積分する。
- support外tailを捨て、端のtransition rowを再正規化しない。
- OU kernelはfloat64、forward/backward messageは親と同じfloat32とする。
- exp209 position kernel、Gaussian GR emission、prior、readoutを固定する。
- analytic mass/moment、position parity、小規模dense brute-forceを事前監査する。
- 全32 wellのkernel/prediction/diagnostic SHA freeze後だけrole/fold、truth、
  persistent episode、cause、exp408 parent row ledgerを読む。

## リスク

- 遅れが離散化ではなくGRのrate識別力不足だけで生じるなら改善しない。
- 遠距離rate massがwrong basinへ流れ、controlやwell tailを悪化させ得る。
- full-support化で計算量が増えるため、Stage 0でruntime/RSSを先に判定する。

## 所見

- 32 wells / 156,088 rowsを1,582.080秒、peak RSS 1.123249 GBで完走した。
- technical gateはruntime projectionだけFAILし、16 / 17 PASS。
- mechanism gateはmatched-control安全性2件だけPASSし、2 / 7 PASS。
- under-response share削減は2.297 points、persistent episode SSEは
  1.674%悪化、改善は8 / 16 wells・1 / 5 foldsだった。
- `stage0_fail_closed`として閉じ、Stage 1やsame-OOF救済へ進まない。

## 次のアクション

exp441はterminal closeとする。追加の原因確認が必要な場合だけ、保存済み
rate diagnosticを入力にする0-HMM / 0-predictionのtruth-late attribution
readoutを別実験・別承認で検討する。
