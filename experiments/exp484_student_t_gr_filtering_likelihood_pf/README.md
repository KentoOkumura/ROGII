# exp484_student_t_gr_filtering_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: Stage 1 scientific gate FAIL・terminal close
- CV: `10.897096923`
- LB / Submit: なし
- 親: exp417、実装参照・保存control: exp404

## 仮説

exp374のfixed `df=4` Student-tをparticle filtering尤度へ入れると、大きなGR残差で
正しいPF modeが失われにくくなる。

## 変更点

exp404のGaussian particle log emissionだけをStudent-tへ置換する。
PF dynamics、scale、resampling、500 particles、128 seeds、T=5は固定する。

## 検証方針

well IDだけのstable SHA256順で固定した32 wells / 165,010 suffix rowsの
technical preflight後、全gate PASSと2026-07-30の別承認を受けて全773 wellsを評価する。
保存exp404 scale-5 x1.0をcontrolとして使い、control PFは再実行しない。

Stage 0の実行量は1 scientific variant、32 PF well-runs、4,096 seed-well
trajectories、2,048,000 particle starts。LightGBM、HMM、Beam、GPUは0。

## Stage 1結果

- candidate / 保存exp404 control:
  `10.897096923 / 10.914521913 ft`
- 改善量: `+0.017424990 ft`（必要`+0.05 ft`）
- 改善fold: `2 / 5`（必要`4 / 5`）
- raw GR observed: `-0.068900357 ft`
- raw GR missing: `+0.205368304 ft`
- hidden-like typewell-purged: `-0.130146256 ft`
- by-well delta p95 / worst:
  `+1.455066656 / +16.664889733 ft`
- fixed exp209 HMM/PF 50:50: `+0.017106090 ft`改善

18/18 technical checksとfixed blend guardはPASSしたが、pooled gain、
fold一貫性、raw observed、typewell-purged、well-tailを満たさなかった。
事前登録どおりStudent-t/PF/blend/selector救済を行わずbranchを閉じる。

## 所見

Student-t化はmissing-GR系scopeでは改善したが、実観測GRの識別力を弱め、
少数wellの大きなwrong-basin悪化を防げなかった。fixed32の参考改善は
全773-well CVへ一般化しなかった。

## 成果物

- compact self-contained Jupytext train source / Notebook
- fail-closed inference guard
- target-free stable-hash fixed32 manifest
- Student-t formula、exp404 input parity、stable seed、finite weight、
  truth-late、SHA、notebook contractの専用test

Stage 0/1のKaggle生成物と、Stage 1のfold/by-well/gate/runtime/SHA記録を
保存した。prediction本体の大容量archiveは取得していない。

## 次

このbranchは`terminal_close_without_student_t_or_pf_rescue`として閉じる。
inference、submissionは実行しない。
