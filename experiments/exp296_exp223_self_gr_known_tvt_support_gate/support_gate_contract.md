# exp296 support gate contract

## 目的

exp223のdescriptor-motif self-GRがknown TVT support外のcandidate stateへ与えるboostだけを除去し、wrong-depth attractionを減らせるかを単一差分で検証する。実装前にmaskの入力、適用順序、不変条件、実行規模、停止条件を固定する。

## 正とする単一差分

exp223 `hmm_selfgr_boost_only_a070_c100`を完全固定し、candidate stateがvisible-prefix known TVT range外のときだけself-GR boostを0にする。

```text
known_tvt_min = min(finite horizontal.TVT_input)
known_tvt_max = max(finite horizontal.TVT_input)
state_supported[j] = known_tvt_min <= grid[j] <= known_tvt_max

exp223_boost[row, j] = clip(exp223_centered_self_gr_surface[row, j], 0, 1)
exp296_boost[row, j] = exp223_boost[row, j] if state_supported[j] else 0.0

logL_exp296 = logL_typewell_exp223 + 0.07 * quality_exp223 * exp296_boost
```

maskはexp223のfull-grid surface生成、centering、scaling、positive clipの後に適用する。support内でsurfaceを再計算・再正規化しない。

## 不変条件

- support boundaryはinclusive、padding 0。
- supportはfinite `TVT_input`だけから作る。
- true `TVT`、予測TVT、error、Type Well TVT range、top-k matched anchor rangeをsupport定義に使わない。
- support外self-GR contributionはexact `0.0`。
- support内boostはmask前exp223 boostとexact parity。
- base Type Well emission、HMM grid/transition、posterior stateはsupport外でも有効。
- final predictionをknown rangeへclipしない。
- finite known TVTがなければself-GRだけをall-neutralにする。
- exp225のstate-known `TVT_input -> GR`曲線を使わない。

## Variantとcount

- variant: `hmm_selfgr_boost_only_a070_c100_known_tvt_support_gate`
- planned variants: 1
- HMM well-runs: 773
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent-control retraining: 0
- GPU / PF / Beam / inference / submission: 0

## 判定

technical contractを全PASSし、exp223比pooled RMSE delta `<= -0.05 ft`、4/5 folds改善、true-TVT-outside scope delta `<= -0.10 ft`、inside scope非悪化、1000+/hidden-like非悪化、by-well p95非悪化、最大well regression`<= +0.25 ft`を全PASSした場合だけscientific supportとする。

1条件でもFAILならsupport/alpha/clip/window/top-k/thresholdを救済せず閉じる。PASSでもexp209 HMM/likPF blend 10.269696以下と別承認なしにinferenceへ進めない。

## 次

Kaggle CPU version 3はtechnical 12/12 PASS、performance 2/10 PASSで完了した。pooled RMSEはsaved exp223 `11.349942946`から`12.159749140`へ`+0.809806194 ft`悪化し、true-TVT-outside scopeは`+2.341424645 ft`、worst-wellは`+39.687791204 ft`だった。

事前登録したFAIL actionを適用し、このbranchを閉じる。support padding、nearest/hole-aware/soft gate、alpha/clip/window/top-k/threshold救済、inference、submissionは実行しない。
