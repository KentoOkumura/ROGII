# 要件

## 依頼

- `finite_only_robust_sigma_gr`を`exp307`として新規作成し、steeringと実験ディレクトリで設計を確定する。
- exact HMMの既知prefix `σ_GR`推定で行っている欠損GRの0補完を廃止し、有限なhorizontal/typewell GR pairだけを使うscaleへ変更する。
- 初回依頼はdesign-onlyとし、2026-07-21の後続依頼「exp307を実装してください」で実装を承認した。
- 実装承認はNotebook source/test/正規Notebook/記録更新までとし、Notebook実行、Kaggle package/push/run、inference、submissionは引き続き行わない。

## 仮説

0補完によりwell別`σ_GR`が欠損率へ強く依存している。有限pairだけのscaleへ直し、外れ値に頑健なMADを使えば、GR観測尤度のwell別校正とexact-HMM TVT推定が改善する。

## 制約

- Routeは`pf_beam`、科学的親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とする。
- 保存済みexact-HMM control RMSE `11.9382872349`とdecompressed prediction SHA `8e2f4236...7ae5`を参照し、controlを再実行しない。
- diagnostic `finite_std`は0補完除去だけ、primary `finite_mad`は有限残差のMAD scaleとする。
- `finite_mad = 1.4826 * median(abs(e - median(e)))`、有効pair 20未満は30、両候補とも`[10,60]` clipに固定する。
- evaluation GR補間、typewell GR処理、GR中心/affine係数、Gaussian emission、grid、rate状態、遷移、posterior meanをexp209から変更しない。
- `finite_std`は原因分離用であり、posthocにprimaryへ昇格しない。
- LikPFへの移植、補間GR downweight、状態遷移noise適応、有効標本数縮約は別実験とする。
- 2 variants x 773 wells = 1,546 HMM well-runs、LightGBM/model/fold/booster/PF/Beamは0とする。

## 受け入れ基準

- 773 wells / 3,783,989 evaluation rows、ID/order/target late-join、finite coverage、input/control SHAがPASSする。
- scale auditへcurrent zero-fill、finite std、finite MAD、有効pair数、欠損率、fallback/clip flagを保存する。
- primary `finite_mad`がsaved exact-HMMよりRMSEを0.05 ft以上改善し、4/5 folds改善する。
- 1000+、hidden-like spatial/typewell-purgedを悪化させず、by-well delta p95を0以下、worst regressionを+0.25 ft以下にする。
- saved LikPFとの固定50:50はsaved `10.2696961466`を悪化させない。
- 1 gateでもFAILならprimaryを昇格せず、LikPF port、inference、submission、救済gridへ進まない。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠にする。

## 次のアクション

Kaggle CPU version 2は完走したが、finite MAD primaryはdirect `+3.723054 ft`、fixed LikPF 50:50 `+0.917640 ft`悪化し、promotion gateをFAILした。事前条件どおり救済、inference、submissionを行わず閉鎖する。
