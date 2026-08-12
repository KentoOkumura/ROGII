# 設計

## アプローチ

exp307が作るGR補間系列とfinite-MAD scaleを完全固定し、raw GR missing maskから作る1次元confidenceだけをGaussian log emissionへ掛ける。値の補完法とscaleを変えないため、「補間値を弱く数える」効果だけを測る。

## 実験範囲

- 対象: `exp308_imputed_gr_confidence_downweight`
- Route: `pf_beam`
- 親: `exp307_finite_only_robust_sigma_gr`
- 変更: evaluation行ごとのGR emission weightだけ。
- 固定: GR補間値、`σ_GR`、typewell curve、Gaussian residual/clip、HMM grid/transition/start/posterior mean。
- 除外: 欠損値の再生成、長欠損hard skip、sigma/transition適応、PF/LikPF/Beam、inference、submission。

## 固定confidence契約

- raw horizontal GRがfiniteなら`d=0,w=1`。
- raw GRがmissingなら、同一well全系列で最近傍finite raw GRまでの行距離を`d>=1`とする。
- `w(d)=max(0.25, 2**(-d/8))`。
- finite raw GRがwell全体にないfail-safeは全missing行`w=0.25`。
- exp209 Gaussian log emission `ell_t(v)`を`w_t * ell_t(v)`へ置換する。
- missing maskと距離はraw GRからprediction/truth参照前にfreezeする。

既知prefix自然欠損runはexp294でq90 3行だったため、距離3ではweight約0.771を残す。距離8で0.5、長いgap中央でも0.25を下限にし、exp269の全state-neutral化とは分離する。

## 依存gate

- exp307 primaryがoverall/fold/long-tail/by-well/fixed-blend/SHA gateを全PASSしている。
- exp307 primary prediction、scale audit、scientific contractのexpected SHAをconfig実装時に固定する。
- dependency FAILまたはSHA不一致ならHMMを開始せずstatusをblockedのままにする。

## 検証方法

1. raw well identityとexp307 frozen inputsをpreflightする。
2. raw mask、distance、weightをtruthなしでfreezeし、observed/missing/gap bucket分布とSHAを保存する。
3. parentと同じ1 HMM variantをweightだけ変更して773 wells生成する。
4. prediction freeze後にtruth/5 folds/hidden-like/saved LikPFをlate joinする。
5. overallに加えraw observed/missing、gap 1--3/4--15/16+、distance、long-tail、by-wellを評価する。

## 実行量

- active variants: 1
- HMM well-runs: 773
- model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle CPU、internet off、runtime 8.5時間上限

## 生成物契約

- dependency/input/scientific contract JSON
- raw missing mask/distance/weight audit CSV.gz
- weighted prediction/posterior diagnostics CSV.gz
- overall/fold/gap/distance/hidden-like/by-well metrics
- fixed LikPF blend readoutとgate summary

## 再現性設計

- RNGなし。distance/weightはwell内row orderで決定的に計算する。
- raw mask、distance、weight、parent input、prediction、metricsのcontent SHAを記録する。
- exp307 controlは再生成せずexpected SHAで固定する。
- Kaggle kernel version/bootstrap source/config SHAを実行時に記録する。model/submission SHAは非該当。

## リスク

- exp269ではraw missing emissionのblanket neutralityがexact HMMを+1.410212 ft悪化させた。本案も補間evidenceを弱めるため同方向のリスクがある。
- half-life 8/floor 0.25は単一事前設定であり最適性は未証明。FAIL後のgridを禁止する。
- raw missingがGR eventと非ランダムに対応するとdownweight自体が情報を捨てる。observed/missing/gap readoutで分離する。
- dependency exp307がFAILなら、本案単独のために旧zero-fill sigmaへ戻して実行しない。

## 次のアクション

self-contained train/inference Notebookとcontract testの実装まで完了した。exp307 promotion gate FAILにより必須dependencyは成立しなかったため、SHA/metricsを実行用に固定せず、Kaggle package/push/run、inference、submissionなしで閉鎖する。
