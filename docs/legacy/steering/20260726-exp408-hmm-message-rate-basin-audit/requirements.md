# 要件

## 依頼

exp209 exact HMM の長い offset が、GR matching の別 mode 選択だけで説明できるのか、
それとも transition / hidden rate / backward smoothing / sum-product readout のどこで
形成されるのかを、保存されていない内部 message を再計算して特定する。
重い処理はローカルではなく Kaggle CPU で実行する。

## 制約

- Route: `pf_beam`
- 親の current HMM だけを再計算する。position width、exact-mean、momentum、
  emission、mode retention などの treatment は実行しない。
- 対象は事前固定した persistent-offset 450 wells。HMM は
  `1 variant x 450 wells = 450 well-runs`。
- LightGBM config / trained fold / booster / model / PF / Beam / GPU はすべて 0。
- raw horizontal の未知 suffix `TVT` は decoder 入力として読まない。
  truth と episode 境界は、その well の prediction と message SHA を freeze した後だけ
  diagnostic mask と score に使う。
- exp209 の grid、41 rate states、rate transition、5-cell position transition、
  Gaussian GR emission、GR interpolation、known-prefix scale、initial position/rate prior、
  `mom=0.998` を変えない。
- full alpha / beta tensorは永続化しない。episode rowの集計値、episode summary、
  well manifestだけを保存する。
- Kaggle bootstrap、input、prediction、message、row ledger の SHA を記録する。

## 受け入れ基準

- 保存済み exp270 posterior mean と `1e-5 ft` 以内で一致し、450/450 wellsを処理する。
- predictive（emission前）、filtered（emission後）、smoothed（beta後）の
  position/rate mass が有限かつ各行で正規化される。
- truth / posterior-mean / Viterbi basin mass、GR log-odds寄与、beta寄与、
  rate平均・分散・近傍mass・edge mass、position-rate covariance、
  current/exact-mean 1-step moment差、logsum/max gapを保存する。
- persistent 638 episodes / 807,710 rowsと完全一致し、escape/re-capture と
  排他的な原因分類を出力する。
- truth/error/episode-detail の pre-freeze read が 0 である。
- Kaggle CPU runtimeが9時間以内、peak RSSが25 GB以内で完了する。
- deterministic anchor とは扱わないが、logical/decompressed content SHA と
  Kaggle kernel versionを記録する。

