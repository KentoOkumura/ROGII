# exp279_exp226_geop_centered_exact_hmm_redecode

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train version 1完了、promotion guard FAIL、branch closed
- CV: 10.035987（`geop_hmm`）
- Public LB: 未実行
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-18
- 科学的親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 入力親: exp226 group-safe OOF、exp209 exact HMM、exp072 likelihood-PF

## 仮説

exact HMM / PFがGRの別modeへ逸脱した後にoffsetを維持する一因は、transitionが前状態を
伝搬する一方、各行で絶対位置へ戻す基準がないことにある。exp226のGR未補正geometry
`tvt_geop`を毎行の弱いGaussian unaryにすれば、形状を維持しつつmode slip後の復帰を促せる。

## 変更点

- exp209 exact HMMのgrid、41 rate states、transition、GR emission、calibration、missing-GRを固定。
- exp226 `tvt_geop`中心の`Gaussian(sigma=20 ft, lambda=0.50)`を1点追加。
- exp226 `tvt_pred` / `gr_delta`をdecoderから除外し、GR二重利用を防止。
- 保存済みexp226 / exp209 / exp072からexp263 fixed formulaを比較用に再構成。
- 誤差10 ft超が128行以上続くepisodeの256/512行以内復帰率をtruth-only事後診断。

## 検証方針

- Fold: exp226保存済み5 folds
- Group: `well`
- Score rows: train unknown suffix 3,783,989 rows / 773 wells
- Leakage check: exp226 fold-by-well 1 fold、OOF decompressed SHA、decoderへのtrue TVT非流入、candidate freeze後truth結合
- Promotion baseline: exp263 fixed formula OOF 8.238331 / Public LB 7.800
- 実行量: HMM 1 variant / 773 well-runs / LightGBM config 0 / trained fold 0 / booster 0

## 実行入口

- 学習 notebook: `exp279_exp226_geop_centered_exact_hmm_redecode_train.ipynb`
- 推論 notebook: `exp279_exp226_geop_centered_exact_hmm_redecode_inference.ipynb`（guard通過まで無効）
- 初回実行: Kaggle private CPU、GPU/internet off
- local full notebook実行は禁止。合成単体テストと静的検証だけをローカルで行う。

## 結果

| メトリック | 値 |
| --- | --- |
| `tvt_geop` fixed-grid coverage事前監査 | 100%（3,783,989 / 3,783,989 rows、773 / 773 wells） |
| exp209 exact HMM | 11.938287 |
| exp226 prediction | 9.427110 |
| exp263 fixed baseline | 8.238332 |
| `geop_hmm` | 10.035987（exp263比 `+1.797655 ft`） |
| 改善fold | 0 / 5 |
| persistent-offset 512行以内復帰 | 11.85%（exp263 9.07%） |
| Public LB | 未実行 |

## リスク / 注意

- exp226 geometry自体が誤るwellでは、unaryが正しいGR補正を弱める可能性がある。
- exp221相当のfull exact HMMはKaggle CPUで約5時間を見込む。
- sigma / lambda / grid / process-noiseの事後探索は行わない。
- 全promotion guard通過前にinference・submissionへ進まない。

## 所見

geometry unaryはexact HMM単体を約1.90 ft改善したため、毎行の絶対位置情報はHMMのmode slipを
部分的に抑えた。ただしexp263より全5 folds、near / 1000+ / hidden-like、worst-wellが悪化した。
またpersistent-offsetの512行以内復帰率は少し上がったがepisode数は551から802へ増えた。
固定のgeometry anchorだけでは誤ったmodeを安全に解除できず、exp263を比較基準として維持する。

## 次

sigma / lambda / grid / process-noiseの救済探索、PF追加、raw-test inference、submissionへ進まず、
本branchをnegative resultとして閉じ、同一unaryのparameter救済や直接置換は追加しない。
ただし2026-07-19のユーザー指示により、この直接枝とは分離して、修正版exp264の既存候補を
維持したまま`geop_hmm`を疎なadd-only selector候補として評価するbacklogを
`backlog/KAGGLE_DIRECTION.md`へ追加した。
