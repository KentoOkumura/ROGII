# exp339 missing-gap pseudo-mask uncertainty readout

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0完了、固定gate FAIL、枝を閉鎖
- 優先度: P1
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 後続候補: `exp341_missing_gap_calibrated_soft_variance_exp226_residual_hmm`

## 仮説

既知prefix内の有限GRを、実欠損run長分布に合わせて決定的に隠し、同じ線形補間を再現すれば、欠損補間誤差の分散を外側foldに対して安全に校正できる。

## 検証方針

- HMM、TVT予測、学習器を使わないStage 0 readoutとする。
- gap長は `1-3 / 4-7 / 8-15 / 16-31 / 32-64`、最近傍anchor距離は `1 / 2 / 3-4 / 5-8 / 9-16 / 17-32` に固定する。
- 2次元分散表をsupport 200で `length bin -> outer-train global` に縮約する。
- outer-validのpseudo-gap誤差は表の推定に使わず、真のGRは予測内容SHA固定後にだけ復元する。
- 合否条件は [config.yaml](config.yaml) と steering の requirements/design を正とする。

## 所見

global constant分散比のNLL、校正、gap長相関は通ったが、real placementのcircular control比fold勝利が2/5で固定条件4/5をFAILした。自然欠損への転送根拠が不足するためHMM改善を主張せず、exp341へ表を渡さない。

## 実装境界

compact self-contained trainとfail-closed inferenceを正規Notebookへ採用済み。Kaggle CPU version 1を完了し、trainはouter-train histogram、real/circular pseudo-gap、補間予測freeze、late hidden-GR join、階層分散表、全gate、SHA付き生成物を保存した。HMM、raw-test inference、提出は未実装・未実行のまま閉じた。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp339-missing-gap-pseudomask-uncertainty-readout/`
- 設定: `config.yaml`
- 設計時点の記録: `SESSION_NOTES.md`
