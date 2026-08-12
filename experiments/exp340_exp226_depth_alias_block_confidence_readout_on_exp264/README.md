# exp340 exp226 depth-alias block confidence readout on exp264

## 状態

- Route: `ensemble`
- 状態: Kaggle CPU Stage 0 version 1完了、科学gate FAILで閉鎖
- Kernel: `kentookumura/exp340-exp226-alias-readout-on-exp264-train` version 1
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 固定入力: exp280の13-shift likelihood bank

## 仮説

depth aliasを直接修正する前に、truth非依存のlikelihood形状だけから、exp264が悪化しやすい512-row blockを識別できる。

## 検証方針

- 候補・予測・selectorは一切変更しない0-booster readoutとする。
- confidence familyはmargin、entropy、重み付きshift標準偏差、zero rank、top shift絶対値、prior block jump、3-block符号不整合の7個に固定する。
- feature、fold別Q1/Q4境界、内容SHAを凍結後にだけTVT誤差を結合する。
- exp264 block RMSE、10 ft以上の誤差、exp226がexp264を0.25 ft以上上回るblockを評価する。
- 合否条件は [config.yaml](config.yaml) と steering を正とする。

## 所見

全familyのcoverageとtarget-free freezeは正常だったが、必須のrow-weighted
`abs_error>=10 ft` AUC 0.60を満たすfamilyはなかった。最良のprior-block jumpでも
AUC `0.574392`、circular control勝利は3/5 foldsだった。depth aliasの大きさと誤差層には
関係が見える一方、安定した検知器として補正へ昇格できる強さではない。

## 実装境界

compact self-containedのtrain/inferenceと正規Notebookを実装済み。trainはexp280 SHA固定score
から7 family、fold別Q1/Q4、circular controlをtruthなしでfreezeし、その後だけ保存OOFを
結合した。Stage 0は7/7 family FAILで完了し、実行スイッチは再実行防止のため消費済みに
戻した。補正、モデル学習、HMM、推論、提出は行っていない。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp340-exp226-depth-alias-block-confidence-readout-on-exp264/`
- 設定: `config.yaml`
- 結果: `result.md`
- 成果物: `kaggle/output/train_v1/artifacts/`
