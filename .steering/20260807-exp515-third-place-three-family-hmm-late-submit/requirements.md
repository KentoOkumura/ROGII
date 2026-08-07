# 要件

## 依頼と手法契約

- 依頼原文: 「3位のHMMを再現する実験行ってください。late submitであることがわかるようにしてください。」
- 期待する成果: 3位writeupの3-family HMMを公開情報だけから再構成し、fold-safe OOFとhidden-test code inferenceを実行する。技術検証後、コンペ終了後の再現監査であることを明記した固定版を1回だけlate submitする。
- 一次資料: Kaggle discussion `733319`、`3rd Place Solution`（tereka、2026-08-06）。公開Notebook/source codeは2026-08-07のKaggle CLI検索では見つからなかった。
- input: horizontal wellの`MD, Z, GR, TVT_input`、対応typewellの`TVT, GR`、train horizontal wellの`TVT, GR`からfold-safeに作るsame-typewell sibling reference。
- target / objective: 学習lossなし。train OOFでは未知suffixの真の`TVT`を生成完了後に結合してrow-level RMSEを測る。
- output: 各unknown rowのTVT posterior meanを3 familyの固定`0.50 / 0.20 / 0.30`で混合し、formation-rate projectionを適用した連続path。
- loss: なし。GR emissionはStudent-t(df=1、Cauchy) log likelihood。
- decode: compact joint hidden state `(TVT offset, formation rate, GR bias, reference family)`上のexact forward-backward posterior mean。最後にprefix rate quantile、absolute rate 0.25、integrated correction 10 ftでprojectionする。
- context unit: whole-well。last 128 known rowsをHMM sequenceへprependし、初期rateはlast 256 known rowsからrobust推定する。
- 実装区分: `proxy`。元コードが非公開で、絶対TVTの状態の刻み方、遷移値、self-referenceの重みを推定し、Local-DTWを3つの伸縮率を選ぶ処理で近似する。
- 省略する機構: なし。ただしLocal-DTWは非公開実装そのものではなく、3つの小さなTVT stretch reference stateとして再構成する。
- 検証できない主張: 原チームのsource parity、原チームの非公開state discretization、CV split assignment、Local-DTW実装、CV 5.9703の厳密再現。
- proxyの場合のユーザー承認: 「推測したパラメータと簡略化した状態空間でHMMを実装した。元コードの忠実な再現ではない」と説明後、ユーザーから「実行してください」と明示された。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- CPU-only、RNGなし、model / LightGBM / booster / PF / Beamは0。
- parent/controlは保存済みmetricsだけを参照し、再学習・再生成しない。
- raw test、well数、row数、IDをハードコードせず、runtimeのsample submissionをschemaと順序の正とする。
- Public LBを見たweight、state lattice、reference、projectionの変更は禁止する。
- submission message、kernel title、`config.yaml`、`SESSION_NOTES.md`、`result.md`に`LATE SUBMIT`または`post-competition reproduction audit`を明記する。

## 受け入れ基準

- 手法契約の `input / target / output / loss / decode / context unit` がコードと一致する。
- 実装区分と実験名が実装した機構を正確に表す。
- `proxy` の場合は、省略点、検証不能な主張、ユーザー承認が記録されている。
- TODO
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
- 3 family、hidden-state 4要素、prefix 128、rate 256、reference bin 0.25/0.0625、Student-t df=1、fixed weights、projectionのコード契約testが通る。
- OOFはvalidation fold wellsをsibling reference sourceから完全に除外し、prediction content SHAをfreezeしてからtruthをscoreする。
- inferenceは現在のcompetition train/testを動的列挙し、sample submissionのIDへ1対1整列し、欠損・重複・余分ID・非finiteを0にする。
- 技術gateを通った同一fixed versionだけを1回late submitする。OOF/LB後の救済gridは行わない。
