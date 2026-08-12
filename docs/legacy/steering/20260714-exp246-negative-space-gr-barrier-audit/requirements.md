# 要件

## 目的・仮説

GR mismatch heatmapの高不一致ridgeを、もっともらしいpathを生成するpositive evidenceではなく、既知prefixから別corridorへ越境するpathを除外するnegative-space evidenceとして利用できるかを検証する。正しいpathをほぼ切らずにmode jump候補だけを高precisionで除外できることが仮説である。

## 依頼

horizontal well と typewell の GR mismatch heatmap で高不一致の赤い ridge を構成し、既知 prefix 終端に連結した青い corridor から隣接 corridor へ候補 path が越境していないかを監査する。正しい path を新規生成するのではなく、絶対に通れない可能性が高い negative space を候補除外に使えるかを、予測を変更しない train-side diagnostic で先に判定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- raw train horizontal/typewell と保存済み exp072 candidate cache を固定入力にする。
- barrier mask、anchor component、candidate crossing 判定に evaluation-tail true TVT、target、candidate error、oracle scoreを使わない。
- true TVT は barrier 完成後の cut / survival / oracle 評価にだけ使う。
- active variant は `diagnostic_only` 1本。LightGBM config 0、fold training 0、booster 0、親/control再学習なし、GPUなし。
- 初回では HMM/PF/Beam の transition、candidate値、selector、submissionを変更しない。
- GR missing、局所flat、state gridの大半が高不一致になるunsupported rowはhard wallにしない。

## 受け入れ基準

- Jupytext percent形式のself-contained train notebookで、入力確認、barrier生成、corridor/component生成、candidate監査、metrics保存がセル単位で追える。
- 全773 wellsを対象に、true path cut率、anchor component survival、candidate endpoint/crossing率、bad-candidate prune precision、good-candidate false-prune率、candidate-union oracle before/after、distance bucket、hidden-like、by-well worstを保存する。
- strict exclusionで候補が0件になる行数と、未変更baselineへ戻す診断上のfallback行数を分けて記録する。
- `config.yaml` にthreshold、smooth window、ridgeのMD持続長、TVT厚み、grid step/cap、candidate定義、go/no-go guardを置く。
- inference notebookは診断不採用contractを明示し、submissionを生成しない。
- deterministic submission anchor として扱わない。input/cache SHA、config SHA、出力CSVのcontent SHA、kernel versionを記録対象にする。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次のアクション

実装・静的検証済みのKaggle CPU train packageをfull 773-well auditとして実行し、5つのsafety guardを判定する。guard通過前はhard edge-cut、raw-test inference、submissionへ進めない。
