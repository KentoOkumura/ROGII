# exp377_formation_relative_k16_slope_identifiability_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU v2完了・Stage 0 PASS / Stage 1 scientific FAIL・branch終了
- CV: `38.776238 ft`（primary path RMSE、direct `16.100131 ft`）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp226

## 仮説

TVTのK16勾配を直接補間するより、`S=TVT+Z`を各train-only地層面からの相対勾配へ分解し、対象位置の地層面勾配を戻す方が、坑井間の構造差を分離できる。

## 変更点

- outer-train donorの`d(S-F_f)/dMD`を6地層別に生成する。
- outer-trainだけの`FormationPlaneKNN(k=10)`から対象区間の`dF_f/dMD`を求める。
- exp226のK16、方位projection、XY最近傍50、bandwidth 500 ft、ridge 1を固定する。
- primaryを6系列すべてのmedianへ事前固定する。
- 予測器、HMM、PF、ML、current-test候補、submissionは作らない。

## 実装

- 正の実装候補はJupytext percent形式の
  `*_compact_selfcontained_train.py`と、それから変換した同名`.ipynb`。
- compact self-contained train候補を正規`*_train.ipynb`へ採用した。
- inference候補はdiagnostic-only契約を確認後にfail closedする。
- 専用test 8件、Jupytext round-trip、構文、Ruff、strict実験validationをPASSした。

## 検証方針

- Fold: exp226 outer 5-fold
- Group: well
- Stage 0: 3,783,989行、773坑井、12,368区間、valid truth/formation read 0、coverage/support/fallback。
- Stage 1: rate relative gain、累積path gain、fold、scope、by-well tail。
- Leakage: valid側`TVT`と6地層列はfreeze前にparseせず、Stage 0 PASS後だけtruthをlate joinする。

## 結果

| メトリック | 値 |
| --- | --- |
| 実装検証 | PASS |
| 専用test | 8 passed |
| Kaggle kernel | v2 / id_no `128452991` / `COMPLETE` |
| Stage 0 | PASS（effective donors p05 `2.5947`はreport-only warning） |
| Stage 1 / Kaggle CV | FAIL: direct `16.100131` → primary `38.776238 ft` |
| path改善fold | `0 / 5` |
| by-well | 164改善 / 609悪化、p95 `+49.434562 ft` |
| Public LB | - |

## 所見

### 良かった点

- design-frozenのouter-fold、K16、XY kernel、primary、gateをconfigとコードの両方で固定した。
- valid側のtruth/formation read guardとStage 0 fail-closed境界を専用testで確認した。
- relative-rate復元、K16積分、Stage 1固定AND gateを合成データで検証した。

### Kaggle CPU v2で判明した点

- 3,783,989 rows / 773 wells / 12,368 segments / 5 folds、valid truth/formation read 0、
  coverage 1.0、surface fallback 0.0はPASSした。
- direct controlとrelative系列に共通するeffective donors p05 `2.5947`は数値を保存し、
  report-only warningとした。他のStage 0 blocking checkはすべてPASSした。
- truth-late Stage 1ではsegment rate RMSEが`0.012301 → 0.038454`、
  累積path RMSEが`16.100131 → 38.776238 ft`へ悪化した。rate/pathとも改善foldは`0/5`。
- 609/773坑井が悪化し、well差分p95は`+49.434562 ft`、worstは
  `a247e7cf`の`+408.044686 ft`だった。
- 6個の個別formation系列もpath RMSE `39.022--40.356 ft`で全てdirectより悪い。
  median6 primaryの集約方法だけが原因ではない。

## リスク / 注意

- `clean273`は行集合でなくML特徴allowlistなので、本readoutではpooled契約の別名とする。
- 対象側Formation列の参照はleak。role read guardを必須とする。
- v2でもK / bandwidth / surface / scopeは変更しなかった。
- effective-donor checkは数値を保存したままreport-onlyとし、ほかのintegrity
  checkを満たしたためStage 1へ進んだ。

## 次

固定したformation-relative K16方式は科学的に不支持として終了する。
surface / kernel / scopeのposthoc救済、exp378 / exp379 / exp380 / exp382、
inference、submissionへ進まない。再訪する場合は同じ方式の微調整ではなく、
独立した物理モデルを事前設計する。
