# exp286_geop_hmm_sparse_addonly_candidate_on_exp264

## 状態

- ルート: `ensemble`
- 状態: Stage D version 1完了・parent12比pooled改善・安定性guard FAIL
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 追加候補: `exp279_exp226_geop_centered_exact_hmm_redecode/geop_hmm`
- inference / submission: disabled / disabled

## 仮説

Stage 0では`geop_hmm`を加えたfull 13候補oracleがrow / block512 / whole-wellを5/5 foldsで改善した。
固定top-25% gateはwhole-well gainを27.71%しか保持できなかったが、これはsparse gateの失敗であって、
候補を正式にselectorへ加えたときの学習効果を測っていない。そこでgateを使わず、全OOF行で
`geop_hmm`を13番目候補としてexp264と同じselectorへ追加し、12候補版と比較する。

## 13番目候補の情報契約

- candidate ID: `geop_hmm`。string artifactを保持し、model入力はone-hot。ordinal indexは禁止。
- kind / family: `primitive` / `geop_centered_exact_hmm`。
- availability: exp263とexp279のkeyが一致する3,783,989行すべてで1。gateなし。
- generic proxy: candidate TVT、finite、last-anchor差、local shape 32/128/512、bank disagreement、
  2 legal-domain統計を他候補と同じ共有実装で生成する。
- native confidence: `geop_hmm_std -> sigma_tvt`、`geop_hmm_loglik -> source_loglik`、
  well評価行数で割った`loglik_per_row`、`candidate_finite_source`、`confidence_valid`。
- `true_tvt_readout_only`、実誤差、oracle、catalog RMSEはfeature/confidenceへ読み込まない。

機械可読な正は`candidate_contract.yaml`。`geop_hmm`をprimary domain 11→12、fixed comparison domain
7→8へ追加し、既存12候補の順序・式・fallbackは変更しない。

## Stage B実行範囲

- Stage A feature/confidence audit: 0 booster
- active variant: 1
- LightGBM config: 1（exp264 corrected Stage B v5と同一）
- objectives: 2（`pred_abs_error` / `p_within10`）
- outer folds: 5
- 合計: 10 CPU boosters
- parent/control再学習: 0
- HMM/PF再生成: 0 well-runs
- Stage C: 0 / Stage D: 0 / GPU: 0
- inference / submission: 0 / 0

## 比較

再学習しないexp264 corrected Stage B v5を12候補baselineとしてSHA固定する。primary判定は13候補版の
hard selector OOF RMSEがparent 8.587004を下回り、3/5 folds以上改善し、`geop_hmm`が実際に選択され、
13候補score guardもPASSすること。hard path、fold path、shared-12 candidate scoreは直接比較できる。
全候補pooled scoreはlong行数が12→13へ変わるため補助指標として扱う。

## 検証方針

well-grouped outer 5-foldで`pred_abs_error`と`p_within10`を学習し、保存済みparent12のhard/fold
pathとshared-12 candidate scoreへ比較する。候補ID feature、availability、native confidence coverage、
model count、OOF/model manifest SHAも監査する。

## 所見

Stage 0は`geop_hmm`の候補多様性を支持したが、固定sparse gateだけを否定した。Stage Bはgateを使わず、
selectorがその多様性を一部回収し、hard OOF RMSEをparent12比`-0.109265 ft`改善した。ただし
fixed fallback 8.238332には`+0.239408 ft`負けているため、hard selector inferenceには採用しない。

## 実行入口

- train: `exp286_geop_hmm_sparse_addonly_candidate_on_exp264_train.ipynb`
- inference: `exp286_geop_hmm_sparse_addonly_candidate_on_exp264_inference.ipynb`（disabled）
- Stage B/C kernel: `kentookumura/exp286-geop-hmm-sparse-addonly-exp264-train`
- Stage D kernel: `kentookumura/exp286-geop-hmm-sparse-addonly-exp264-tvt-train`

## Stage 0履歴

Kaggle CPU version 1（id_no `127856113`）は0 boosterで完了。full 13候補oracleは3粒度・5/5 foldsで
改善したが、固定gateのwhole-well gain保持率`27.710961% < 50%`でsparse gate guardがFAILした。
この結果を受けて固定gate分岐だけを閉じ、ユーザーの明示指示によりfull-all-well Stage Bへ進む。

## Stage B結果

- Kaggle private CPU version 4 / id_no `127856113` / runtime 1,518.253秒 / 10 models。
- hard primary OOF RMSE: parent12 `8.587004` → new13 `8.477740`、delta `-0.109265 ft`。
- fold: 0/1/2改善、3/4悪化で3/5 folds改善。事前selector-addition guardはPASS。
- `geop_hmm`選択: `pred_abs_error` 737,876行（19.50%）、`p_within10` 206,557行（5.46%）。
- candidate ID featureは採用89列schemaに存在。availability、validity、`sigma_tvt`、
  `source_loglik`、`loglik_per_row`、finite flagは全fold coverage 1.0。
- score guardはMAE/logloss/Brierすべてpooled + 5/5 folds PASS。shared-12 scoreも3指標平均で改善。
- fixed fallback RMSE `8.238332`よりhard selectorは`+0.239408 ft`悪く、hard readout guardはFAIL。
- このStage B単体ではparent/control再学習、HMM/PF再生成、Stage C/D、GPU、inference、submissionは0。

## Stage C結果

- 40 CPU selector models、25 compact partitions、18,919,945 compact rowsを生成した。
- nested hard RMSEはparent12 `8.652532`からfull13 `8.448682`へ`-0.203850 ft`改善した。
- 4/5 folds改善し、score guardとleakage auditはともにPASSした。
- compact featureは77列で、`geop_hmm`の両objective scoreとtop1 flagを含む。

## Stage D結果

- private T4 version 1 / id_no `127886849`で15/15 boostersを完走した。
- clean 273 + full13 compact 77 = 350列のadd-only RMSEは、parent12 `8.460811`から
  full13 `8.403784`へ`-0.057027 ft`改善した。
- near / mid / 1000+とhidden-like 2面はすべて改善した。
- 一方、fold改善は2/5、373 wells改善・400 wells悪化、worst wellは`+5.862833 ft`だった。
- pooled改善はPASSしたが、3/5 folds条件とworst-well `+0.25 ft`条件で総合guardはFAILした。

## 次

13番目候補の追加はStage B/C/Dのpooled RMSEをすべて改善したが、Stage Dのfold/well安定性が不足した。
exp286の実験は完了とし、inference/submissionへ進めない。必要なら保存済みparent12/full13 OOFだけを使う
target-free tail-risk attributionを別承認の0-booster readoutとして実施する。
