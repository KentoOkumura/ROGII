# 要件

## 依頼

`learned_pf_likelihood_weight_or_feature_followup` を実装する。exp111 の learned PF observation likelihood を direct replacement ではなく、PF weight / verifier gate / ML add-only feature の材料として評価できるようにする。

## 制約

- Route: `pf_beam`
- 親実験: `exp111_learned_pf_observation_likelihood_probe`
- 入力は exp111 OOF likelihood long cache と exp099 v2 wide cache に固定する。
- 新しい selector / regressor は学習しない。
- learned top1、PF weight、verifier gate をそのまま提出候補にしない。
- ML feature artifact には true TVT、abs error、within labels を含めない。
- 再現性: `docs/06_reproducibility.md` に従い、gzip 生成物は decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/` に config、補助コード、train/inference notebook、記録ファイルがある。
- train notebook から posthoc audit script を実行できる。
- script は metrics、OOF predictions、by-well、bucket、selection distribution、ML feature cache、feature schema、summary JSON を出力する。
- `likpf_mean_single`、learned top1、multiobs top1、PF weight alpha、conservative verifier gate、oracle を同一 surface で比較する。
- `make validate-exp EXP=exp112_learned_pf_likelihood_weight_or_feature_followup` が通る。
- deterministic anchor とは扱わず、raw-test parity が必要であることを `config.yaml`、`README.md`、`result.md` に明記する。
