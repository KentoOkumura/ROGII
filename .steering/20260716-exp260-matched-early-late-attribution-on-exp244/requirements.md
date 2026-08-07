# 要件

## 依頼

`exp244_bidirectional_prediction_start_pseudotail_augmentation`で同時投入したearly / late
prediction-start pseudo rowsを、同一cache・sampling・weight・fold・LightGBM設定のまま方向別に分離し、
mixed augmentationのhidden-like改善とworst-well崩壊を原因分解する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親/controlを再学習しない。保存済みexp218 OOFとexp244 mixed OOFを比較対象にする。
- 変更する変数はpseudo directionだけとし、early-onlyは`-1000/-250`、late-onlyは`+250/+1000`を使う。
- official rows 3,783,989、official weight 1.0、pseudo weight 0.5、各view最大250 rows、380-feature schema、exp218互換5-foldを固定する。
- active variants 2、LightGBM configs 3、folds 5、合計30 boosters。親/control再学習0。
- ユーザーは2026-07-16、上記30-booster実験へ進むことを明示承認した。
- outer-valid source well由来pseudo rowsは各foldのtrainから除外し、validationはofficial-start rowsだけにする。
- current-test inferenceとsubmissionは行わない。

## 受け入れ基準

- early-only / late-onlyの各OOFを、raw exp218とexp244 mixedに対してoverall、6 distance buckets、
  1000+、hidden-like 2面、5 folds、by-well、worst-wellで比較できる。
- late-onlyの独立補償guardはoverall改善、1000+非悪化、hidden-like 2面非悪化、worst-well +2 ft以内、
  3 / 5 folds以上改善とする。
- early-only / late-onlyの30 models、feature importance、OOF、metrics、by-well、model manifest、SHAを保存する。
- exp244 mixed OOFのdecompressed SHAとrow/fold identityをhard assertionする。
- notebookはJupytext percent形式から生成し、入力・variant・学習・評価・生成物がセルで追える。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
