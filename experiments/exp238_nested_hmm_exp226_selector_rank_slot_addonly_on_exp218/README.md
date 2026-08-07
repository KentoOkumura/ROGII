# exp238 nested HMM/exp226 selector rank-slot add-only on exp218

exp237 selector の rank・margin・continuity・confidence surfaceを、strict nested stackingでexp218へadd-onlyするtrain-side実験です。selector pathの直接置換や提出は行いません。

## 状態

Kaggle selector train v4、hidden-safe final inference v3、code submission ref `54662073`まで完了しました。Public LBは`7.775`で、従来のML route anchor exp218 `7.843`を0.068改善しました。final nested OOF `lgb_mean`は7.936690です。selector worst-well guardは+37.680897で不通過、historical exp218とはouter fold割当が異なるcaveatは残ります。先行提出ref `54647064`のpublic-test行artifact依存によるhidden rerun errorは、current test再生成へ修正して解消しました。

## 仮説

outer-fold safeなselector confidenceで危険な候補選択を識別できれば、exp218へadd-onlyしてもnear rowとworst-wellを壊さずlong-tailを改善できます。

## 検証方針

outer 5 folds、inner 4 foldsのwell GroupKFoldでselectorをnested生成します。CPUの`*_selector_train.ipynb`でglobal、near、1000+、worst-well guardとfold別score生成物を作り、別GPU notebookの`*_train.ipynb`でfold固有rank-slot特徴をexp218へadd-onlyします。guard不通過後のfinal trainは、ユーザーの明示承認による開始条件overrideとして実行しました。

## 所見

35 boostersは完走しました。数値上はexp218 8.475794から-0.539104ですが、fold非一致のため特徴効果の厳密な差とは扱いません。nested OOFの同一fold target leakageはありません。次に原因分離するなら、追加GPUなしのwell-risk selector auditを先行し、同一fold base-only 15 boostersは別途承認がある場合だけ実行します。

## Raw-test copcf parity修正

従来のselector inferenceでは、trainで使った184 contextのうちexp109/exp114由来
`copcf_*` 41列とexp226診断4列をcurrent testで作らず、45列を全行NaNとしていました。
trainから41列を削除したexp245はparity修正ではなくablationとして扱います。

`*_rawtest_copcf_parity.ipynb`は、exp238の保存済み20 selectorと184列schemaを維持し、
test wellを固定train typewell clusterへ独立に割り当て、full-train typewell/spatial prior、
cluster confidence 41列、exp226診断4列をcurrent testから再生成します。test-test近傍、
selector/final再学習、中央値/0 fallback、submissionはありません。Kaggle v1で14,151行・
184 contextを生成し、missing列0、nonfinite値0、41 `copcf_*`全列finite、exp226診断4列finite、
outer別5 score面各14,151行を確認しました。selector 20 modelは再学習せず読み込んでいます。

`*_inference_copcf_parity.ipynb`は、このgeneratorを既存のhidden-safe final inferenceへ
接続した別名の採用候補です。保存済み20 selectorからouter別35 rank-slot特徴を作り、同じ
outerの保存済み3 final LightGBMへ渡して全15本を平均します。exp218 380 + selector 35 =
415特徴、selector/final/control学習0、test-test近傍なし、public-test行artifactなしです。
`submission.csv`は生成しますがcompetition submitは行いません。静的検証済み・Kaggle未実行です。

## 主要生成物

- selector train v4: outer 5 × inner 4の保存済み20 selector modelとSHA付きmanifest。
- selector inference v3: public test監査用のouter fold別5 predicted-error score面。hidden提出では使用しない。
- final inference v2: 14,151行のprediction、415特徴schema、`submission.csv`、inference summary。
- hidden-safe inference: current test上でbase replay、HMM、exp226 K16、multiobs、exp145 learned likelihood、GRWR、outer別selector scoreを再生成し、保存済み20 selectorと15 LightGBMを適用する。
- successful submission: ref `54662073`、scriptVersionId `334897917`、Public LB `7.775`、submission SHA `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e`。
- submission SHA: `dc0eb2e8f4581d0e91a8a6748f13cae17742e86539cbc234fa3a42fad6ec1f9d`。
