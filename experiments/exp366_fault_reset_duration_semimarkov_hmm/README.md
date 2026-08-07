# exp366_fault_reset_duration_semimarkov_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical PASS / scientific FAIL・fail-closed完了
- CV / LB / Submit: なし
- 作成日: 2026-07-23
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

GR change pointとexp209 emission surpriseが同時に出た箇所だけ、boundedな位置reset branchを
固定期間保持すれば、通常区間を変えずmode slip後の再捕捉ができる。

## 変更点

- trigger行だけbase + 4 jump × 3 durationのexplicit-duration branchを作る。
- jumpは`±6.3/±12.6 ft`、durationは`128/256/512`行、commit marginは5 log units。
- branch内のrate/position dynamicsとGR emissionはexp209のまま。
- atlas、rate predictor、全row branch、oracle commitは使わない。

## Stage 0実装

- raw GR changeのvisible-prefix robust zと、保存済みexp209 pathのGaussian
  emission surpriseがともにprefix q99.5以上の行だけをtrigger候補にする。
- trigger後512行は重複spawnを禁止する。negative controlはaccepted trigger scoreを
  well内で512行circular shiftする。
- 全候補を固定512行窓で比較する。reset branchは設定durationだけbase pathへjumpを加え、
  exp209 position grid内へclipし、duration終了後はbaseへ戻すため、13候補の累積GR log
  emissionとtruth RMSEを同じ窓で比較できる。
- branch順は`base → |jump| → sign(-,+) → duration(128,256,512)`に固定する。
- trigger ledger、13 branchのpath content SHA / GR score / evidence rank、reporting foldを
  suffix truth前にgzip保存してSHA再読込する。
- freeze後だけ512行base bad-event、oracle branch、within-10 coverage、MRR、hidden-like方向を読む。

## 検証方針

- Stage 0は保存済みexp209 pathからtriggerと12 branchをtruthなしで凍結する。
- truth join後にtrigger AUC、circular control、alternative coverage、GR-selected MRRを評価する。
- Stage 1は全gateと別承認後だけ1 variant / 773 semi-Markov HMM runs。

## Notebook

- `exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py`:
  Jupytext percent形式のStage 0編集元
- `exp366_fault_reset_duration_semimarkov_hmm_train.ipynb`:
  実行承認によりcompact候補を採用した正規Stage 0 Notebook
- `exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.ipynb`:
  正規版と同一SHAのcompact候補
- `exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.ipynb`:
  Stage 1未実装を強制するfail-closed候補
- 正規inference Notebook placeholderは未上書き。Stage 1未実装のため実行対象外。

## 結果

Kaggle private CPU version 2（id_no `128543224`）を`666.798832 sec`で完了した。
3,389,090 eligible rowsから40 eventsだけ発火し、trigger AUC `0.500004`、
circular差`0.000003`、trigger率`0.0000118`、MRR gain `-0.123356`、
passing fold `0/5`だった。selected branchはbaseより`1.005307 ft`悪化した。
alternative within-10 coverage `0.90`だけはPASSしたが、固定AND gateはscientific FAIL。
Stage 1、inference、submissionは未実装・未実行のまま閉じた。

## 所見

### 良かった点

- reset必要性をrate予測へ結び付けず、triggerとbranch coverageの2条件へ分離した。
- HMMを1 wellも再実行せず、保存済みexp209 pathだけで先行条件をfail-closedに判定できる。

### 悪かった点

- exp289/290/231ではfault trigger、persistent offset、atlas recoveryの根拠が弱かった。
- exp366でもGR change AND exp209 emission surprise triggerはbad event AUCがほぼ0.5で、
  GR evidenceによるbranch選択はbaseより悪化した。

### リスク / 注意

- Stage 0 gateはFAIL。HMMを実装せずbranchを閉じる。
- trigger、jump、duration、marginのgrid searchは禁止する。

## 次

同じfault/reset familyをparameter救済で再開しない。独立した新しい識別根拠がない限り、
Stage 1の773 semi-Markov HMM runs、inference、submissionへ進まない。
