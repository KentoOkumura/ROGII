# Idea portfolio schema

`idea_portfolio.json`はUTF-8 JSONとし、次を含める。

```json
{
  "schema_version": "2",
  "task_summary": "string",
  "evidence_cutoff": "string",
  "allowed_sources": ["string"],
  "assumptions": ["string"],
  "closure_ledger": [
    {
      "evidence_id": "string",
      "closed_instantiation": "string",
      "not_closed": "string",
      "reason": "string"
    }
  ],
  "idea_cards": [
    {
      "id": "I01",
      "title": "string",
      "mechanism_family": "representation | information | candidate_generation | fusion_uncertainty | data_generation | validation | compute_enabler | other",
      "origin_pass": "task_first | evidence_inversion | cross_pollination",
      "roles": ["string"],
      "information_sources": ["string"],
      "hypothesis": "string",
      "evidence_ids": ["string"],
      "changed_mechanism": "string",
      "input_target_decode": "string",
      "deployment_error_simulated": "string",
      "preserved_invariants": ["string"],
      "nearest_prior_attempt": "string",
      "exact_difference": "string",
      "counterevidence": "string",
      "cheap_test": "string",
      "full_test": "string",
      "kill_criterion": "string",
      "reopen_criterion": "string",
      "coverage_test": "string",
      "selectability_test": "string",
      "hidden_inference_contract": "string",
      "compute_estimate": "string",
      "is_parameter_only": false,
      "novelty_level": "incremental | role_change | representation_change",
      "confidence": "A | B | C"
    }
  ],
  "portfolio": [
    {
      "idea_id": "I01",
      "slot": "safe | exploration | orthogonal | compute_enabler",
      "why": "string"
    }
  ],
  "rejected": [
    {
      "idea": "string",
      "failed_gate": "string",
      "reopen_condition": "string"
    }
  ]
}
```

Rules:

- `idea_cards`: 10–14件、重複IDなし。
- `portfolio`: 5件、すべて`idea_cards`内のID。
- `idea_cards`全体で4 family以上。
- `portfolio`は`representation`、`information`、`data_generation`、`candidate_generation | fusion_uncertainty`、`validation | compute_enabler`を各1件以上含む。
- `portfolio`には`origin_pass=task_first`と`novelty_level=representation_change`を各1件以上含む。
- `is_parameter_only=true`: 全体で最大2件、portfolioで最大1件。
- `roles`、`information_sources`、`evidence_ids`、`preserved_invariants`: 空配列不可。
- 文字列フィールドは空にしない。非該当は理由付きで`not applicable: ...`とする。
- `coverage_test`と`selectability_test`は常に別々に記載する。
