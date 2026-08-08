from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SOURCE = HERE / "exp516_sixth_place_pfa_tw_late_submit_compact_selfcontained_inference.py"
PUBLIC_CONFIG = HERE / "pf_banks_config_v96_public.json"
CONFIG = HERE / "config.yaml"
GENERATOR = ROOT / "scripts" / "prepare_exp516_sixth_pfa_tw.py"

PUBLIC_CONFIG_SHA256 = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"
CHECKPOINT_SHA256 = {
    "stageA_enccapaug_f0.pt": "9ce3763b14a68ae5f05e78467ae4faa5b696ab714c270fd129a6e7d468cdd007",
    "stageA_enccapaug_f1.pt": "d6b1dc0956e47b778bb2089c644d71c9006fdf8948a9cc962c6a40d27dccdfec",
    "stageA_enccapaug_f2.pt": "2f312ce8de05feab9d8480a7971d91eb4e8289cce7e654f895f1b63d95cf5208",
    "stageA_enccapaug_f3.pt": "8bf70f128c12a86e6f65177912d7ee90b5d24a7a67caf91f40df992bdd2c599f",
    "stageA_enccapaug_f4.pt": "b438d113b8fc07d8e9842cb66d01cfff2971ef4a5d213841574ec2680b2ab170",
}


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def top_level_literals(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return values


def test_public_pfa_v96_contract_is_sha_pinned() -> None:
    raw = PUBLIC_CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PUBLIC_CONFIG_SHA256
    public = json.loads(raw)
    pfa = public["params"]["pfA"]

    assert public["physics_banks"] == ["pfA"]
    assert public["w_nn_bank"] == {"pfA": 0.01}
    assert public["n_seed"] == 32
    assert public["smooth_mode"] == "full"
    assert pfa["n_particles"] == 600
    assert pfa["use_anchor"] is True
    assert pfa["anchor_mult"] == 20.0
    assert pfa["grid_step"] == 0.2
    assert pfa["smooth_lag"] == 192


def test_raw_config_file_and_normalized_embedded_text_have_separate_hashes() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    literals = top_level_literals(source)
    text_sha = hashlib.sha256(PUBLIC_CONFIG.read_text(encoding="utf-8").encode()).hexdigest()

    assert literals["PUBLIC_CONFIG_SHA256"] == PUBLIC_CONFIG_SHA256
    assert literals["PUBLIC_CONFIG_TEXT_SHA256"] == text_sha
    assert PUBLIC_CONFIG_SHA256 != text_sha
    assert (
        'hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() '
        '!= PUBLIC_CONFIG_TEXT_SHA256'
    ) in source


def test_experiment_contract_is_standalone_pfa_twgr_late_submit() -> None:
    config = read_yaml(CONFIG)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["method_contract"]["fidelity"] == "faithful"
    assert config["method_contract"]["component"] == "pfa_twgr_standalone"
    assert config["pf"]["active_banks"] == ["pfA"]
    assert config["pf"]["active_representations"] == ["tw"]
    assert config["pf"]["n_particles"] == 600
    assert config["pf"]["n_seeds"] == 32
    assert config["pf"]["smooth_mode"] == "full"
    assert config["anchor"]["folds"] * config["anchor"]["seeds_per_fold"] == 15
    assert config["emission"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert config["late_submission"]["one_shot_fixed_version"] is True
    assert config["late_submission"]["allow_lb_retuning"] is False
    assert config["late_submission"]["message"].startswith("LATE SUBMIT | exp516 |")
    assert config["authorization"]["second_submission_approved"] is False


def test_generated_orchestration_runs_only_pfa_twgr() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    literals = top_level_literals(source)
    assert literals["PF_BANK"] == "pfA"
    assert literals["PF_REPRESENTATION"] == "tw"
    assert literals["PF_GENERATION_SEED"] == 4423098
    assert literals["PF_N_PARTICLES"] == 600
    assert literals["PF_N_SEEDS"] == 32
    assert literals["PF_SMOOTH_MODE"] == "full"
    assert literals["EXPECTED_CHECKPOINT_SHA256"] == CHECKPOINT_SHA256

    run_section = source.split("# ## 8. Run pfA × twGR only", maxsplit=1)[1].split(
        "# ## 9. Sample-ID alignment", maxsplit=1
    )[0]
    assert run_section.count("P = bank_param(PF_BANK)") == 1
    assert run_section.count("outs = run_smoother_ext(") == 1
    assert "for bank in" not in run_section
    assert "for representation in" not in run_section
    assert 'PF_BANK = "pfA"' in source
    assert 'PF_REPRESENTATION = "tw"' in source


def test_state_likelihood_and_full_ancestry_decode_are_preserved() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    required_fragments = [
        "pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()",
        "v = pos - zi",
        "anc_hist = torch.empty((T, B, N)",
        "a = torch.gather(anc_hist[i], 1, a).long()",
        "P[\"anchor_mult\"]",
        "w_nn=P[\"_w_nn\"]",
        "wwv = np.exp(lk / max(ls_scale, 1e-6))",
        "nn.functional.huber_loss(net(xb), yb, delta=8.0",
    ]
    for fragment in required_fragments:
        assert fragment in source


def test_runtime_uses_sample_schema_and_has_no_visible_test_branch() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "TARGET_COLUMN = str(sample.columns[1])" in source
    assert 'sample_keys.str.rsplit("_", n=1, expand=True)' in source
    assert "expected_ids = set(sample_keys)" in source
    assert "actual_ids = set(prediction_lookup)" in source
    assert "submission[TARGET_COLUMN]" in source
    assert not re.search(r"if\s+sample_(?:rows|keys)\s*==", source)
    assert not re.search(r"if\s+len\(sample\)\s*==", source)
    assert not re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        source,
        re.IGNORECASE,
    )


def test_late_submit_label_is_visible_and_generator_is_reproducible() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "# # LATE SUBMIT — exp516" in source
    assert '"LATE_SUBMIT": True' in source
    assert (
        '"submission_message": "LATE SUBMIT | exp516 | 6th-place pfA x twGR '
        'standalone faithful replay | fixed v1"'
    ) in source
    subprocess.run([sys.executable, str(GENERATOR), "--check"], cwd=ROOT, check=True)
