from pathlib import Path

from scripts.validate_skills import validate_skills


def write_skill(
    skills_dir: Path,
    *,
    directory: str = "sample-skill",
    name: str = "sample-skill",
    description: str = "Use this skill for a repeatable sample workflow.",
    body: str = "# Sample skill\n\nFollow the workflow.",
) -> Path:
    skill_dir = skills_dir / directory
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    )
    return skill_dir


def test_validate_skills_accepts_skill_without_optional_ui_metadata(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir)

    errors, skill_count, metadata_count = validate_skills(skills_dir)

    assert not errors
    assert skill_count == 1
    assert metadata_count == 0


def test_validate_skills_rejects_name_mismatch_and_empty_body(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, name="different-skill", body="")

    errors, _, _ = validate_skills(skills_dir)

    assert any("name must match directory name" in error for error in errors)
    assert any("Markdown instructions are required" in error for error in errors)


def test_validate_skills_rejects_unexpected_front_matter_field(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = write_skill(skills_dir)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(skill_path.read_text().replace("---\n\n#", "metadata: {}\n---\n\n#"))

    errors, _, _ = validate_skills(skills_dir)

    assert any("unexpected front matter field(s): metadata" in error for error in errors)


def test_validate_skills_checks_optional_ui_metadata(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = write_skill(skills_dir)
    metadata_dir = skill_dir / "agents"
    metadata_dir.mkdir()
    (metadata_dir / "openai.yaml").write_text(
        "interface:\n"
        '  display_name: "Sample skill"\n'
        '  short_description: "too short"\n'
        '  default_prompt: "Run the sample workflow."\n'
    )

    errors, _, metadata_count = validate_skills(skills_dir)

    assert metadata_count == 1
    assert any("short_description must be 25-64 characters" in error for error in errors)
    assert any("default_prompt must mention $sample-skill" in error for error in errors)
