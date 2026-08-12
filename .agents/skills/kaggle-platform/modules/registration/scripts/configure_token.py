from __future__ import annotations

import getpass
from pathlib import Path


def main() -> None:
    destination = Path.home() / ".kaggle" / "access_token"
    if destination.exists():
        raise SystemExit(
            f"credential already exists at {destination}; rotate it in Kaggle Settings "
            "and remove the local file explicitly before replacing it"
        )

    token = getpass.getpass("Kaggle API token (input hidden): ").strip()
    if not token:
        raise SystemExit("empty token; nothing was written")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(token)
    destination.chmod(0o600)
    print(f"Saved Kaggle API token to {destination} with mode 600")


if __name__ == "__main__":
    main()
