"""Shell client for the isolated Science task environment."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Bash command to execute inside the task container")
    parser.add_argument("--cwd", help="Absolute task-container directory; defaults to the official workdir")
    parser.add_argument("--timeout", type=float, default=120, help="Command timeout, at most 300 seconds")
    args = parser.parse_args()
    endpoint = json.loads(Path(__file__).with_name("science-endpoint.json").read_text())
    data = json.dumps({"command": args.command, "cwd": args.cwd, "timeout": args.timeout}).encode()
    request = Request(endpoint["base_url"] + "/exec", data=data,
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + endpoint["token"]})
    with urlopen(request, timeout=args.timeout + 15) as response:
        print(json.dumps(json.load(response), ensure_ascii=False))


if __name__ == "__main__":
    main()
