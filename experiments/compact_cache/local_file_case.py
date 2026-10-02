"""Two real user turns over the same two-file local workspace."""
from __future__ import annotations

import shutil
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "local_config" / "config"
FIRST = ("请查看当前工作区的配置文件，告诉我 production 环境的 retry_delay_ms "
         "实际是多少毫秒，以及这个值来自哪个文件。")
SECOND = "按你刚才找到的 production retry_delay_ms，连续等待 3 次总共是多少毫秒？"
EXPECTED_DELAY = "750"
EXPECTED_TOTAL = "2250"


def install(workspace: Path) -> None:
    target = workspace / "config"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("default.json", "production.json"):
        shutil.copyfile(FIXTURES / name, target / name)
