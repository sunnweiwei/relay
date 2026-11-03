# Image for testing real harnesses through Relay. Built once (`docker build --pull`: OpenClaw needs
# the newest Node 24); the repo, Codex and Claude Code binaries, credentials and a throwaway HOME
# are mounted at run time.
FROM node:24-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir httpx starlette uvicorn truststore
RUN npm install -g --no-fund --no-audit \
    @earendil-works/pi-coding-agent opencode-ai @google/gemini-cli \
    @charmland/crush @kilocode/cli openclaw
RUN pip3 install --break-system-packages --no-cache-dir zstandard
RUN curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh \
    | GOOSE_BIN_DIR=/usr/local/bin CONFIGURE=false bash
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh \
    | KIMI_INSTALL_DIR=/opt/kimi-code KIMI_NO_MODIFY_PATH=1 bash \
    && ln -sf /opt/kimi-code/bin/kimi /usr/local/bin/kimi && chmod -R a+rX /opt/kimi-code
RUN npm install -g --no-fund --no-audit @deepseek-ai/dsh
RUN npm install -g --no-fund --no-audit @tencent-ai/codebuddy-code
RUN pip3 install --break-system-packages --no-cache-dir nanobot-ai
# Hermes Agent left PyPI (0.19.0 there) for its own installer, on Python 3.14 and its uv.lock:
# its tagged source, synced from that lock.
RUN pip3 install --break-system-packages --no-cache-dir uv \
    && git clone -q --depth 1 -b v0.22.5 https://github.com/NousResearch/hermes-agent /opt/hermes-agent \
    && cd /opt/hermes-agent && UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PROJECT_ENVIRONMENT=/opt/hermes-venv \
       uv sync --frozen --no-dev --python 3.14 \
    && ln -sf /opt/hermes-venv/bin/hermes /usr/local/bin/hermes && chmod -R a+rX /opt/hermes-agent /opt/hermes-venv /opt/uv-python
# RELAY_STRATEGY=rlm runs the RLM authors' package inside Relay.
RUN pip3 install --break-system-packages --no-cache-dir rlms==0.1.3
# LiteLLM's proxy and mini-swe-agent (whose litellm is a library) in venvs of their own: litellm
# needs mcp>=2.2, nanobot needs mcp<2.
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/litellm && /opt/litellm/bin/pip install --no-cache-dir 'litellm[proxy]' \
    && ln -sf /opt/litellm/bin/litellm /usr/local/bin/litellm \
    && python3 -m venv /opt/mini && /opt/mini/bin/pip install --no-cache-dir mini-swe-agent mcp \
    && ln -sf /opt/mini/bin/mini /usr/local/bin/mini
