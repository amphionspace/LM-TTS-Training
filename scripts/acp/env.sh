#!/usr/bin/env bash
ACP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACP_DOWNLOADS="$(cd -- "$ACP_DIR/../../../acp" && pwd)"
export SCO_HOME="$ACP_DOWNLOADS/.sco"
export SCO_DATA_HOME="$ACP_DOWNLOADS/.data"
export SCO_CONFIG="$ACP_DOWNLOADS/.config"
export PATH="$SCO_HOME/bin:$PATH"
