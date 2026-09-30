#!/bin/sh
# Au premier démarrage, construit corpus et index dans le volume /data (conservé ensuite).
set -e
if [ ! -f "${MERIMEE_INDEX_DIR:-/data/index}/meta.json" ]; then
    echo "Premier démarrage : préparation du corpus et des index (5 à 20 min selon la machine)…"
    merimee-rag prepare
    if [ -n "${MERIMEE_NO_DENSE:-}" ]; then
        merimee-rag index --no-dense
    else
        merimee-rag index
    fi
fi
exec "$@"
