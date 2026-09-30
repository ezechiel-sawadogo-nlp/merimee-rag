#!/bin/sh
# Au premier démarrage, construit corpus et index dans le volume /data (conservé ensuite).
set -e
if [ ! -f "${MERIMEE_INDEX_DIR:-/data/index}/meta.json" ]; then
    echo "Premier démarrage : préparation du corpus et des index (de 5 min avec GPU à ~40 min sur CPU ; une seule fois)…"
    merimee-rag prepare
    if [ -n "${MERIMEE_NO_DENSE:-}" ]; then
        merimee-rag index --no-dense
    else
        merimee-rag index
    fi
fi
exec "$@"
