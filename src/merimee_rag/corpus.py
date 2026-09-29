"""Chargement de la base Mérimée, filtrage des notices avec historique, chunking.

Le CSV Mérimée a changé plusieurs fois de format (codes POP « REF/TICO/HIST »,
libellés longs « Historique », « Reference »…). On résout donc les colonnes par
alias normalisés plutôt que par noms figés.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

POP_URL = "https://pop.culture.gouv.fr/notice/merimee/{ref}"

# Champ canonique -> alias possibles (déjà normalisés, par ordre de préférence).
COLUMN_ALIASES: dict[str, list[str]] = {
    "ref": ["reference", "ref", "reference_de_la_notice"],
    "title": ["titre_editorial_de_la_notice", "appellation_courante", "tico", "titre_courant", "titre", "appellation"],
    "denomination": ["denomination_de_l_edifice", "denomination", "deno"],
    "commune": ["commune_forme_editoriale", "commune_forme_index", "commune", "com"],
    "departement": ["departement_en_lettres", "departement_lettres", "departement", "dpt_lettre", "departement_format_numerique", "dpt"],
    "region": ["region", "reg"],
    "siecle": ["siecle_de_la_campagne_principale_de_construction", "siecle", "scle"],
    "datation": ["datation_de_l_edifice", "datation", "date"],
    "protection": ["typologie_de_la_protection", "nature_de_la_protection", "protection", "prot"],
    "statut": ["statut_juridique_de_l_edifice", "statut_juridique", "statut", "stat"],
    "auteurs": ["auteur_de_l_edifice", "auteurs", "auteur", "autr"],
    "historique": ["historique", "hist"],
    "description": ["description_de_l_edifice", "description", "desc"],
    "coords": ["coordonnees_au_format_wgs84", "coordonnees_wgs84", "coordonnees", "coor", "geolocalisation"],
}
# Si aucun alias exact ne matche, on accepte une colonne qui *commence* par ces préfixes.
PREFIX_FALLBACK = {"historique": "historique", "title": "titre_editorial", "commune": "commune",
                   "siecle": "siecle", "coords": "coordonnees"}
REQUIRED = ("ref", "historique")


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def resolve_columns(columns: Iterable[str]) -> dict[str, str]:
    """Associe chaque champ canonique à une colonne réelle du CSV."""
    norm = {normalize_name(c): c for c in columns}
    mapping: dict[str, str] = {}
    for field_, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in norm:
                mapping[field_] = norm[a]
                break
        else:
            prefix = PREFIX_FALLBACK.get(field_)
            if prefix:
                hits = [orig for n, orig in norm.items() if n.startswith(prefix)]
                if hits:
                    mapping[field_] = hits[0]
    missing = [f for f in REQUIRED if f not in mapping]
    if missing:
        raise ValueError(
            f"Colonnes introuvables pour {missing}. Colonnes du CSV : {list(columns)[:60]}…\n"
            "Ajoute l'alias dans COLUMN_ALIASES (src/merimee_rag/corpus.py)."
        )
    return mapping


def sniff_delimiter(path: Path, encoding: str) -> str:
    with open(path, encoding=encoding, newline="") as f:
        head = f.readline()
    counts = {d: head.count(d) for d in (";", ",", "|", "\t")}
    return max(counts, key=counts.get)


def detect_encoding(path: Path) -> str:
    raw = Path(path).open("rb").read(200_000)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace(" ", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_coords(value: str) -> tuple[float | None, float | None]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", value or "")
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        # Heuristique France : lat ∈ [41, 52] ; sinon on suppose (lon, lat).
        # (Les DOM-TOM ont des latitudes hors de cet intervalle : on garde l'ordre lu.)
        if 41 <= b <= 52 and not 41 <= a <= 52:
            a, b = b, a
        return a, b
    return None, None


@dataclass
class Notice:
    ref: str
    title: str
    denomination: str
    commune: str
    departement: str
    region: str
    siecle: str
    datation: str
    protection: str
    statut: str
    auteurs: str
    historique: str
    description: str
    lat: float | None
    lon: float | None

    @property
    def url(self) -> str:
        return POP_URL.format(ref=self.ref)

    def header(self) -> str:
        """Contexte court préfixé à chaque chunk (« contextual chunking »)."""
        loc = ", ".join(x for x in (self.commune, self.departement) if x)
        parts = [self.title or self.denomination or self.ref]
        if loc:
            parts.append(f"({loc})")
        head = " ".join(parts)
        extra = []
        if self.denomination and self.denomination.lower() not in head.lower():
            extra.append(f"Dénomination : {self.denomination}")
        if self.siecle:
            extra.append(f"Siècle : {self.siecle}")
        if self.protection:
            extra.append(f"Protection : {self.protection}")
        if self.auteurs:
            extra.append(f"Auteur(s) : {self.auteurs}")
        return head + (". " + ". ".join(extra) if extra else "") + "."


def iter_csv_notices(path: Path, min_hist_chars: int = 80) -> Iterator[Notice]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    enc = detect_encoding(path)
    delim = sniff_delimiter(path, enc)
    with open(path, encoding=enc, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = resolve_columns(reader.fieldnames or [])
        get = lambda row, k: clean(row.get(cols[k], "")) if k in cols else ""  # noqa: E731
        seen: set[str] = set()
        for row in reader:
            ref = get(row, "ref")
            hist = get(row, "historique")
            if not ref or ref in seen or len(hist) < min_hist_chars:
                continue
            seen.add(ref)
            lat, lon = parse_coords(get(row, "coords"))
            yield Notice(
                ref=ref, title=get(row, "title"), denomination=get(row, "denomination"),
                commune=get(row, "commune"), departement=get(row, "departement"),
                region=get(row, "region"), siecle=get(row, "siecle"), datation=get(row, "datation"),
                protection=get(row, "protection"), statut=get(row, "statut"), auteurs=get(row, "auteurs"),
                historique=hist, description=get(row, "description"), lat=lat, lon=lon,
            )


def _join(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(clean(str(x)) for x in v if x not in (None, ""))
    return clean(str(v))


def _siecles(v) -> str:
    if not v:
        return ""
    if isinstance(v, list):
        nums = [int(x) for x in v if str(x).lstrip("-").isdigit()]
        if nums:
            return ", ".join(f"{n}e" for n in nums) + " siècle" + ("s" if len(nums) > 1 else "")
    return _join(v)


def iter_json_notices(path: Path, min_hist_chars: int = 80) -> Iterator[Notice]:
    """Format JSON « nettoyé » (celui préparé pour l'index Solr du cours) :
    liste d'objets {id, titre, denomination[], commune, departement, region, protection,
    annee_protection, statut_proprietaire, historique, siecles[], auteur[], coordonnees "lat,lon"}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # {"docs": [...]} ou similaire
        data = next((v for v in data.values() if isinstance(v, list)), [])
    seen: set[str] = set()
    for d in data:
        ref = clean(str(d.get("id") or d.get("ref") or d.get("reference") or ""))
        hist = clean(d.get("historique"))
        if not ref or ref in seen or len(hist) < min_hist_chars:
            continue
        seen.add(ref)
        prot = _join(d.get("protection"))
        if prot and d.get("annee_protection"):
            prot += f" en {d['annee_protection']}"
        if prot and d.get("protection_partielle"):
            prot += " (partiellement)"
        lat, lon = parse_coords(_join(d.get("coordonnees")))
        yield Notice(
            ref=ref, title=clean(d.get("titre")), denomination=_join(d.get("denomination")),
            commune=clean(d.get("commune")).replace(";", ", "), departement=clean(d.get("departement")),
            region=clean(d.get("region")), siecle=_siecles(d.get("siecles")),
            datation=_join(d.get("periode")), protection=prot,
            statut=_join(d.get("statut_proprietaire")), auteurs=_join(d.get("auteur")),
            historique=hist, description=_join(d.get("domaine")), lat=lat, lon=lon,
        )


def iter_notices(path: Path, min_hist_chars: int = 80) -> Iterator[Notice]:
    """Choisit le lecteur selon l'extension (.json, .json.gz ou .csv)."""
    if str(path).lower().endswith((".json", ".json.gz")):
        return iter_json_notices(path, min_hist_chars)
    return iter_csv_notices(path, min_hist_chars)


def save_corpus(notices: Iterable[Notice], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for nt in notices:
            f.write(json.dumps(asdict(nt), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_corpus(path: Path) -> list[Notice]:
    with open(path, encoding="utf-8") as f:
        return [Notice(**json.loads(line)) for line in f if line.strip()]


# --------------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    chunk_id: str
    ref: str
    position: int
    text: str  # header + passage : c'est ce qui est indexé et montré au LLM


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def chunk_notice(nt: Notice, chunk_words: int = 180, overlap: int = 40) -> list[Chunk]:
    """Découpe l'historique en fenêtres de ~chunk_words mots, alignées sur les phrases.

    Chaque chunk est préfixé par l'en-tête de la notice, pour qu'un passage isolé
    (« Il fut reconstruit en 1760… ») reste rattaché à son monument et sa commune.
    """
    header = nt.header()
    sents = [s for s in _SENT_SPLIT.split(nt.historique) if s]
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_len = 0
    i = 0
    while i < len(sents):
        s = sents[i]
        w = len(s.split())
        if cur and cur_len + w > chunk_words:
            chunks.append(" ".join(cur))
            # recouvrement : on reprend les dernières phrases jusqu'à ~overlap mots
            back, back_len = [], 0
            for prev in reversed(cur):
                back_len += len(prev.split())
                if back_len > overlap:
                    break
                back.insert(0, prev)
            cur, cur_len = back, sum(len(x.split()) for x in back)
            if cur_len + w > chunk_words:  # la phrase seule est trop longue
                cur, cur_len = [], 0
            continue
        cur.append(s)
        cur_len += w
        i += 1
    if cur:
        chunks.append(" ".join(cur))
    return [Chunk(f"{nt.ref}#{k}", nt.ref, k, f"{header}\n{txt}") for k, txt in enumerate(chunks)]


def build_chunks(notices: Iterable[Notice], chunk_words: int = 180, overlap: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for nt in notices:
        out.extend(chunk_notice(nt, chunk_words, overlap))
    return out
