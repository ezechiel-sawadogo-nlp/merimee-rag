"""Métadonnées structurées de toute la base Mérimée, pour filtrer et compter.

Le RAG ne voit que les 23 442 notices qui ont un historique ; les filtres et comptages de
l'agent portent sur les 24 824 notices, sinon « combien d'églises classées dans l'Aube ? »
serait faux. Les valeurs sont comparées sans accents ni casse (« Cote-d'Or » = « Côte-d'Or »).
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Iterable

POP_URL = "https://pop.culture.gouv.fr/notice/merimee/{ref}"


def norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _as_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x not in (None, "")]
    return [str(v)]


@dataclass
class Record:
    ref: str
    titre: str
    commune: list[str]
    departement: str
    region: str
    denomination: list[str]
    domaine: list[str]
    siecles: list[int]
    siecle_principal: int | None
    protection: str
    annee_protection: int | None
    statut: str
    auteurs: list[str]
    historique: str
    coordonnees: str
    # formes normalisées, calculées une fois
    n: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: dict) -> "Record":
        siecles = []
        for s in _as_list(d.get("siecles")):
            m = re.match(r"-?\d+", s)
            if m:
                siecles.append(int(m.group(0)))
        r = cls(
            ref=str(d.get("id") or d.get("ref") or ""), titre=str(d.get("titre") or ""),
            commune=[c.strip() for c in str(d.get("commune") or "").split(";") if c.strip()],
            departement=str(d.get("departement") or ""), region=str(d.get("region") or ""),
            denomination=_as_list(d.get("denomination")), domaine=_as_list(d.get("domaine")),
            siecles=siecles, siecle_principal=d.get("siecle_principal"),
            protection=str(d.get("protection") or ""), annee_protection=d.get("annee_protection"),
            statut=str(d.get("statut_proprietaire") or ""), auteurs=_as_list(d.get("auteur")),
            historique=str(d.get("historique") or ""), coordonnees=str(d.get("coordonnees") or ""),
        )
        r.n = {
            "commune": [norm(c) for c in r.commune], "departement": norm(r.departement),
            "region": norm(r.region), "denomination": [norm(x) for x in r.denomination],
            "domaine": [norm(x) for x in r.domaine], "protection": norm(r.protection),
            "statut": norm(r.statut), "auteurs": [set(norm(a).split()) for a in r.auteurs],
            "texte": norm(f"{r.titre} {r.historique}"),
        }
        return r

    @property
    def url(self) -> str:
        return POP_URL.format(ref=self.ref)

    def summary(self) -> dict:
        """Forme compacte renvoyée au LLM (listes de résultats)."""
        return {"ref": self.ref, "titre": self.titre, "commune": ", ".join(self.commune),
                "departement": self.departement, "denomination": ", ".join(self.denomination),
                "siecles": self.siecles, "protection": self.protection,
                "annee_protection": self.annee_protection}


# Champs acceptés par filter_notices / count, et leur sémantique de comparaison.
FILTER_FIELDS = ("commune", "departement", "region", "denomination", "domaine", "siecle",
                 "protection", "statut", "auteur", "annee_min", "annee_max", "mot_cle")
GROUP_FIELDS = ("region", "departement", "commune", "denomination", "siecle_principal", "protection",
                "statut", "domaine")


PROTECTION_ALL = ("tous", "toute", "protege", "all")


class FilterError(ValueError):
    """Valeur de filtre inconnue ; le message propose les valeurs proches (utile à l'agent)."""


def _word_in(needle: str, hay: str) -> bool:
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}", hay) is not None


def _parse_siecle(v: Any) -> int:
    m = re.search(r"-?\d+", str(v))
    if not m:
        raise FilterError(f"siècle illisible : {v!r} (attendu un nombre, ex. 12 pour le 12e siècle)")
    return int(m.group(0))


class MetadataStore:
    def __init__(self, records: Iterable[Record]) -> None:
        self.records = [r for r in records if r.ref]
        self.by_ref = {r.ref: r for r in self.records}
        self._vocab = {
            "departement": sorted({r.departement for r in self.records if r.departement}),
            "region": sorted({r.region for r in self.records if r.region}),
            "commune": sorted({c for r in self.records for c in r.commune}),
            "denomination": sorted({d for r in self.records for d in r.denomination}),
        }
        self._vocab_n = {k: {norm(v): v for v in vals} for k, vals in self._vocab.items()}

    @classmethod
    def load(cls, path: Path) -> "MetadataStore":
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        return cls(Record.from_json(d) for d in data)

    # ------------------------------------------------------------------ filtrage
    def _check_vocab(self, field_: str, value: str) -> str:
        """Vérifie qu'une valeur existe ; sinon lève FilterError avec des suggestions."""
        v = norm(value)
        vocab = self._vocab_n[field_]
        if field_ in ("departement", "region", "commune"):
            if v in vocab:
                return v
        else:  # dénomination : « croix » couvre « croix de chemin » ; « églises » → « église »
            for cand in (v, v[:-1] if v.endswith(("s", "x")) else None, v[:-3] + "al" if v.endswith("aux") else None):
                if cand and any(_word_in(cand, k) for k in vocab):
                    return cand
        close = get_close_matches(v, list(vocab), n=4, cutoff=0.6)
        close += [k for k in vocab if v and v in k and k not in close][:4]
        hint = f" Valeurs proches : {', '.join(vocab[c] for c in close[:5])}." if close else ""
        # Erreur fréquente : une région passée comme département (ou l'inverse), une commune comme département…
        for other in ("region", "departement", "commune"):
            if other == field_:
                continue
            ov = self._vocab_n[other]
            hits = [ov[k] for k in ov if k == v] or [ov[k] for k in ov if v and v in k][:3]
            if hits:
                hint += f" « {value} » correspond à : {', '.join(hits)} (champ « {other} »)."
                break
        raise FilterError(f"{field_} inconnu : « {value} ».{hint}")

    def match(self, **filters: Any) -> list[Record]:
        f = {k: v for k, v in filters.items() if v not in (None, "", [])}
        # « tous » / « protégé » : toute la base est protégée, ce n'est pas un filtre
        if isinstance(f.get("protection"), str) and norm(f["protection"]).startswith(PROTECTION_ALL):
            del f["protection"]
        unknown = set(f) - set(FILTER_FIELDS)
        if unknown:
            raise FilterError(f"filtre(s) inconnu(s) : {sorted(unknown)}. Filtres possibles : {list(FILTER_FIELDS)}")
        multi = [k for k, v in f.items() if isinstance(v, (list, dict, tuple, set))]
        if multi:
            raise FilterError(
                f"une seule valeur par filtre ({', '.join(multi)} reçu une liste ou un objet). Pour comparer "
                "plusieurs départements, régions ou communes, appelle count une seule fois avec le filtre "
                "englobant (ex. region=…) et group_by='departement' ; le résultat donne le nombre pour chacun.")
        crit: dict[str, Any] = {}
        for k in ("commune", "departement", "region", "denomination"):
            if k in f:
                crit[k] = self._check_vocab(k, str(f[k]))
        for k in ("domaine", "protection", "statut", "auteur", "mot_cle"):
            if k in f:
                crit[k] = norm(f[k])
        if "siecle" in f:
            crit["siecle"] = _parse_siecle(f["siecle"])
        for k in ("annee_min", "annee_max"):
            if k in f:
                crit[k] = int(_parse_siecle(f[k]))

        out = []
        for r in self.records:
            n = r.n
            if "commune" in crit and crit["commune"] not in n["commune"]:
                continue
            if "departement" in crit and crit["departement"] != n["departement"]:
                continue
            if "region" in crit and crit["region"] != n["region"]:
                continue
            if "denomination" in crit and not any(_word_in(crit["denomination"], d) for d in n["denomination"]):
                continue
            if "domaine" in crit and not any(crit["domaine"] in d for d in n["domaine"]):
                continue
            if "protection" in crit and not n["protection"].startswith(crit["protection"][:5]):
                continue
            if "statut" in crit and not n["statut"].startswith(crit["statut"][:5]):
                continue
            # auteur : tous les mots demandés présents dans un même nom, dans n'importe quel ordre
            # (« Jean Bossu » trouve « Bossu Jean (architecte) »)
            if "auteur" in crit and not any(set(crit["auteur"].split()) <= a for a in n["auteurs"]):
                continue
            if "mot_cle" in crit and not _word_in(crit["mot_cle"], n["texte"]):
                continue
            if "siecle" in crit and crit["siecle"] not in r.siecles:
                continue
            if "annee_min" in crit and (r.annee_protection is None or r.annee_protection < crit["annee_min"]):
                continue
            if "annee_max" in crit and (r.annee_protection is None or r.annee_protection > crit["annee_max"]):
                continue
            out.append(r)
        return out

    def group_counts(self, records: list[Record], by: str) -> list[tuple[str, int]]:
        if by not in GROUP_FIELDS:
            raise FilterError(f"regroupement impossible par « {by} ». Possibles : {list(GROUP_FIELDS)}")
        counts: dict[str, int] = {}
        for r in records:
            vals: list[str]
            if by == "commune":
                vals = r.commune
            elif by in ("denomination", "domaine"):
                vals = getattr(r, by)
            elif by == "siecle_principal":
                vals = [f"{r.siecle_principal}e siècle"] if r.siecle_principal else []
            elif by == "statut":
                vals = [r.statut] if r.statut else []
            else:
                v = getattr(r, by)
                vals = [v] if v else []
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
