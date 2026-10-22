"""Tests du générateur (sans réseau).  python -m pytest tests/test_generator.py -v"""
import pytest

from src.generation import generator as g

CHUNKS = [
    {"chunk_id": f"d::000{i}", "doc": "AVIS ANP", "title": "Page 1", "page_start": 1,
     "source": "native", "text": f"Texte {i} : le cautionnement provisoire est fixé à 430 000 DH."}
    for i in range(1, 4)
]


# ---------------------------------------------------------------- normalisation / extraction
@pytest.mark.parametrize("brut, attendu", [
    ("Caution : 430 000 DH [S1].", "Caution : 430 000 DH [S1]."),
    ("Montant [S1][S3].", "Montant [S1][S3]."),
    ("Montant [S1, S3].", "Montant [S1][S3]."),
    ("Montant [S1; S3].", "Montant [S1][S3]."),
    ("Montant [S 2].", "Montant [S2]."),
    ("Montant [ S2 ].", "Montant [S2]."),
    ("Montant (S2).", "Montant [S2]."),
    ("Montant 【S2】.", "Montant [S2]."),
    ("Montant [S1-S3].", "Montant [S1][S2][S3]."),
    ("Montant [S1 à S2].", "Montant [S1][S2]."),
    ("Montant [S1, p.7].", "Montant [S1]."),
    ("Montant [Extrait 2].", "Montant [S2]."),
    ("Montant [Source 3, page 1].", "Montant [S3]."),
    ("Surface (10 m²) et délai [30 jours].", "Surface (10 m²) et délai [30 jours]."),   # pas de citation : intact
])
def test_normalize_cites(brut, attendu):
    assert g.normalize_cites(brut) == attendu


def test_normalize_est_idempotent():
    once = g.normalize_cites("A [S1, S3] et (S2).")
    assert g.normalize_cites(once) == once


def test_extract_cites_valides_et_invalides():
    valides, invalides = g.extract_cites("A [S1][S3]. B [S9].", n_max=5)
    assert valides == [1, 3] and invalides == [9]


def test_extract_cites_aucune():
    assert g.extract_cites("Aucune citation ici.", 5) == ([], [])


# ---------------------------------------------------------------- refus
def test_refus_avec_citation_est_nettoye():
    assert g.clean_answer(g.NOT_FOUND + " [S1]") == g.NOT_FOUND
    assert g.clean_answer("Je ne trouve pas cette information dans les documents fournis. [Extrait 3, page 1]") == g.NOT_FOUND


def test_reponse_partielle_commencant_par_un_refus_est_conservee():
    longue = g.NOT_FOUND + " Toutefois, l'avis précise que le dépôt des plis se fait au bureau d'ordre [S2]."
    assert g.clean_answer(longue) == longue


# ---------------------------------------------------------------- generate() avec un faux Groq
class _Msg:
    def __init__(self, content): self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]
        self.usage = None


class FauxGroq:
    """Renvoie les réponses données, dans l'ordre."""
    def __init__(self, reponses):
        self.reponses, self.appels = list(reponses), []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        self.appels.append(kw)
        return _Resp(self.reponses.pop(0))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(g, "retrieve", lambda q, k=5: (CHUNKS, "ANP"))

    def installer(reponses):
        faux = FauxGroq(reponses)
        monkeypatch.setattr(g, "Groq", lambda **kw: faux)
        return faux
    return installer


def test_generate_citation_normale(env):
    faux = env(["La caution est de 430 000 DH [S1, S2]."])
    r = g.generate("Quelle est la caution ANP ?")
    assert r["answer"] == "La caution est de 430 000 DH [S1][S2]."
    assert [s["id"] for s in r["cited"]] == ["S1", "S2"]
    assert r["invalid_citations"] == [] and r["retried"] is False and len(faux.appels) == 1
    assert set(r["timings"]) == {"retrieval_s", "llm_s", "total_s"}
    assert "error" not in r          # eval_rag.py teste la présence de la clé


def test_generate_relance_si_aucune_citation(env):
    faux = env(["La caution est de 430 000 DH.", "La caution est de 430 000 DH. [S1]"])
    r = g.generate("Quelle est la caution ANP ?")
    assert r["retried"] is True and len(faux.appels) == 2
    assert [s["id"] for s in r["cited"]] == ["S1"]
    # la relance contient bien la réponse précédente puis le rappel
    roles = [m["role"] for m in faux.appels[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_generate_pas_de_relance_pour_un_refus(env):
    faux = env([g.NOT_FOUND + " [S1]"])
    r = g.generate("Qui est l'attributaire ?")
    assert r["answer"] == g.NOT_FOUND and r["answerable"] is False
    assert r["cited"] == [] and r["retried"] is False and len(faux.appels) == 1


def test_generate_citation_inexistante_signalee(env):
    env(["Caution 430 000 DH [S1][S9]."])
    r = g.generate("Quelle est la caution ANP ?")
    assert r["invalid_citations"] == [9] and [s["id"] for s in r["cited"]] == ["S1"]


def test_generate_reponse_vide_reessaie_en_effort_low(env):
    faux = env(["", "Caution 430 000 DH [S1]."])
    r = g.generate("Quelle est la caution ANP ?")
    assert len(faux.appels) == 2 and faux.appels[1]["extra_body"] == {"reasoning_effort": "low"}
    assert "error" not in r and r["cited"]


def test_generate_reponse_toujours_vide_est_signalee(env):
    env(["", ""])
    r = g.generate("Quelle est la caution ANP ?")
    assert r["error"] == "reponse_vide" and r["answerable"] is False


# ---------------------------------------------------------------- dédoublonnage / filtre document
def test_dedupe_garde_l_ordre():
    doublon = [CHUNKS[0], {**CHUNKS[0], "chunk_id": "autre"}, CHUNKS[1]]
    assert [c["chunk_id"] for c in g.dedupe(doublon, 5)] == ["d::0001", "d::0002"]


@pytest.mark.parametrize("q, doc", [
    ("Quelle est la caution du marché ANP ?", "ANP"),
    ("Que dit le décret sur la caution ?", "Decret"),
    ("Pièces du règlement de consultation ?", "Reglement"),
    ("Quelle est la capitale ?", None),
])
def test_detect_doc_filter(q, doc):
    assert g.detect_doc_filter(q) == doc