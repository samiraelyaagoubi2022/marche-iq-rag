"""Tests du chunker.  Lancer depuis la racine du projet :  python -m pytest tests/test_chunker.py -v

Cette suite remplace test_chunker_v2.py et reprend les tests du chunker de test_ingestion.py.
"""
import pytest

from src.ingestion import chunker


def page(n, texte, source="native", qr=None):
    return {"page": n, "source": source, "qr_codes": qr or [], "text": texte}


def pages_exemple():
    return [
        page(1, "Cahier des prescriptions spéciales\nMarché de travaux de voirie."),
        page(2, "Article 1 : Objet du marché\nConstruction d'une route communale en zone rurale."),
        page(3, "Article 2 : Caution provisoire\nLe montant est de 20 000 dirhams, à joindre au dossier.", "ocr"),
    ]


# =============================================================== titres
@pytest.mark.parametrize("ligne", [
    "Article 1 : Objet du marché", "ARTICLE 12 - Délai d'exécution", "Art. 3 Caution provisoire",
    "ART. 148. – Mesures en faveur des PME", "Article premier", "Article 1er", "Chapitre II",
    "المادة 4", "المادة5", "المادة 12 : الأجل", "الفصل 3", "المادة الأولى",
])
def test_titres_detectes(ligne):
    assert chunker.HEADING_RE.match(ligne)


@pytest.mark.parametrize("ligne", [
    "Le présent article précise les pièces à fournir.", "Voir l'article 5 ci-dessus",
    "Montant : 20 000 dirhams", "", "Article1", "المادة", "هذه المادة 5 تنص على",
    # lignes en MAJUSCULES : ne sont plus des titres (cause du montant perdu dans l'avis ANP)
    "QUATRE CENT TRENTE MILLE DIRHAMS", "BULLETIN OFFICIEL", "AGENCE NATIONALE DES PORTS",
])
def test_phrases_non_titres(ligne):
    assert not chunker.HEADING_RE.match(ligne)


# =============================================================== découpage long
def test_texte_long_respecte_la_taille():
    morceaux = chunker.split_long("Phrase de test numéro un. " * 300)
    assert len(morceaux) > 1
    assert all(len(m) <= chunker.CHUNK_SIZE for m in morceaux)


def test_debut_et_fin_conserves():
    morceaux = chunker.split_long("Debut" + "x" * 4000 + " fin.")
    assert morceaux[0].startswith("Debut") and morceaux[-1].endswith("fin.")


def test_aucun_mot_coupe_au_milieu():
    mots = " ".join(f"mot{i:03d}" for i in range(600))
    for morceau in chunker.split_long(mots):
        assert all(m.startswith("mot") and len(m) == 6 for m in morceau.split())


def test_pas_de_coupe_juste_apres_deux_points():
    texte = "x" * 700 + " fixé à:\nVALEUR " + "y" * 700
    cut = chunker._cut_point(texte, 0, 1200, 1200)
    assert not texte[:cut].rstrip().endswith(":")


# =============================================================== structure par articles
def test_un_chunk_par_article_avec_pages_et_source():
    chunks = {c["title"]: c for c in chunker.chunk_document("doc", pages_exemple())}
    assert chunks["Article 1 : Objet du marché"]["page_start"] == 2
    assert chunks["Article 1 : Objet du marché"]["source"] == "native"
    assert chunks["Article 2 : Caution provisoire"]["page_start"] == 3
    assert chunks["Article 2 : Caution provisoire"]["source"] == "ocr"


def test_texte_avant_le_premier_article_a_un_vrai_titre():
    chunks = chunker.chunk_document("doc", pages_exemple())
    assert chunks[0]["title"] == "Cahier des prescriptions spéciales"


def test_ids_uniques_champs_et_pas_de_chunk_vide():
    chunks = chunker.chunk_document("doc", pages_exemple())
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        assert set(c) >= {"chunk_id", "doc", "title", "page_start", "page_end", "source", "text"}
        assert len(c["text"]) >= chunker.MIN_CHUNK_CHARS


def test_articles_arabes_colles_sont_decoupes():
    pages = [page(i, f"المادة{i}\nنص المادة رقم {i} مع محتوى كاف لاجتياز جميع المرشحات والتحقق من الجودة.")
             for i in range(1, 4)]
    chunks = chunker.chunk_document("ar", pages)
    assert [c["title"] for c in chunks] == ["المادة1", "المادة2", "المادة3"]


# =============================================================== documents sans articles (avis)
def test_document_sans_titres_nest_pas_un_preambule():
    chunks = chunker.chunk_document("avis", [page(
        1, "AVIS D'APPEL D'OFFRES OUVERT\n\nLe jeudi 12 novembre 2026 il sera procédé à l'ouverture des plis.")])
    assert chunks
    assert "Page 1" in chunks[0]["title"] and "AVIS" in chunks[0]["title"]


def test_avis_court_une_page_un_chunk():
    texte = "AVIS D'APPEL D'OFFRES OUVERT\n\n" + ("Texte de l'avis, informations diverses. " * 45)
    chunks = chunker.chunk_document("avis", [page(1, texte)])
    assert len(chunks) == 1


def test_chaque_chunk_a_ses_propres_pages():
    pages = [
        page(1, "Première page, texte arabe ou autre de plus de trente caractères.", "ocr"),
        page(2, "Deuxième page, texte français de plus de trente caractères."),
    ]
    ancien = chunker.CHUNK_SIZE
    chunker.CHUNK_SIZE = 70
    try:
        chunks = chunker.chunk_document("d", pages)
    finally:
        chunker.CHUNK_SIZE = ancien
    assert len(chunks) == 2
    assert chunks[0]["page_start"] == chunks[0]["page_end"] == 1 and chunks[0]["source"] == "ocr"
    assert chunks[1]["page_start"] == chunks[1]["page_end"] == 2 and chunks[1]["source"] == "native"


def test_avis_sans_titres_ne_melange_pas_deux_pages():
    pages = [
        page(1, "Page arabe : texte assez long pour dépasser le seuil minimum.", "ocr"),
        page(2, "Page française : texte assez long pour dépasser le seuil minimum.", "ocr"),
    ]
    chunks = chunker.chunk_document("avis", pages)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 1), (2, 2)]


# =============================================================== le cas réel : montant de la caution (avis ANP)
def test_montant_reste_avec_sa_phrase():
    texte = (
        "Le\n14/09/2026\nà 11H00\nIl sera procédé à l'ouverture des plis.\n"
        "Le cautionnement provisoire est fixé à la somme de:\n"
        "QUATRE CENT TRENTE MILLE DIRHAMS\n"
        "L'estimation du coût des prestations est de :\nTTC\nTOUTE TAXES COMPRISES\n"
        "ROYAUME DU MAROC\nAGENCE NATIONALE DES PORTS\n"
        "AVIS D'APPEL D'OFFRES OUVERT INTERNATIONAL\n"
    )
    chunks = chunker.chunk_document("avis", [page(1, texte)])
    t = next(c["text"] for c in chunks if "fixé à la somme" in c["text"])
    assert "QUATRE CENT TRENTE MILLE DIRHAMS" in t


def test_montant_numerique_nest_pas_filtre_comme_bruit():
    texte = ("Article 1 : Cautionnement\nCautionnement provisoire\n430 000,00\n"
             "Article 2 : Objet\nLe présent marché a pour objet la fourniture de matériel.")
    chunks = chunker.chunk_document("cps", [page(1, texte)])
    assert any("430 000,00" in c["text"] for c in chunks)


# =============================================================== en-têtes / numéros de page
def test_entetes_et_numeros_de_page_retires():
    pages = [page(i, f"BULLETIN OFFICIEL\n{900 + i}\nArticle {i} : Objet numéro {i}\n"
                     f"Texte de l'article {i} avec assez de contenu pour passer tous les filtres.\n{i}")
             for i in range(1, 9)]
    chunks = chunker.chunk_document("decret", pages)
    assert len(chunks) == 8
    assert all("BULLETIN OFFICIEL" not in c["text"] for c in chunks)
    assert all("\n90" not in c["text"] for c in chunks)


def test_petit_document_garde_ses_lignes_repetees():
    # < 6 pages : pas de détection d'en-têtes (risque de supprimer du vrai contenu)
    pages = [page(i, f"Article {i} : Objet {i}\nTexte suffisamment long pour le chunk numéro {i}.\nMention légale")
             for i in range(1, 4)]
    chunks = chunker.chunk_document("petit", pages)
    assert all("Mention légale" in c["text"] for c in chunks)


# =============================================================== filtres : sommaire / titre seul
def test_sommaire_a_points_de_suite_ignore():
    toc = "\n".join(f"Article {i} : Objet du marché ........ {i + 2}" for i in range(1, 5))
    corps = ("Article 1 : Objet du marché\nConstruction d'une route communale en zone rurale.\n"
             "Article 2 : Prix\nLe prix est ferme et non révisable pour toute la durée.")
    chunks = chunker.chunk_document("cps", [page(1, toc), page(2, corps)])
    assert chunks and all("...." not in c["text"] for c in chunks)


def test_formulaire_a_trous_nest_pas_un_sommaire():
    texte = ("Article 1 : Déclaration\nNom du soumissionnaire : ..........\nAdresse du siège : ..........\n"
             "Fait à ......... le .......\n"
             "Article 2 : Objet\nLe présent marché a pour objet la fourniture de mobilier.")
    chunks = chunker.chunk_document("form", [page(1, texte)])
    assert any("Nom du soumissionnaire" in c["text"] for c in chunks)


def test_titre_seul_est_ignore():
    texte = ("Article 1 : Objet\nArticle 2 : Prix\n"
             "Le prix est ferme et non révisable pour toute la durée du marché.")
    chunks = chunker.chunk_document("d", [page(1, texte)])
    assert [c["title"] for c in chunks] == ["Article 2 : Prix"]


# =============================================================== fusion et contexte
def test_petit_morceau_fusionne_avec_le_precedent():
    long_par = ("Phrase de test numéro un. " * 44)[:1150]
    texte = ("Article 1 : Objet\n" + long_par + "\n\nFin du texte de l'article, mention finale.\n"
             "Article 2 : Prix\nLe prix est ferme et non révisable pour toute la durée du marché.")
    chunks = [c for c in chunker.chunk_document("d", [page(1, texte)]) if c["title"].startswith("Article 1")]
    assert len(chunks) == 1 and "mention finale" in chunks[0]["text"]


def test_contexte_document_injecte():
    pages = [page(1, "Article 1 : Objet\nTexte suffisamment long pour le chunk.\n"
                     "Article 2 : Prix\nAutre texte suffisamment long pour le chunk.")]
    avec = chunker.chunk_document("monDoc", pages)
    sans = chunker.chunk_document("monDoc", pages, with_context=False)
    assert all(c["text"].startswith("[Document : monDoc |") for c in avec)
    assert all("[Document :" not in c["text"] for c in sans)


# =============================================================== QR codes
def test_qr_codes_injectes_dans_le_texte():
    pages = [page(1, "Article 1 : Objet du marché\nConstruction d'une route communale.",
                  qr=["https://example.com/qr1"])]
    chunks = chunker.chunk_document("doc_qr", pages)
    assert len(chunks) == 1
    assert "[Liens QR Codes détectés :" in chunks[0]["text"]
    assert "https://example.com/qr1" in chunks[0]["text"]


def test_qr_codes_dedoublonnes():
    pages = [page(1, "Article 1 : Objet du marché\nConstruction d'une route communale.\n\n"
                     "Suite du même article avec un second paragraphe suffisamment long.",
                  qr=["https://a.ma", "https://a.ma"])]
    texte = chunker.chunk_document("d", pages)[0]["text"]
    assert texte.count("https://a.ma") == 1