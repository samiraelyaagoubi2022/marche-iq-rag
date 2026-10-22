"""Interface MarchéIQ (Streamlit) : assistant RAG sur appels d'offres."""
import html
import json
import os
import re
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="MarchéIQ", page_icon="📑", layout="wide")

# ---------------------------------------------------------------------------
# Style : palette "dossier d'appel d'offres" (encre, papier, surligneur)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');
:root{
  --encre:#12263A; --papier:#F4F6F7; --sarcelle:#0E7C86;
  --surligneur:#FFE066; --gris:#5B6B7A; --bordure:#D5DCE1;
}
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea{
  font-family:'IBM Plex Sans', sans-serif;
}
h1, h2, h3, .miq-title{
  font-family:'Bricolage Grotesque', sans-serif !important;
  color:var(--encre); letter-spacing:-0.02em;
}
.block-container{ padding-top:2.2rem; max-width:1100px; }

.miq-title{ font-size:2.6rem; font-weight:700; line-height:1.05; margin:0; }
.miq-sub{ color:var(--gris); font-size:1.05rem; margin:.4rem 0 1.6rem 0; max-width:42rem; }

.cite{
  display:inline-block; min-width:1.5em; text-align:center;
  background:var(--encre); color:#fff; border-radius:4px;
  font-size:.7em; font-weight:500; padding:0 .35em; margin:0 .1em;
  vertical-align:super;
}

.passage{
  border-left:3px solid var(--sarcelle); background:#fff;
  padding:.8rem 1rem; border-radius:0 6px 6px 0;
  white-space:pre-wrap; font-size:.92rem; line-height:1.55; color:var(--encre);
}
.passage mark{ background:var(--surligneur); padding:0 2px; border-radius:2px; }

.chips{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.8rem 0 .4rem 0; }
.chip{
  border:1px solid var(--bordure); background:#fff; color:var(--gris);
  border-radius:999px; padding:.15rem .75rem; font-size:.82rem;
}
.chip b{ color:var(--encre); font-weight:500; }
.chip.warn{ border-color:#C0392B; color:#C0392B; background:#FDF3F2; }

.pipe{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.8rem; margin:1rem 0; }
.pipe div{ background:#fff; border:1px solid var(--bordure); border-radius:8px; padding:.9rem 1rem; font-size:.9rem; color:var(--gris); }
.pipe div strong{ display:block; color:var(--encre); font-family:'Bricolage Grotesque'; font-size:1.05rem; margin-bottom:.2rem; }

.stButton button{ border:1px solid var(--bordure); border-radius:8px; text-align:left; }
.stButton button:hover{ border-color:var(--sarcelle); color:var(--sarcelle); }
.ex-tag{ font-size:.8rem; color:var(--sarcelle); font-weight:500; margin-bottom:.2rem; }

@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
STOPWORDS = {
    "quelle", "quel", "quels", "quelles", "dans", "pour", "avec", "est", "des",
    "les", "une", "que", "qui", "sur", "marché", "date", "pourcentage",
}

EXEMPLES = [
    ("ANP", "Quelle est la caution provisoire du marché ANP ?"),
    ("BERD", "Quelle est la date d'ouverture des plis du marché BERD ?"),
    ("FNUAP", "Quel est le pourcentage du cautionnement définitif dans le CPS FNUAP ?"),
]


def highlight(text: str, question: str) -> str:
    """Échappe le texte et surligne les mots de la question."""
    safe = html.escape(text)
    terms = {
        w for w in re.findall(r"\w+", question.lower())
        if len(w) > 3 and w not in STOPWORDS
    }
    if not terms:
        return safe
    pattern = re.compile(
        "(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + ")",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", safe)


def answer_html(answer: str) -> str:
    """Transforme les [1], [2]... en badges de citation."""
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', html.escape(answer))


def to_markdown(entry: dict) -> str:
    lines = [f"# {entry['q']}", "", entry["answer"], "", "## Sources", ""]
    for s in entry["cited"]:
        lines.append(f"- [{s['id']}] {s['doc']}, {s['title']} (p.{s['page']})")
    lines.append(f"\n_Généré par MarchéIQ le {entry['ts']}_")
    return "\n".join(lines)


def save_feedback(idx: int) -> None:
    value = st.session_state.get(f"fb_{idx}")
    if value is None:
        return
    entry = st.session_state.history[idx]
    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
        record = {"ts": entry["ts"], "question": entry["q"], "useful": bool(value)}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@st.cache_resource(show_spinner=False)
def get_generate():
    """Charge le modèle une seule fois, au premier usage."""
    from src.generation.generator import generate
    return generate


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
def render_sources(entry: dict) -> None:
    for s in entry["cited"]:
        with st.expander(f"[{s['id']}] {s['doc']} : {s['title']} (p.{s['page']})"):
            st.markdown(
                f"<div class='passage'>{highlight(s['text'][:1500], entry['q'])}</div>",
                unsafe_allow_html=True,
            )


def render_assistant(idx: int, entry: dict, stream: bool = False) -> None:
    with st.chat_message("assistant"):
        slot = st.empty()
        if stream:
            buffer = ""
            for token in re.split(r"(\s+)", entry["answer"]):
                buffer += token
                slot.markdown(html.escape(buffer) + " ▌")
                time.sleep(0.008)
        slot.markdown(answer_html(entry["answer"]), unsafe_allow_html=True)

        n_src = len(entry["cited"])
        n_docs = len({s["doc"] for s in entry["cited"]})
        chips = (
            f"<span class='chip'><b>{n_src}</b> source(s) citée(s)</span>"
            f"<span class='chip'><b>{n_docs}</b> document(s)</span>"
            f"<span class='chip'>réponse en <b>{entry['latency']:.1f} s</b></span>"
        )
        if n_src == 0:
            chips += (
                "<span class='chip warn'>Aucune source citée : "
                "vérifiez la réponse dans le document</span>"
            )
        st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)

        if n_src:
            # Création de deux colonnes : le titre à gauche, le bouton à droite
            col_title, col_btn = st.columns([3, 1])
            
            with col_title:
                st.markdown("**Passages sources**")
                
            with col_btn:
                # Le bouton est placé dans la colonne de droite
                st.download_button(
                    "Exporter",
                    to_markdown(entry),
                    file_name=f"marcheiq_{idx + 1}.md",
                    key=f"dl_{idx}",
                )
            
            # Affichage des sources en dessous
            render_sources(entry)

        # Le bouton "feedback" (pouce en l'air/bas) reste tout en bas
        c1, _ = st.columns([1, 5])
        with c1:
            st.feedback("thumbs", key=f"fb_{idx}", on_change=save_feedback, args=(idx,))


def run_query(prompt: str, top_k: int) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)

    start = time.time()
    with st.status("Recherche dans les appels d'offres…", expanded=False) as status:
        try:
            generate = get_generate()
            result = generate(prompt, top_k=top_k)
        except Exception as e:
            status.update(label="La recherche a échoué", state="error")
            st.error(
                "Impossible de générer une réponse. Vérifiez la clé GROQ_API_KEY "
                "et l'index des documents."
            )
            with st.expander("Détail technique"):
                st.code(str(e))
            return
        latency = time.time() - start
        status.update(label=f"Réponse prête en {latency:.1f} s", state="complete")

    entry = {
        "q": prompt,
        "answer": result["answer"],
        "cited": result.get("cited", []),
        "latency": latency,
        "ts": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    st.session_state.history.append(entry)
    render_assistant(len(st.session_state.history) - 1, entry, stream=True)


def render_empty_state() -> None:
    st.markdown(
        "<h1 class='miq-title'>Trouvez l'article, la page, la preuve.</h1>"
        "<p class='miq-sub'>MarchéIQ répond à vos questions sur les appels d'offres "
        "et cite chaque passage du CPS ou du règlement de consultation qui "
        "justifie la réponse.</p>"
        "<p class='miq-sub'>Utilisez les questions d'exemple à gauche, "
        "ou tapez directement votre question en bas.</p>",
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    st.subheader("Comment MarchéIQ construit une réponse")
    st.markdown(
        """
        <div class="pipe">
          <div><strong>1. Ingestion</strong>Les PDF des marchés (CPS, RC, avis) sont lus et nettoyés.</div>
          <div><strong>2. Découpage</strong>Chaque document est coupé en passages, avec son numéro de page.</div>
          <div><strong>3. Indexation</strong>Chaque passage est transformé en vecteur de recherche.</div>
          <div><strong>4. Recherche</strong>Les passages les plus proches de la question sont retenus.</div>
          <div><strong>5. Génération</strong>Le modèle rédige la réponse et cite ses sources.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Fiabilité")
    st.markdown(
        "- Chaque affirmation renvoie à un passage numéroté, surligné avec les mots de votre question.\n"
        "- Sans source citée, l'interface le signale en rouge.\n"
        "- Vérifiez toujours les montants et les dates dans le document d'origine avant de déposer une offre.\n"
        "- Le bouton « Utile » enregistre vos retours dans `logs/feedback.jsonl` pour mesurer la qualité."
    )


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
st.session_state.setdefault("history", [])

with st.sidebar:
    st.markdown("### MarchéIQ")
    top_k = st.slider(
        "Passages analysés", 3, 10, 5,
        help="Plus de passages : réponse plus complète, mais plus lente."
    )

    st.markdown("### Questions d'exemple")
    for tag, question in EXEMPLES:
        if st.button(question, key=f"side_{tag}", use_container_width=True):
            st.session_state["pending"] = question
            st.rerun()

    st.markdown("### Historique")
    history_box = st.container()

    st.divider()
    

# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending", None)
prompt = st.chat_input("Posez une question : caution, dates, pénalités, délais…") or pending

tab_chat, tab_how = st.tabs(["Assistant", "Comment ça marche"])

with tab_chat:
    if not st.session_state.history and not prompt:
        render_empty_state()

    for i, e in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(e["q"])
        render_assistant(i, e)

    if prompt:
        run_query(prompt, top_k)

with tab_how:
    render_how_it_works()

# Historique de la sidebar : dessiné en dernier pour inclure la question du tour
with history_box:
    if st.session_state.history:
        for i, e in enumerate(reversed(st.session_state.history)):
            if st.button(e["q"][:60], key=f"hist_{i}", use_container_width=True):
                st.session_state["pending"] = e["q"]
                st.rerun()
        if st.button("Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Vos questions apparaîtront ici.")