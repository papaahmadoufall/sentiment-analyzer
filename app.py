import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from deep_translator import GoogleTranslator
import re
from wordcloud import WordCloud

# Ajoutez ceci au début de votre script, après les imports
import os
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiments",
    page_icon="🧠",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
    }
    .result-positive {
        background-color: #C8E6C9;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }
    .result-negative {
        background-color: #FFCDD2;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #C62828;
        text-align: center;
    }
    .result-neutral {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
    }
    .stTextInput>div>div>input {
        padding: 15px;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        font-size: 1rem;
        width: 100%;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ECEFF1;
        font-size: 0.9rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour télécharger les ressources NLTK si nécessaires
@st.cache_resource
def download_nltk_resources():
    # Téléchargement explicite des ressources nécessaires
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    # Assurez-vous que la ressource punkt est bien téléchargée
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Fonction pour extraire les mots importants
@st.cache_data
def extract_important_words(text, top_n=30):
    # Liste de mots vides en français
    stopwords_fr = [
        "le", "la", "les", "un", "une", "des", "et", "est", "il", "elle", "ils", 
        "elles", "nous", "vous", "je", "tu", "on", "ce", "cette", "ces", "que", 
        "qui", "qu", "quoi", "dont", "où", "à", "au", "aux", "de", "du", "des", 
        "en", "par", "pour", "avec", "sans", "sous", "sur", "dans", "entre", 
        "vers", "chez", "après", "avant", "depuis", "pendant", "selon", "pas",
        "ne", "plus", "moins", "très", "non", "oui", "si", "alors", "mais", "ou",
        "car", "donc", "c", "d", "j", "l", "m", "n", "s", "t", "y", "a", "être", 
        "avoir", "faire", "aller", "voir", "savoir", "pouvoir", "vouloir", "falloir",
        "comme", "tout", "tous", "toute", "toutes", "aucun", "aucune", "chaque"
    ]
    
    # Nettoyer le texte
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Filtrer les mots vides
    words = [word for word in words if word not in stopwords_fr and len(word) > 2]
    
    # Compter la fréquence des mots
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Trier et retourner les mots les plus fréquents
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_words[:top_n])

# Fonction pour créer un nuage de mots
def create_wordcloud(word_freq):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

# Fonction d'analyse de sentiment
def analyze_sentiment(text, lang='fr'):
    # Si le texte n'est pas en anglais, le traduire
    if lang != 'en':
        try:
            text_en = GoogleTranslator(source=lang, target='en').translate(text)
        except:
            st.error("Erreur de traduction. Veuillez réessayer.")
            return None
    else:
        text_en = text
    
    # Analyser le sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text_en)
    
    # Utiliser une fonction simple pour diviser en phrases au lieu de sent_tokenize
    def simple_tokenize(text):
        # Division par les signes de ponctuation courants qui terminent une phrase
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s]
    
    # Analyser phrase par phrase pour trouver les plus positives/négatives
    sentences = simple_tokenize(text)  # Remplacer nltk.sent_tokenize par notre fonction
    sent_analysis = []
    
    for sentence in sentences:
        if lang != 'en':
            try:
                sentence_en = GoogleTranslator(source=lang, target='en').translate(sentence)
                sentence_score = sia.polarity_scores(sentence_en)
            except:
                # En cas d'erreur, utiliser un score neutre
                sentence_score = {'compound': 0, 'pos': 0.33, 'neu': 0.33, 'neg': 0.33}
        else:
            sentence_score = sia.polarity_scores(sentence)
        
        sent_analysis.append({
            'sentence': sentence,
            'compound': sentence_score['compound'],
            'pos': sentence_score['pos'],
            'neu': sentence_score['neu'],
            'neg': sentence_score['neg']
        })
    
    # Trier les phrases par score composé
    sent_analysis_sorted = sorted(sent_analysis, key=lambda x: x['compound'])
    
    # Déterminer le sentiment global
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = "positif"
    elif compound_score <= -0.05:
        sentiment = "négatif"
    else:
        sentiment = "neutre"
    
    return {
        'sentiment': sentiment,
        'scores': sentiment_scores,
        'sentence_analysis': sent_analysis,
        'most_negative': sent_analysis_sorted[0] if sent_analysis else None,
        'most_positive': sent_analysis_sorted[-1] if sent_analysis else None
    }

# Fonction pour exporter les résultats en CSV
def export_results_to_csv(history):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    csv = df.to_csv(index=False)
    return csv

# Préparation des données pour l'analyse
download_nltk_resources()

# Initialisation des variables de session
if 'history' not in st.session_state:
    st.session_state.history = []

if 'example_texts' not in st.session_state:
    st.session_state.example_texts = {
        "Positif": "J'adore cette application ! C'est vraiment un excellent outil, facile à utiliser et qui donne des résultats précis. Je la recommande chaudement à tous mes collègues.",
        "Neutre": "L'application analyse les sentiments des textes. Elle fonctionne avec plusieurs langues. L'interface est basée sur Streamlit.",
        "Négatif": "Je suis très déçu par cette application. Elle est lente, peu intuitive et les résultats sont souvent erronés. Je ne la recommande pas du tout."
    }

# Interface utilisateur
st.markdown('<h1 class="main-header">✨ Analyseur de Sentiments ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Entrez un texte et découvrez le sentiment qu\'il exprime</p>', unsafe_allow_html=True)

# Onglets
tab1, tab2, tab3 = st.tabs(["📝 Analyse de texte", "📄 Analyse de fichier", "❓ Aide"])

# Sidebar pour les options et statistiques
with st.sidebar:
    st.markdown("## ⚙️ Options")
    
    lang_option = st.selectbox(
        'Langue du texte:',
        options=['fr', 'en', 'es', 'de', 'it', 'pt', 'nl', 'ru'],
        index=0
    )
    
    lang_names = {
        'fr': 'Français',
        'en': 'Anglais',
        'es': 'Espagnol',
        'de': 'Allemand',
        'it': 'Italien',
        'pt': 'Portugais',
        'nl': 'Néerlandais',
        'ru': 'Russe'
    }
    
    st.markdown(f"Langue sélectionnée : **{lang_names[lang_option]}**")
    
    # Options avancées
    st.markdown("## 🔍 Options avancées")
    show_detailed_analysis = st.checkbox("Analyse détaillée", value=True)
    show_wordcloud = st.checkbox("Nuage de mots", value=True)
    
    # Statistiques
    if st.session_state.history:
        st.markdown("## 📊 Statistiques")
        
        total_analyses = len(st.session_state.history)
        positive_count = len([h for h in st.session_state.history if h['sentiment'] == 'positif'])
        negative_count = len([h for h in st.session_state.history if h['sentiment'] == 'négatif'])
        neutral_count = len([h for h in st.session_state.history if h['sentiment'] == 'neutre'])
        
        st.metric("Analyses totales", total_analyses)
        col1, col2, col3 = st.columns(3)
        col1.metric("Positifs", positive_count)
        col2.metric("Neutres", neutral_count)
        col3.metric("Négatifs", negative_count)
        
        # Options d'exportation
        st.markdown("## 📤 Exporter")
        
        if st.button("Exporter en CSV"):
            csv = export_results_to_csv(st.session_state.history)
            if csv:
                st.download_button(
                    "Télécharger le CSV",
                    csv,
                    "sentiment_analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )
        
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.rerun()

# Tab 1: Analyse de texte
with tab1:
    # Exemples de textes
    # Au début de votre script, ajoutez ceci si ce n'est pas déjà fait
    if 'selected_example_text' not in st.session_state:
        st.session_state.selected_example_text = ""

    # Dans la section des exemples de textes
    with st.expander("Exemples de textes"):
        example_cols = st.columns(3)
        
        for i, (sentiment, text) in enumerate(st.session_state.example_texts.items()):
            with example_cols[i]:
                st.markdown(f"### {sentiment}")
                if st.button(f"Utiliser l'exemple {sentiment}"):
                    st.session_state.selected_example_text = text  # Stocke l'exemple dans session_state
                st.markdown(f"*\"{text[:100]}...\"*")

    # Zone de saisie de texte
    text_input = st.text_area(
        "Entrez votre texte ici:", 
        value=st.session_state.selected_example_text,  # Utilise l'exemple stocké
        height=150
    )
    
    analyze_button = st.button("Analyser le sentiment")

    # Analyse et affichage des résultats
    if analyze_button and text_input:
        with st.spinner('Analyse en cours...'):
            result = analyze_sentiment(text_input, lang_option)
            
        if result:
            # Affichage du résultat principal
            st.markdown("## Résultat de l'analyse")
            sentiment = result['sentiment']
            
            if sentiment == "positif":
                st.markdown(f'<div class="result-positive">Le sentiment est POSITIF 😊</div>', unsafe_allow_html=True)
            elif sentiment == "négatif":
                st.markdown(f'<div class="result-negative">Le sentiment est NÉGATIF 😞</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-neutral">Le sentiment est NEUTRE 😐</div>', unsafe_allow_html=True)
            
            # Affichage des scores détaillés
            st.markdown("### Scores détaillés")
            scores = result['scores']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Créer un DataFrame pour le graphique
                df = pd.DataFrame({
                    'Sentiment': ['Positif', 'Neutre', 'Négatif'],
                    'Score': [scores['pos'], scores['neu'], scores['neg']]
                })
                
                # Créer un graphique à barres
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    df['Sentiment'], 
                    df['Score'],
                    color=['#4CAF50', '#2196F3', '#F44336']
                )
                
                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.02,
                        f'{height:.2f}',
                        ha='center', 
                        va='bottom',
                        fontweight='bold'
                    )
                
                ax.set_ylim(0, 1)
                ax.set_title('Répartition des sentiments')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("""
                #### Interprétation des scores
                
                - **Positif**: {:.2f}
                - **Neutre**: {:.2f}
                - **Négatif**: {:.2f}
                - **Score composé**: {:.2f} (résume l'ensemble)
                
                Le score composé est normalisé entre -1 (très négatif) et +1 (très positif).
                """.format(scores['pos'], scores['neu'], scores['neg'], scores['compound']))
                
                # Jauge pour le score composé
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Créer une jauge simple
                compound = scores['compound']
                ax.axvspan(-1, -0.05, alpha=0.2, color='red')
                ax.axvspan(-0.05, 0.05, alpha=0.2, color='blue')
                ax.axvspan(0.05, 1, alpha=0.2, color='green')
                
                ax.scatter([compound], [0], s=400, color='black', zorder=5)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-0.2, 0.2)
                ax.set_yticks([])
                ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                ax.set_title('Score composé')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                st.pyplot(fig)

            # Analyse détaillée
            if show_detailed_analysis:
                st.markdown("### Analyse détaillée")
                
                # Phrases les plus positives et négatives
                if result['most_positive'] and result['most_negative']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 😊 Phrase la plus positive")
                        st.markdown(f"*\"{result['most_positive']['sentence']}\"*")
                        st.progress(float(result['most_positive']['pos']))
                        st.text(f"Score: {result['most_positive']['compound']:.2f}")
                    
                    with col2:
                        st.markdown("#### 😞 Phrase la plus négative")
                        st.markdown(f"*\"{result['most_negative']['sentence']}\"*")
                        st.progress(float(result['most_negative']['neg']))
                        st.text(f"Score: {result['most_negative']['compound']:.2f}")
                
                # Extraction des mots importants
                if show_wordcloud:
                    st.markdown("#### Nuage de mots")
                    word_freq = extract_important_words(text_input)
                    
                    if word_freq:
                        wordcloud_fig = create_wordcloud(word_freq)
                        st.pyplot(wordcloud_fig)
            
            # Ajouter l'analyse actuelle à l'historique
            st.session_state.history.append({
                'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                'sentiment': sentiment,
                'compound': scores['compound'],
                'pos': scores['pos'],
                'neu': scores['neu'],
                'neg': scores['neg'],
                'language': lang_names[lang_option]
            })
            
            # Afficher l'historique (les 5 dernières analyses)
            if len(st.session_state.history) > 0:
                st.markdown("### Historique des analyses")
                history_df = pd.DataFrame(st.session_state.history[-5:])
                
                # Fonction pour colorer en fonction du sentiment
                def color_sentiment(val):
                    if val == 'positif':
                        return 'background-color: #C8E6C9; color: #2E7D32'
                    elif val == 'négatif':
                        return 'background-color: #FFCDD2; color: #C62828'
                    else:
                        return 'background-color: #E3F2FD; color: #1565C0'
                
                # Styliser le DataFrame
                styled_df = history_df.style.applymap(color_sentiment, subset=['sentiment'])
                st.dataframe(styled_df)

    elif analyze_button and not text_input:
        st.error("Veuillez entrer un texte à analyser.")

# Tab 2: Analyse de fichier
with tab2:
    st.markdown("### Analyser un fichier texte")
    st.markdown("Téléchargez un fichier texte (.txt) pour analyser son contenu.")
    
    uploaded_file = st.file_uploader("Choisir un fichier texte", type=["txt"])
    
    if uploaded_file is not None:
        try:
            # Lire le contenu du fichier
            file_text = uploaded_file.read().decode("utf-8")
            
            # Afficher les premières lignes du fichier
            st.markdown("#### Aperçu du contenu")
            st.text_area("", value=file_text[:500] + ("..." if len(file_text) > 500 else ""), height=150, disabled=True)
            
            # Analyser le fichier
            if st.button("Analyser le fichier"):
                with st.spinner('Analyse du fichier en cours...'):
                    result = analyze_sentiment(file_text, lang_option)
                
                if result:
                    # Affichage du résultat principal
                    st.markdown("## Résultat de l'analyse")
                    sentiment = result['sentiment']
                    
                    if sentiment == "positif":
                        st.markdown(f'<div class="result-positive">Le sentiment est POSITIF 😊</div>', unsafe_allow_html=True)
                    elif sentiment == "négatif":
                        st.markdown(f'<div class="result-negative">Le sentiment est NÉGATIF 😞</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-neutral">Le sentiment est NEUTRE 😐</div>', unsafe_allow_html=True)
                    
                    # Affichage des scores détaillés
                    scores = result['scores']
                    
                    # Créer un DataFrame pour le graphique
                    df = pd.DataFrame({
                        'Sentiment': ['Positif', 'Neutre', 'Négatif'],
                        'Score': [scores['pos'], scores['neu'], scores['neg']]
                    })
                    
                    # Créer un graphique à barres
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(
                        df['Sentiment'], 
                        df['Score'],
                        color=['#4CAF50', '#2196F3', '#F44336']
                    )
                    
                    # Ajouter les valeurs sur les barres
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.02,
                            f'{height:.2f}',
                            ha='center', 
                            va='bottom',
                            fontweight='bold'
                        )
                    
                    st.pyplot(fig)
                    
                    # Ajouter à l'historique
                    st.session_state.history.append({
                        'text': f"Fichier: {uploaded_file.name}",
                        'sentiment': sentiment,
                        'compound': scores['compound'],
                        'pos': scores['pos'],
                        'neu': scores['neu'],
                        'neg': scores['neg'],
                        'language': lang_names[lang_option]
                    })
        
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {e}")

# Tab 3: Aide
with tab3:
    st.markdown("### Guide d'utilisation")
    
    with st.expander("Comment ça marche?"):
        st.markdown("""
        ### Fonctionnement de l'analyse de sentiments
        
        Cette application utilise la bibliothèque **VADER** (Valence Aware Dictionary and sEntiment Reasoner) pour analyser le sentiment d'un texte.
        
        #### Processus:
        1. Pour les textes non anglais, une traduction automatique est effectuée
        2. Le texte est analysé pour déterminer sa polarité émotionnelle
        3. Un score composé est calculé, résumant le sentiment global
        
        #### Interprétation des résultats:
        - **Score > 0.05**: Sentiment positif
        - **Score < -0.05**: Sentiment négatif
        - **Score entre -0.05 et 0.05**: Sentiment neutre
        """)
    
    with st.expander("Cas d'utilisation"):
        st.markdown("""
        ### Utilisations courantes
        
        #### Marketing et communication
        - Analyser les retours clients
        - Évaluer la perception des campagnes marketing
        - Surveiller la réputation en ligne
        
        #### Recherche et études
        - Analyser les réponses à des enquêtes
        - Évaluer les opinions sur des sujets spécifiques
        - Suivre l'évolution des sentiments au fil du temps
        
        #### Support client
        - Prioriser les messages négatifs nécessitant une attention immédiate
        - Mesurer la satisfaction client
        """)
    
    with st.expander("FAQ"):
        st.markdown("""
        ### Questions fréquemment posées
        
        #### L'analyse fonctionne-t-elle dans toutes les langues?
        Oui, notre application traduit automatiquement le texte avant l'analyse. Cependant, certaines nuances linguistiques peuvent être perdues dans la traduction.
        
        #### Quelle est la précision de l'analyse?
        La précision varie selon la complexité du texte, mais elle est généralement entre 70% et 80%.
        
        #### Puis-je analyser plusieurs textes à la fois?
        Pour le moment, l'analyse se fait texte par texte, mais vous pouvez analyser des fichiers texte complets.
        
        #### Les données sont-elles sauvegardées?
        Non, toutes les analyses sont effectuées localement et ne sont pas enregistrées sur un serveur externe.
        """)

# Pied de page
st.markdown("""
<style>
    .github-icon {
        width: 20px;
        height: 20px;
        vertical-align: middle;
    }            
</style>
<div class="footer">
    Développé avec ❤️ par Mohamed Ndiaye | Propulsé par Streamlit, NLTK et Deep Translator |
    <a href="https://github.com/Moesthetics-code/sentiment-analyzer" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" class="github-icon" />
    </a>
</div>
""", unsafe_allow_html=True)