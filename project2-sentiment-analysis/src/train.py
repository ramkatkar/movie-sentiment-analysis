"""
Movie Sentiment Analysis — NLP Pipeline
========================================
Dataset  : IMDB Movie Reviews (via sklearn fetch)
Task     : Binary Classification (positive / negative)
Models   : TF-IDF + Logistic Regression, Naive Bayes, SVM, SGD
Author   : [Ram Katkar]
Date     : 2026
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import joblib, re, string

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ─────────────────────────────────────────────
# SECTION 1: GENERATE / LOAD DATASET
# ─────────────────────────────────────────────

# Curated IMDB-style reviews for a self-contained, reproducible dataset
POSITIVE_REVIEWS = [
    "This movie was absolutely brilliant. The storytelling was masterful and the performances were outstanding.",
    "A cinematic masterpiece. Every scene was crafted with such care and attention to detail.",
    "I was completely captivated from start to finish. One of the best films I've seen in years.",
    "The director brought an extraordinary vision to life. Truly inspiring and emotionally powerful.",
    "Exceptional performances all around. The chemistry between the leads was electric and believable.",
    "A breathtaking film that left me speechless. The cinematography alone is worth watching for.",
    "Incredible storytelling with rich character development. I laughed, I cried, I loved every minute.",
    "The script was sharp, witty, and deeply moving. A rare achievement in modern cinema.",
    "Superb acting and a gripping plot that kept me on the edge of my seat throughout.",
    "One of the most emotionally resonant films I have ever experienced. A true masterwork.",
    "Beautifully shot and expertly directed. The pacing was perfect and the ending was deeply satisfying.",
    "A wonderful film that celebrates the human spirit. Uplifting and thoroughly entertaining.",
    "The performances here are career-best for almost every actor involved. Simply remarkable.",
    "Funny, heartfelt, and surprisingly deep. This film exceeded every expectation I had going in.",
    "A near-perfect film. The balance of humour and drama is handled with exceptional skill.",
    "The soundtrack perfectly complemented the emotional beats of the story. Loved every second.",
    "An instant classic. It reminded me why I love cinema so much. Cannot recommend highly enough.",
    "Riveting from the first frame to the last. The tension builds expertly throughout.",
    "A deeply human story told with honesty and compassion. This film will stay with me for years.",
    "Absolutely loved this. The world-building was immersive and the characters felt completely real.",
    "The special effects were stunning but never overshadowed the heart of the story. Brilliant balance.",
    "A feel-good film that never feels manipulative. Genuinely moving and thoroughly enjoyable.",
    "Fantastic performances backed by a razor-sharp screenplay. A triumph of contemporary filmmaking.",
    "This exceeded all my expectations. Smart, entertaining, and visually gorgeous throughout.",
    "I haven't been this moved by a film in a very long time. A genuine emotional experience.",
    "Outstanding direction and a cast that delivers on every level. One of the year's best films.",
    "A profound exploration of the human condition wrapped in an incredibly entertaining package.",
    "Clever, funny, and touching in equal measure. A film that manages to do everything right.",
    "The best film I've seen all year without question. Engaging, thoughtful, and beautifully made.",
    "Powerful, poignant, and perfectly paced. This film deserves every award it receives.",
    "The dialogue sparkles with intelligence. Every scene felt essential and purposeful.",
    "Wonderfully crafted film with heart, humour and genuine emotional depth. Highly recommended.",
    "The lead performance is a revelation. A career-defining turn that anchors the whole film.",
    "I was completely absorbed from the opening scene. This is filmmaking at its very finest.",
    "Exhilarating and emotionally satisfying. The filmmakers truly understood what they were doing.",
    "A bold and ambitious film that succeeds on almost every level. A genuine achievement.",
    "Smart writing, strong performances, and beautiful direction. This film has it all.",
    "The emotional climax hit me like a freight train. I was not prepared for how good this was.",
    "A love letter to cinema itself. Playful, inventive, and utterly absorbing throughout.",
    "Rarely does a film manage to be both intellectually stimulating and genuinely entertaining.",
    "Visually stunning and emotionally resonant. This film will absolutely stand the test of time.",
    "The ensemble cast is extraordinary. Each performance is perfectly calibrated and memorable.",
    "A joyful, life-affirming experience. Left the cinema feeling genuinely happy and uplifted.",
    "Brilliant work by everyone involved. The film is greater than the sum of its already impressive parts.",
    "A deeply satisfying narrative with twists that genuinely surprised me. Gripping throughout.",
    "The film has a magical quality that is very hard to define but impossible to ignore.",
    "Every technical aspect of this film is polished to perfection. A class act from start to finish.",
    "The world needs more films this honest, this funny, and this genuinely moving. A true gem.",
    "An absolute joy. Funny, warm, and brimming with life. Easily one of the best of the decade.",
    "I cannot say enough good things. Transformative, beautiful, and unforgettable. See it immediately.",
    "The editing was exceptional, keeping the film tight and kinetic. Never a dull moment anywhere.",
    "Extraordinarily well-acted and written. A film that rewards repeat viewings with new discoveries.",
    "The third act is a masterclass in emotional payoff. Everything earned, nothing manipulated.",
    "Genuinely surprised by how much I loved this. Smart, funny, and deeply affecting. Wonderful.",
    "A film that dares to ask big questions while still being thoroughly entertaining. Impressive feat.",
    "The chemistry between the cast members is palpable. Every interaction feels authentic and real.",
    "Simply put, this is extraordinary filmmaking. Confident, assured, and deeply moving throughout.",
    "A triumph of storytelling. The narrative threads come together in a deeply satisfying conclusion.",
    "Visually inventive and emotionally raw. A bold filmmaking statement that absolutely pays off.",
    "This film earns every emotional beat. Nothing feels cheap or unearned. Masterful filmmaking.",
    "Pure cinema magic. A film that reminds you why we tell stories and why they matter so much.",
]

NEGATIVE_REVIEWS = [
    "A complete waste of time. The plot was incoherent and the acting was painfully wooden throughout.",
    "Deeply disappointing. The trailer promised something exciting but the film delivered absolutely nothing.",
    "One of the worst films I have ever seen. The script was amateurish and the direction was flat.",
    "The story made no sense whatsoever. Characters behaved in completely illogical ways throughout.",
    "Boring, predictable, and utterly forgettable. I struggled to stay awake through the second half.",
    "The acting was so bad it became unintentionally funny. Truly terrible in every possible way.",
    "A painfully mediocre film that wastes an excellent cast on a dreadful screenplay.",
    "I cannot believe this got funded. The whole thing felt like an extended rough cut with no polish.",
    "Incredibly slow and pointless. Two hours of my life I will never get back unfortunately.",
    "The special effects were embarrassing for a film with this budget. Looked like a student project.",
    "The dialogue was cringe-worthy and the characters were paper-thin stereotypes throughout.",
    "A hollow, soulless cash grab that cynically exploits a popular franchise. Deeply depressing.",
    "The pacing was absolutely dreadful. The film dragged for what felt like an eternity.",
    "Poorly written, poorly directed, and poorly acted. A perfect storm of cinematic failure.",
    "The plot holes were so enormous you could drive a truck through them. Just awful.",
    "Utterly devoid of any original ideas. A lazy, derivative mess from start to miserable finish.",
    "The attempts at humour fell completely flat. The comedy was forced, obvious, and painfully unfunny.",
    "I fell asleep twice. When I woke up nothing seemed to have happened. Glacially slow film.",
    "The twist ending was telegraphed from the first scene. Insulted the audience's intelligence.",
    "A catastrophic misfire. The director clearly had no idea what kind of film they were making.",
    "Genuinely unwatchable. I left the cinema feeling cheated and deeply dissatisfied.",
    "The film had no sense of purpose or direction. It just meandered until it eventually stopped.",
    "Terrible writing combined with some of the worst performances I have seen this year.",
    "The cinematography was flat and uninspired. Everything looked grey and underlit throughout.",
    "A complete tonal mess. The film could not decide what it wanted to be and suffered greatly.",
    "Crammed with pointless subplots that went nowhere and added nothing to the main story.",
    "The lead actor seemed bored throughout. Their complete lack of energy killed every scene.",
    "A film that mistakes tedium for depth and confusion for complexity. Profoundly irritating.",
    "Simply dreadful. The worst film in what is already a weak franchise. Avoid completely.",
    "The ending was so abrupt it felt like the filmmakers simply ran out of money and stopped.",
    "A cynical exercise in franchise milking with nothing original or creative to offer audiences.",
    "Every cliché in the book is deployed without any self-awareness or intelligence. Painful.",
    "The action sequences were incoherently edited. I had no idea what was happening at any point.",
    "Laughably bad CGI combined with a nonsensical story. A true disaster from start to finish.",
    "The romantic subplot was unconvincing and unnecessary. It added nothing and wasted precious time.",
    "A film so tedious it made me angry. Two hours of my life completely and utterly wasted.",
    "The villain had no motivation whatsoever. Just a bad guy because the plot needed one. Lazy.",
    "Deeply offensive and not in an interesting way. Just mean-spirited garbage dressed as comedy.",
    "The third act completely abandoned everything the first two acts established. A betrayal.",
    "Amateurish in every department. Hard to believe this actually received a theatrical release.",
    "The music choices were bizarre and actively fought against the scenes they accompanied.",
    "An interminable slog that mistakes length for importance. Shamelessly self-indulgent nonsense.",
    "The dialogue was so unnatural it was impossible to believe any of these people were human.",
    "A trainwreck from beginning to end. Nothing worked, nothing landed, nothing was worth watching.",
    "The supposed emotional moments fell completely flat because we cared nothing for the characters.",
    "A film that somehow managed to make an interesting premise profoundly dull and uninvolving.",
    "Worst screenplay I have encountered in years. Every scene felt written by a first-year student.",
    "The comedic timing was catastrophically off throughout. The jokes missed every single time.",
    "An empty spectacle with no substance whatsoever underneath the flashy but meaningless visuals.",
    "I genuinely wanted to leave after thirty minutes but optimistically stayed. A mistake.",
    "The film disrespected its own source material in almost every possible way. Deeply frustrating.",
    "Painfully slow setup that paid off with an ending even more disappointing than the journey.",
    "Just absolutely terrible. No redeeming qualities whatsoever. A complete failure on every level.",
    "The script was clearly written over a weekend and nobody bothered to revise or improve it.",
    "Cheap-looking, cheaply made, and cheaply conceived. Everything about this film was subpar.",
    "The supporting characters were so underdeveloped they might as well have not been present.",
    "A deeply unpleasant experience from start to finish. Grim, joyless, and utterly pointless.",
    "The potential was enormous. The execution was catastrophically poor. A heartbreaking failure.",
    "Nothing worked. The comedy, the drama, the action — all execrably done. Shocking incompetence.",
    "I cannot understand how this received positive reviews. It was genuinely one of the worst films ever.",
    "Derivative, dull, and devoid of any spark of creativity. Cinema deserves so much better than this.",
]


def create_dataset():
    texts  = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
    labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)
    df = pd.DataFrame({"text": texts, "label": labels, 
                       "sentiment": ["positive" if l==1 else "negative" for l in labels]})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/reviews.csv", index=False)
    print(f"✅ Dataset ready: {len(df)} reviews | "
          f"Positive: {sum(labels)} | Negative: {len(labels)-sum(labels)}")
    return df


# ─────────────────────────────────────────────
# SECTION 2: TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text):
    """Basic text normalisation."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)           # strip HTML
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # keep letters only
    text = re.sub(r"\s+", " ", text).strip()      # normalise whitespace
    return text


def explore_data(df):
    """EDA — review lengths, word frequencies, sentiment split."""
    df = df.copy()
    df["clean_text"]    = df["text"].apply(clean_text)
    df["review_length"] = df["clean_text"].apply(lambda x: len(x.split()))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Class distribution
    sentiment_counts = df["sentiment"].value_counts()
    axes[0].pie(sentiment_counts, labels=sentiment_counts.index,
                autopct="%1.1f%%", colors=["#2ECC71","#E74C3C"],
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Sentiment Distribution", fontweight="bold", fontsize=12)

    # 2. Review length by class
    for sentiment, color, label in [("positive","#2ECC71","Positive"), ("negative","#E74C3C","Negative")]:
        lengths = df[df["sentiment"]==sentiment]["review_length"]
        axes[1].hist(lengths, bins=15, alpha=0.7, color=color, label=label)
    axes[1].set_xlabel("Words per Review"); axes[1].set_ylabel("Count")
    axes[1].set_title("Review Length Distribution", fontweight="bold", fontsize=12)
    axes[1].legend()

    # 3. Average review length by class
    avg_lens = df.groupby("sentiment")["review_length"].mean()
    bars = axes[2].bar(avg_lens.index, avg_lens.values,
                       color=["#E74C3C","#2ECC71"], alpha=0.85, edgecolor="white")
    axes[2].set_title("Avg Words per Review by Class", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Average Word Count")
    for bar, val in zip(bars, avg_lens.values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", fontweight="bold")
    plt.suptitle("Sentiment Dataset — Exploratory Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/01_eda.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/01_eda.png")

    print(f"\n📋 Avg review length: {df['review_length'].mean():.0f} words")
    print(f"📋 Max review length: {df['review_length'].max()} words")
    print(f"📋 Min review length: {df['review_length'].min()} words")
    return df


# ─────────────────────────────────────────────
# SECTION 3: BUILD & TRAIN MODELS
# ─────────────────────────────────────────────

def build_nlp_models():
    """Pipelines: vectoriser + classifier."""
    return {
        "Logistic Regression (TF-IDF)" : Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                      max_features=15000, sublinear_tf=True,
                                      stop_words="english")),
            ("clf",   LogisticRegression(C=1.0, max_iter=500, random_state=42))
        ]),
        "Multinomial Naive Bayes (TF-IDF)" : Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                      max_features=15000, sublinear_tf=True,
                                      stop_words="english")),
            ("clf",   MultinomialNB(alpha=0.1))
        ]),
        "Complement Naive Bayes (BoW)" : Pipeline([
            ("bow",  CountVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                     max_features=15000, stop_words="english")),
            ("clf",  ComplementNB(alpha=0.1))
        ]),
        "Linear SVM (TF-IDF)" : Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                      max_features=15000, sublinear_tf=True,
                                      stop_words="english")),
            ("svc",   CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, random_state=42)))
        ]),
        "SGD Classifier (TF-IDF)" : Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                      max_features=15000, sublinear_tf=True,
                                      stop_words="english")),
            ("clf",   SGDClassifier(loss="modified_huber", max_iter=100,
                                    random_state=42, n_jobs=-1))
        ]),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    print("\n─── Training NLP Models ──────────────────────────")
    results = []
    for name, pipeline in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring="f1", n_jobs=-1)
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        res = {
            "Model"        : name,
            "CV F1 (5-fold)": f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
            "Test Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision"     : round(precision_score(y_test, y_pred), 4),
            "Recall"        : round(recall_score(y_test, y_pred), 4),
            "F1 Score"      : round(f1_score(y_test, y_pred), 4),
            "ROC-AUC"       : round(roc_auc_score(y_test, y_proba), 4),
            "_pipeline"     : pipeline,
            "_y_pred"       : y_pred,
            "_y_proba"      : y_proba,
        }
        print(f"  ✅ {name[:45]:<45} | F1={res['F1 Score']:.4f} | AUC={res['ROC-AUC']:.4f}")
        results.append(res)
    return results


# ─────────────────────────────────────────────
# SECTION 4: VISUALISE
# ─────────────────────────────────────────────

def plot_results(results, y_test):
    names  = [r["Model"] for r in results]
    f1s    = [r["F1 Score"] for r in results]
    aucs   = [r["ROC-AUC"] for r in results]
    accs   = [r["Test Accuracy"] for r in results]

    # Metrics bar chart
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, accs, w, label="Accuracy", color="#3498DB", alpha=0.85)
    ax.bar(x,     f1s,  w, label="F1 Score",  color="#E74C3C", alpha=0.85)
    ax.bar(x + w, aucs, w, label="ROC-AUC",   color="#2ECC71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" (")[0] for n in names], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.55, 1.05); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("NLP Model Comparison — Accuracy / F1 / ROC-AUC",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7.5, color="dimgray")
    plt.tight_layout()
    plt.savefig("plots/02_model_comparison.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/02_model_comparison.png")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498DB","#E74C3C","#2ECC71","#9B59B6","#E67E22"]
    for r, c in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{r['Model'].split(' (')[0]} (AUC={r['ROC-AUC']:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random Baseline")
    ax.set_xlabel("FPR", fontsize=11); ax.set_ylabel("TPR", fontsize=11)
    ax.set_title("ROC Curves — NLP Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/03_roc_curves.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/03_roc_curves.png")

    # Best model confusion matrix
    best = max(results, key=lambda r: r["F1 Score"])
    cm = confusion_matrix(y_test, best["_y_pred"])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"],
                linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best['Model'].split(' (')[0]}", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/04_confusion_matrix.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/04_confusion_matrix.png")
    return best


def plot_top_features(best):
    """Top positive and negative TF-IDF words."""
    try:
        vectoriser = best["_pipeline"].named_steps.get("tfidf") or best["_pipeline"].named_steps.get("bow")
        clf = best["_pipeline"].named_steps.get("clf") or best["_pipeline"].named_steps.get("svc")
        if not hasattr(clf, "coef_"):
            return
        feature_names = vectoriser.get_feature_names_out()
        coef = clf.coef_.ravel() if hasattr(clf, "coef_") else np.zeros(len(feature_names))
        top_pos = np.argsort(coef)[-20:][::-1]
        top_neg = np.argsort(coef)[:20]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh([feature_names[i] for i in reversed(top_pos)],
                     [coef[i] for i in reversed(top_pos)], color="#2ECC71", alpha=0.85)
        axes[0].set_title("Top 20 Positive Sentiment Words", fontweight="bold", fontsize=11)
        axes[0].set_xlabel("TF-IDF Coefficient")

        axes[1].barh([feature_names[i] for i in reversed(top_neg)],
                     [coef[i] for i in reversed(top_neg)], color="#E74C3C", alpha=0.85)
        axes[1].set_title("Top 20 Negative Sentiment Words", fontweight="bold", fontsize=11)
        axes[1].set_xlabel("TF-IDF Coefficient")

        plt.suptitle(f"Feature Importances — {best['Model'].split(' (')[0]}",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig("plots/05_top_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  📈 Saved: plots/05_top_features.png")
    except Exception as e:
        print(f"  ℹ️  Skipped feature plot: {e}")


# ─────────────────────────────────────────────
# SECTION 5: INFERENCE DEMO
# ─────────────────────────────────────────────

def predict_sentiment(model, reviews):
    """Demonstrate model on custom reviews."""
    print("\n─── Live Prediction Demo ─────────────────────────")
    for review in reviews:
        pred    = model.predict([review])[0]
        prob    = model.predict_proba([review])[0]
        label   = "✅ POSITIVE" if pred == 1 else "❌ NEGATIVE"
        conf    = prob[pred]
        short   = (review[:80] + "...") if len(review) > 80 else review
        print(f"  Review: \"{short}\"")
        print(f"  Result: {label} (confidence: {conf:.1%})\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  MOVIE SENTIMENT ANALYSIS — NLP PIPELINE")
    print("=" * 58)

    df = create_dataset()
    df = explore_data(df)

    X, y = df["text"].values, df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

    models  = build_nlp_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\n─── Generating Plots ─────────────────────────────")
    best = plot_results(results, y_test)
    plot_top_features(best)

    # Save
    display_cols = ["Model","CV F1 (5-fold)","Test Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
    df_res = pd.DataFrame([{k: r[k] for k in display_cols} for r in results])
    df_res = df_res.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df_res.to_csv("models/results_summary.csv", index=False)

    joblib.dump(best["_pipeline"], "models/best_sentiment_model.pkl")

    print(f"\n─── Results Summary ──────────────────────────────")
    print(df_res.to_string(index=False))
    print(f"\n💾 Best model saved → models/best_sentiment_model.pkl")
    print(f"   Model  : {best['Model']}")
    print(f"   F1     : {best['F1 Score']}")
    print(f"   ROC-AUC: {best['ROC-AUC']}")

    print("\n─── Classification Report ────────────────────────")
    print(classification_report(y_test, best["_y_pred"],
                                target_names=["Negative", "Positive"]))

    # Demo predictions
    demo_reviews = [
        "This was an absolutely magnificent film. The performances were breathtaking and I loved every single moment.",
        "Terrible movie. I wanted to leave after ten minutes. The plot made no sense at all.",
        "A decent film with some strong moments but ultimately fails to deliver on its premise.",
    ]
    predict_sentiment(best["_pipeline"], demo_reviews)

    print("=" * 58)
    print("✅ NLP Pipeline complete. Check /plots and /models.")
    print("=" * 58)
