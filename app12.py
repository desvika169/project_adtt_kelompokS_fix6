import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Sentimen BUMI - Stockbit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, .main { background-color: #f5f0e8; }
[data-testid="stAppViewContainer"] { background-color: #f5f0e8; }
[data-testid="stSidebar"] { background-color: #ede5d8; border-right: 1px solid #d4c5b0; }
[data-testid="metric-container"] {
    background: #ede5d8;
    border: 1px solid #c8b89a;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #7a6650 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #3d2b1f !important;
    font-size: 1.75rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
h1, h2, h3, h4 { color: #3d2b1f !important; }
p, li, label, .stMarkdown { color: #4a3728 !important; }
.section-title {
    color: #8b5e3c;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-left: 4px solid #8b5e3c;
    padding-left: 10px;
    margin: 24px 0 14px 0;
}
.insight-box {
    background: #ede5d8;
    border: 1px solid #c8b89a;
    border-left: 4px solid #8b5e3c;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #3d2b1f;
    font-size: 0.9rem;
    line-height: 1.7;
}
.tag-pos { background:#d4e8d4; color:#3a6b3a; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tag-neg { background:#f0d8d0; color:#8b3a2a; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tag-neu { background:#e8e0cc; color:#6b5a30; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { background-color: #ede5d8; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #7a6650 !important; }
.stTabs [aria-selected="true"] { color: #8b5e3c !important; border-bottom: 2px solid #8b5e3c !important; }
.stButton>button { background:#8b5e3c; color:#f5f0e8; border:none; border-radius:8px; font-weight:600; }
.stButton>button:hover { background:#6b4428; color:#f5f0e8; }
hr { border-color: #c8b89a !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── COLOR PALETTE ────────────────────────────────────────────────────────────
BG    = "#ede5d8"
TEXT  = "#3d2b1f"
BLUE  = "#6b8f71"   # sage green
RED   = "#a0522d"   # sienna
GREEN = "#6b8c5a"   # olive green
TEAL  = "#8b5e3c"   # brown
GOLD  = "#c8963c"   # amber
GRID  = "#c8b89a"

# ── PLOT BASE ────────────────────────────────────────────────────────────────
def new_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return fig, ax

def sh(label):
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

# ── LOAD DATA — embedded langsung dari file CSV di repo ──────────────────────
@st.cache_data
def load():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data.csv")
    df = pd.read_csv(path)
    df["tanggal_dt"] = pd.to_datetime(df["tanggal"], errors="coerce", utc=True)
    return df

df = load()

# ── STATISTIK DASAR ───────────────────────────────────────────────────────────
lc    = df["label"].value_counts()
pos   = int(lc.get(1, 0))
neg   = int(lc.get(0, 0))
neu   = int(lc.get(2, 0))
total = len(df)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# Dashboard Analisis Sentimen Saham BUMI")
st.markdown(
    "Analisis komentar investor saham **BUMI** di platform **Stockbit** "
    f"— periode **{df['tanggal_dt'].min().strftime('%d %b %Y')}** s/d "
    f"**{df['tanggal_dt'].max().strftime('%d %b %Y')}** "
    f"| Total **{total:,}** komentar"
)
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ringkasan", "Preprocessing", "EDA", "Sentimen & Insight", "Model Naive Bayes"
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — RINGKASAN
# ════════════════════════════════════════════════════════════════════
with tab1:
    sh("Metrik Utama")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Komentar",  f"{total:,}")
    c2.metric("Positif",  f"{pos:,}",  f"{pos/total*100:.1f}%")
    c3.metric("Negatif",  f"{neg:,}",  f"{neg/total*100:.1f}%")
    c4.metric("Netral",   f"{neu:,}",  f"{neu/total*100:.1f}%")
    c5.metric("Periode Data", f"{(df['tanggal_dt'].max()-df['tanggal_dt'].min()).days + 1} Hari")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        sh("Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        sizes  = [neg, pos, neu]
        labels = ["Negatif", "Positif", "Netral"]
        colors = [RED, BLUE, GOLD]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
        for t in texts + autotexts: t.set_color(TEXT)
        ax.set_title("Distribusi Sentimen Keseluruhan", color=TEXT)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        sh("Tren Komentar Harian")
        tren = df.set_index("tanggal_dt").resample("D")["label"].count().fillna(0)
        fig, ax = new_fig(8, 4)
        ax.plot(tren.index, tren.values, color=TEAL, linewidth=2, marker="o", markersize=4)
        ax.fill_between(tren.index, tren.values, alpha=0.15, color=TEAL)
        if len(tren) > 0:
            pk = tren.idxmax()
            ax.annotate(f"Puncak: {int(tren.max())}",
                        xy=(pk, tren.max()), xytext=(pk, tren.max()*1.18),
                        ha="center", fontsize=9, color=GOLD,
                        arrowprops=dict(arrowstyle="->", color=GOLD))
        ax.set_title("Tren Komentar BUMI per Hari")
        ax.set_xlabel("Tanggal"); ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout()
        st.pyplot(fig)

    sh("WordCloud Seluruh Komentar")
    full_text = " ".join(df["stem_text"].dropna().astype(str))
    wc = WordCloud(width=1200, height=400, background_color=BG,
                   colormap="cool", max_words=200).generate(full_text)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("WordCloud Komentar BUMI — Stockbit", color=TEXT)
    plt.tight_layout()
    st.pyplot(fig)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════
with tab2:
    sh("Perbandingan Data Sebelum dan Sesudah Preprocessing")

    insight(
        "<b>Pipeline Preprocessing yang digunakan:</b><br>"
        "1. <b>Case Folding</b> — semua huruf diubah menjadi huruf kecil<br>"
        "2. <b>Cleaning</b> — menghapus angka, tanda baca, emoji, dan karakter berulang<br>"
        "3. <b>Tokenisasi</b> — memecah kalimat menjadi token kata<br>"
        "4. <b>Normalisasi Slang</b> — mengganti kata tidak baku (gak→tidak, holdd→hold, dll)<br>"
        "5. <b>Stopword Removal</b> — menghapus kata-kata yang tidak bermakna<br>"
        "6. <b>Negation Handling</b> — menggabungkan kata negasi dengan kata setelahnya (tidak_bagus)<br>"
        "7. <b>Stemming (Sastrawi)</b> — mengubah kata ke bentuk dasar"
    )

    st.divider()

    # Tabel perbandingan sampel
    sample = df[["komentar","clean_text","stem_text"]].head(10).copy()
    sample.columns = ["Sebelum Preprocessing (Komentar Asli)", "Setelah Cleaning", "Setelah Stemming (Final)"]
    st.dataframe(sample, use_container_width=True, height=380)

    st.divider()

    # Statistik sebelum vs sesudah
    sh("Statistik Panjang Teks: Sebelum vs Sesudah")
    df["panjang_sebelum"] = df["komentar"].astype(str).apply(len)
    df["panjang_sesudah"] = df["stem_text"].astype(str).apply(len)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = new_fig(8, 4)
        ax.hist(df["panjang_sebelum"], bins=30, color=RED, alpha=0.7, label="Sebelum", edgecolor=BG)
        ax.hist(df["panjang_sesudah"], bins=30, color=BLUE, alpha=0.7, label="Sesudah", edgecolor=BG)
        ax.set_title("Distribusi Panjang Teks: Sebelum vs Sesudah")
        ax.set_xlabel("Panjang Karakter"); ax.set_ylabel("Jumlah Komentar")
        ax.legend(facecolor=BG, labelcolor=TEXT)
        plt.tight_layout(); st.pyplot(fig)

    with c2:
        stat = pd.DataFrame({
            "Statistik"    : ["Rata-rata", "Median", "Minimum", "Maksimum"],
            "Sebelum (char)": [
                f"{df['panjang_sebelum'].mean():.1f}",
                f"{df['panjang_sebelum'].median():.1f}",
                f"{df['panjang_sebelum'].min()}",
                f"{df['panjang_sebelum'].max()}",
            ],
            "Sesudah (char)": [
                f"{df['panjang_sesudah'].mean():.1f}",
                f"{df['panjang_sesudah'].median():.1f}",
                f"{df['panjang_sesudah'].min()}",
                f"{df['panjang_sesudah'].max()}",
            ],
        }).set_index("Statistik")
        st.dataframe(stat, use_container_width=True)

        reduksi = (1 - df["panjang_sesudah"].mean() / df["panjang_sebelum"].mean()) * 100
        insight(
            f"Rata-rata panjang teks berkurang <b>{reduksi:.1f}%</b> setelah preprocessing. "
            f"Ini menunjukkan bahwa sebagian besar karakter yang dihapus adalah noise seperti angka, "
            f"tanda baca, emoji, dan stopword yang tidak berkontribusi pada makna komentar."
        )

    sh("Jumlah Token Sebelum vs Sesudah Stopword Removal")
    df["token_sebelum"] = df["komentar"].astype(str).apply(lambda x: len(x.split()))
    df["token_sesudah"] = df["token_count"]

    fig, ax = new_fig(10, 4)
    x = np.arange(2)
    means = [df["token_sebelum"].mean(), df["token_sesudah"].mean()]
    bars = ax.bar(["Sebelum Stopword Removal", "Sesudah Stopword Removal"],
                  means, color=[RED, BLUE], width=0.4)
    ax.bar_label(bars, fmt="%.1f", color=TEXT, fontsize=11, padding=4)
    ax.set_title("Rata-rata Jumlah Token per Komentar")
    ax.set_ylabel("Jumlah Token")
    plt.tight_layout(); st.pyplot(fig)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ════════════════════════════════════════════════════════════════════
with tab3:
    sh("Distribusi Panjang Dokumen dan Jumlah Token")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = new_fig(8, 4)
        ax.hist(df["length"].dropna(), bins=30, color=BLUE, alpha=0.85, edgecolor=BG)
        ax.set_title("Distribusi Panjang Dokumen (Karakter)")
        ax.set_xlabel("Panjang Karakter"); ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout(); st.pyplot(fig)
    with c2:
        fig, ax = new_fig(8, 4)
        ax.hist(df["token_count"].dropna(), bins=30, color=RED, alpha=0.85, edgecolor=BG)
        ax.set_title("Distribusi Jumlah Token per Komentar")
        ax.set_xlabel("Jumlah Token"); ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout(); st.pyplot(fig)

    sh("Distribusi Likes dan Replies")
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = new_fig(8, 4)
        ax.hist(df["likes"].dropna(), bins=30, color=TEAL, alpha=0.85, edgecolor=BG)
        ax.set_title("Distribusi Likes Komentar")
        ax.set_xlabel("Jumlah Likes"); ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout(); st.pyplot(fig)
    with c4:
        fig, ax = new_fig(8, 4)
        ax.hist(df["replies"].dropna(), bins=30, color=GOLD, alpha=0.85, edgecolor=BG)
        ax.set_title("Distribusi Replies Komentar")
        ax.set_xlabel("Jumlah Replies"); ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout(); st.pyplot(fig)

    insight(
        f"Mayoritas komentar memiliki panjang di bawah <b>{int(df['length'].median())} karakter</b> "
        f"dengan rata-rata <b>{df['token_count'].mean():.1f} token</b> per komentar. "
        f"Komentar dengan likes tertinggi: <b>{int(df['likes'].max())}</b>. "
        f"Ini menunjukkan komentar di Stockbit umumnya bersifat singkat dan padat."
    )

    sh("Top 10 User Paling Aktif")
    top_u = df["username"].value_counts().head(10)
    fig, ax = new_fig(10, 4)
    bars = ax.barh(list(top_u.index), list(top_u.values), color=BLUE)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=9, padding=4)
    ax.set_title("Top 10 User Paling Aktif")
    ax.set_xlabel("Jumlah Komentar")
    plt.tight_layout(); st.pyplot(fig)

    sh("10 Kata Paling Sering Muncul")
    words  = " ".join(df["stem_text"].dropna().astype(str)).lower().split()
    common = dict(Counter(words).most_common(10))
    fig, ax = new_fig(10, 4)
    bars = ax.bar(list(common.keys()), list(common.values()), color=BLUE)
    ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=9, padding=4)
    ax.set_xticklabels(list(common.keys()), rotation=45, ha="right")
    ax.set_title("10 Kata Paling Sering Muncul (Setelah Preprocessing)")
    ax.set_ylabel("Frekuensi")
    plt.tight_layout(); st.pyplot(fig)

    sh("Top 10 Bigram")
    try:
        vec = CountVectorizer(ngram_range=(2,2), max_features=1000)
        vec.fit(df["stem_text"].dropna())
        bag = vec.transform(df["stem_text"].dropna())
        sw  = bag.sum(axis=0)
        freq = sorted([(w, sw[0, idx]) for w, idx in vec.vocabulary_.items()],
                      key=lambda x: x[1], reverse=True)[:10]
        bg_df = pd.DataFrame(freq, columns=["Bigram","Frekuensi"])
        fig, ax = new_fig(10, 4)
        bars = ax.barh(bg_df["Bigram"].tolist(), bg_df["Frekuensi"].tolist(), color=GREEN)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=9, padding=4)
        ax.set_title("Top 10 Bigram Keseluruhan")
        ax.set_xlabel("Frekuensi")
        plt.tight_layout(); st.pyplot(fig)
    except Exception:
        st.warning("Tidak cukup data untuk bigram.")

    sh("Statistik Deskriptif")
    st.dataframe(df[["length","token_count","likes","replies"]].describe().round(2).rename(columns={
        "length":"Panjang Dokumen","token_count":"Jumlah Token",
        "likes":"Likes","replies":"Replies"
    }), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMEN & INSIGHT
# ════════════════════════════════════════════════════════════════════
with tab4:
    sh("Distribusi Label Sentimen (Leksikon InSet)")
    c1, c2 = st.columns([1,2])
    with c1:
        label_map = {0:"Negatif", 1:"Positif", 2:"Netral"}
        lc2 = df["label"].value_counts().sort_index()
        lc2.index = [label_map[i] for i in lc2.index]
        tbl = pd.DataFrame({
            "Sentimen": lc2.index,
            "Jumlah"  : lc2.values,
            "Persentase": [f"{v/total*100:.1f}%" for v in lc2.values]
        }).set_index("Sentimen")
        st.dataframe(tbl, use_container_width=True)
    with c2:
        fig, ax = new_fig(8, 4)
        colors_bar = [RED if l=="Negatif" else BLUE if l=="Positif" else GOLD for l in lc2.index]
        bars = ax.bar(lc2.index, lc2.values, color=colors_bar)
        ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=10, padding=4)
        ax.set_title("Distribusi Label Sentimen")
        ax.set_ylabel("Jumlah Komentar")
        plt.tight_layout(); st.pyplot(fig)

    insight(
        f"Dari <b>{total:,}</b> komentar BUMI di Stockbit: "
        f"<span class='tag-pos'>Positif {pos:,} ({pos/total*100:.1f}%)</span>&nbsp; "
        f"<span class='tag-neg'>Negatif {neg:,} ({neg/total*100:.1f}%)</span>&nbsp; "
        f"<span class='tag-neu'>Netral {neu:,} ({neu/total*100:.1f}%)</span>. "
        f"Sentimen <b>{'positif' if pos > neg else 'negatif'}</b> mendominasi diskusi BUMI "
        f"di platform Stockbit pada periode pengamatan."
    )

    sh("Distribusi Skor Sentimen")
    fig, ax = new_fig(10, 4)
    ax.hist(df["sentiment_score"].dropna(), bins=40, color=TEAL, alpha=0.85, edgecolor=BG)
    ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Netral (skor=0)")
    ax.set_title("Distribusi Skor Sentimen (InSet Lexicon)")
    ax.set_xlabel("Skor Sentimen"); ax.set_ylabel("Jumlah Komentar")
    ax.legend(facecolor=BG, labelcolor=TEXT)
    plt.tight_layout(); st.pyplot(fig)

    sh("WordCloud per Sentimen")
    wc1, wc2, wc3 = st.columns(3)
    for col, lbl, name, cmap in [(wc1,1,"Positif","Greens"),(wc2,0,"Negatif","Reds"),(wc3,2,"Netral","YlOrBr")]:
        subset = df[df["label"]==lbl]["stem_text"].dropna()
        txt    = " ".join(subset.astype(str))
        if not txt.strip(): txt = "kosong"
        wc = WordCloud(width=600, height=300, background_color=BG,
                       colormap=cmap, max_words=100).generate(txt)
        figw, axw = plt.subplots(figsize=(6, 3))
        figw.patch.set_facecolor(BG); axw.set_facecolor(BG)
        axw.imshow(wc, interpolation="bilinear"); axw.axis("off")
        axw.set_title(f"WordCloud — {name}", color=TEXT)
        plt.tight_layout()
        with col:
            st.markdown(f"**{name}** ({len(subset):,} komentar)")
            st.pyplot(figw)

    sh("Bigram per Sentimen")
    for lbl, name, color in [(1,"Positif",BLUE),(0,"Negatif",RED),(2,"Netral",GOLD)]:
        corpus = df[df["label"]==lbl]["stem_text"].dropna()
        if len(corpus) < 5: continue
        try:
            vec2 = CountVectorizer(ngram_range=(2,2), max_features=500)
            vec2.fit(corpus)
            bag2 = vec2.transform(corpus)
            sw2  = bag2.sum(axis=0)
            freq2 = sorted([(w, sw2[0,idx]) for w,idx in vec2.vocabulary_.items()],
                           key=lambda x: x[1], reverse=True)[:8]
            bg2 = pd.DataFrame(freq2, columns=["Bigram","Frekuensi"])
            fig, ax = new_fig(10, 3.5)
            bars = ax.barh(bg2["Bigram"].tolist(), bg2["Frekuensi"].tolist(), color=color)
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=9, padding=4)
            ax.set_title(f"Top 8 Bigram — {name}")
            ax.set_xlabel("Frekuensi")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            pass

    sh("Insight Utama")
    top_pos = df[df["label"]==1]["stem_text"].dropna()
    top_neg = df[df["label"]==0]["stem_text"].dropna()
    pos_words = Counter(" ".join(top_pos).split()).most_common(5)
    neg_words = Counter(" ".join(top_neg).split()).most_common(5)
    pos_str   = ", ".join([f"<b>{w}</b> ({c}x)" for w,c in pos_words])
    neg_str   = ", ".join([f"<b>{w}</b> ({c}x)" for w,c in neg_words])
    avg_likes_pos = df[df["label"]==1]["likes"].mean()
    avg_likes_neg = df[df["label"]==0]["likes"].mean()

    insight(
        f"<b>Kata dominan komentar Positif:</b> {pos_str}<br>"
        f"<b>Kata dominan komentar Negatif:</b> {neg_str}<br><br>"
        f"Rata-rata likes komentar Positif: <b>{avg_likes_pos:.1f}</b> | "
        f"Rata-rata likes komentar Negatif: <b>{avg_likes_neg:.1f}</b><br><br>"
        f"Komentar dengan sentimen <b>{'positif' if avg_likes_pos > avg_likes_neg else 'negatif'}</b> "
        f"cenderung mendapat lebih banyak respons dari komunitas Stockbit."
    )

    sh("Contoh Komentar per Sentimen")
    for lbl, name in [(1,"Positif"),(0,"Negatif"),(2,"Netral")]:
        with st.expander(f"{name} — 5 Contoh"):
            ex = df[df["label"]==lbl][["komentar","sentiment_score","likes"]].head(5)
            st.dataframe(ex.rename(columns={
                "komentar":"Komentar","sentiment_score":"Skor Sentimen","likes":"Likes"
            }), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 5 — NAIVE BAYES
# ════════════════════════════════════════════════════════════════════
with tab5:
    sh("Klasifikasi Naive Bayes — Positif vs Negatif")

    insight(
        "<b>Metode:</b> Multinomial Naive Bayes dengan representasi TF-IDF.<br>"
        "Label Netral (2) dikeluarkan dari training karena model dirancang untuk klasifikasi biner "
        "(Positif vs Negatif). Label berasal dari skoring Leksikon InSet."
    )

    df_m = df[df["label"] != 2].copy()
    df_m["label"] = df_m["label"].astype(int)

    col_s, col_f = st.columns(2)
    test_size = col_s.slider("Ukuran Test Set (%)", 10, 40, 20, 5) / 100
    max_feat  = col_f.slider("Max Features TF-IDF", 100, 2000, 1000, 100)

    if st.button("Jalankan Model Naive Bayes", use_container_width=True, type="primary"):
        with st.spinner("Training model..."):
            X_tr, X_te, y_tr, y_te = train_test_split(
                df_m["stem_text"], df_m["label"],
                test_size=test_size, random_state=42, stratify=df_m["label"])
            tfidf = TfidfVectorizer(max_features=max_feat)
            Xtr_v = tfidf.fit_transform(X_tr)
            Xte_v = tfidf.transform(X_te)
            nb    = MultinomialNB()
            nb.fit(Xtr_v, y_tr)
            y_pred = nb.predict(Xte_v)
            acc = accuracy_score(y_te, y_pred)
            cm  = confusion_matrix(y_te, y_pred)
            cr  = classification_report(y_te, y_pred,
                    target_names=["Negatif","Positif"], output_dict=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",          f"{acc*100:.2f}%")
        m2.metric("Data Training",      f"{len(X_tr):,}")
        m3.metric("Data Testing",       f"{len(X_te):,}")
        m4.metric("TF-IDF Features",    f"{Xtr_v.shape[1]:,}")

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            sh("Confusion Matrix")
            fig, ax = new_fig(6, 4)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Negatif","Positif"],
                        yticklabels=["Negatif","Positif"],
                        ax=ax, linewidths=0.5, linecolor=GRID,
                        annot_kws={"size":14,"color":"white"})
            ax.set_ylabel("Aktual"); ax.set_xlabel("Prediksi")
            ax.set_title("Confusion Matrix")
            plt.tight_layout(); st.pyplot(fig)

        with cb:
            sh("Classification Report")
            cr_df = pd.DataFrame(cr).T.drop("accuracy", errors="ignore")
            st.dataframe(cr_df.round(3), use_container_width=True)

        sh("Distribusi Hasil Prediksi")
        pc = pd.Series(y_pred).value_counts()
        pc.index = ["Negatif" if i==0 else "Positif" for i in pc.index]
        fig, ax = new_fig(6, 3.5)
        bars = ax.bar(pc.index, pc.values, color=[RED, BLUE], width=0.4)
        ax.bar_label(bars, fmt="%d", color=TEXT, fontsize=11, padding=4)
        ax.set_title("Distribusi Hasil Prediksi")
        ax.set_ylabel("Jumlah")
        plt.tight_layout(); st.pyplot(fig)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
        insight(
            f"Model Naive Bayes mencapai akurasi <b>{acc*100:.2f}%</b> pada data uji. "
            f"Model berhasil mengklasifikasikan dengan benar <b>{tp}</b> komentar Positif "
            f"dan <b>{tn}</b> komentar Negatif. "
            f"Terdapat <b>{fp}</b> false positive dan <b>{fn}</b> false negative."
        )
    else:
        st.info("Klik tombol di atas untuk menjalankan model Naive Bayes.")
