import yt_dlp
import pandas as pd
import time

# ── Config ────────────────────────────────────────────────────────────────────
MAX_VIDEOS_PER_QUERY = 10
MAX_COMMENTS_PER_VIDEO = 300
OUTPUT_FILE = "hasil_crawl_youtube.csv"

# Query batch 2: target +1000 komentar POSITIF baru
# (berbeda dari batch 1: lagu semangat, wisata, makanan, wholesome, konser, dll)
# QUERIES = [
#     # Motivasi & kisah sukses
#     "kisah sukses anak muda indonesia inspiratif",
#     "motivasi hidup indonesia bikin nangis terharu",
#     "podcast motivasi indonesia terbaik",
#     "cerita sukses pengusaha muda indonesia",
#     "beasiswa kuliah luar negeri indonesia bangga",

#     # Hiburan & humor positif
#     "video lucu indonesia bikin ngakak 2024",
#     "comedy sketch indonesia terbaik",
#     "stand up comedy indonesia lucu banget",
#     "kumpulan momen lucu anak indonesia",
#     "react video keren indonesia seru banget",

#     # Kebaikan sosial & heartwarming
#     "aksi sosial indonesia mengharukan",
#     "donasi bantu orang susah indonesia terharu",
#     "random acts of kindness indonesia",
#     "video mengharukan indonesia bikin nangis bahagia",
#     "relawan kemanusiaan indonesia hebat",

#     # Musik & seni
#     "cover lagu viral indonesia suara merdu 2024",
#     "penyanyi indonesia keren debut",
#     "lagu pop indonesia romantis terbaik 2024",
#     "musisi indie indonesia berbakat",
#     "konten seni kreatif anak indonesia",
# ]

# Batch 3: target +1500 komentar POSITIF baru
QUERIES = [
    "anak indonesia juara olimpiade internasional",
    "momen haru wisuda anak kampung indonesia",
    "resep masakan indonesia enak mudah dibuat",
    "traveling indonesia indah pemandangan alam",
    "cover lagu indonesia merdu bikin adem",
    "kucing lucu gemesin indonesia tiktok",
    "baby pertama kali jalan lucu menggemaskan",
    "surprise ulang tahun mengharukan indonesia",
    "atlet indonesia menang emas asian games",
    "video pernikahan adat indonesia megah indah",
    "anak kecil nyanyi merdu indonesia viral",
    "reuni teman lama indonesia terharu",
    "donasi panti asuhan indonesia mengharukan",
    "liburan keluarga indonesia seru bahagia",
    "unboxing hadiah ulang tahun kaget senang",
]

# Load emoji master
df_master = pd.read_csv("master_emoji_id.csv")
emoji_set = set(df_master["Emoji"].tolist())
emoji_mapping = dict(zip(df_master["Emoji"], df_master["Special_Tag_ID"]))

def has_emoji(text):
    return any(ch in emoji_set for ch in text)

def ganti_emoji(teks):
    for em, tag in emoji_mapping.items():
        teks = teks.replace(em, tag)
    return teks


def cari_video_ids(query, max_results=5):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "default_search": "ytsearch",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    return [e["id"] for e in info.get("entries", []) if e.get("id")]


def ambil_komentar(video_id, max_komentar=200):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "getcomments": True,
        "extractor_args": {"youtube": {"max_comments": [str(max_komentar)]}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        comments = info.get("comments") or []
        return [c["text"] for c in comments if c.get("text")]
    except Exception as e:
        print(f"    Skip {video_id}: {e}")
        return []


# ── Crawling ──────────────────────────────────────────────────────────────────
semua_komentar = []

for query in QUERIES:
    print(f'\nQuery: "{query}"')
    video_ids = cari_video_ids(query, MAX_VIDEOS_PER_QUERY)
    print(f"  Ditemukan {len(video_ids)} video")

    for vid in video_ids:
        komen = ambil_komentar(vid, MAX_COMMENTS_PER_VIDEO)
        komen_emoji = [k for k in komen if has_emoji(k)]
        semua_komentar.extend(komen_emoji)
        print(f"    [{vid}] {len(komen_emoji)} komentar dengan emoji (dari {len(komen)})")
        time.sleep(1)

print(f"\nTotal terkumpul: {len(semua_komentar)} komentar")


# ── Dedup & filter ────────────────────────────────────────────────────────────
df = pd.DataFrame({"komentar": semua_komentar})
df = df.drop_duplicates("komentar")
df = df[df["komentar"].str.len() >= 10].reset_index(drop=True)
df["komentar_with_tags"] = df["komentar"].apply(ganti_emoji)
print(f"Setelah dedup & filter: {len(df)} komentar")


# ── Labeling (HuggingFace, gratis & lokal) ────────────────────────────────────
print("\nMemuat model sentimen...")
from transformers import pipeline

sentiment_pipe = pipeline(
    "text-classification",
    model="mdhugol/indonesia-bert-sentiment-classification",
    device=-1,
)
label_map = {"LABEL_0": "negatif", "LABEL_1": "netral", "LABEL_2": "positif"}

BATCH = 32
labels = []
for i in range(0, len(df), BATCH):
    batch = df["komentar"].iloc[i : i + BATCH].tolist()
    batch = [t[:512] for t in batch]
    preds = sentiment_pipe(batch)
    labels.extend([label_map[p["label"]] for p in preds])
    print(f"  Labeling: {min(i+BATCH, len(df))}/{len(df)}")

df["sentimen llm"] = labels

print("\nDistribusi label:")
print(df["sentimen llm"].value_counts())


# ── Simpan ────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDisimpan ke {OUTPUT_FILE}")


# ── Merge ke dataset utama ────────────────────────────────────────────────────
import os
base_file = "dataset_final.csv" if os.path.exists("dataset_final.csv") else "hasil_tweet_tag.csv"
df_asli = pd.read_csv(base_file)
print(f"\nBase file: {base_file} ({len(df_asli)} baris)")

df_baru = pd.DataFrame({
    "sentimen llm":    df["sentimen llm"],
    "tweet_with_tags": df["komentar_with_tags"],
})

df_gabung = pd.concat(
    [df_asli[["sentimen llm", "tweet_with_tags"]], df_baru],
    ignore_index=True
)

# Dedup antar run crawl
sebelum = len(df_gabung)
df_gabung = df_gabung.drop_duplicates("tweet_with_tags").reset_index(drop=True)
print(f"Dedup: {sebelum} → {len(df_gabung)} baris (hapus {sebelum - len(df_gabung)} duplikat)")

print("\nDistribusi kelas setelah merge:")
print(df_gabung["sentimen llm"].value_counts())
print(f"Total data: {len(df_gabung)} baris")

df_gabung.to_csv("dataset_final.csv", index=False)
print("Disimpan ke dataset_final.csv")
