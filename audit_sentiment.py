#!/usr/bin/env python3
"""
Script audit label sentimen menggunakan Claude API.
Menganalisis kebenaran label sentimen pada tweet berbahasa Indonesia.
"""

import csv
import json
import os
import time
import anthropic

# Konfigurasi
MODEL = "claude-haiku-4-5"
BATCH_SIZE = 20
SAVE_INTERVAL = 1000
PROGRESS_INTERVAL = 500

INPUT_FILES = {
    "negatif": "/Users/hanaazizah/Developer/sentiment-analysis-emoji/negatif.csv",
    "netral": "/Users/hanaazizah/Developer/sentiment-analysis-emoji/netral.csv",
    "positif": "/Users/hanaazizah/Developer/sentiment-analysis-emoji/positif.csv",
}

OUTPUT_FILE = "/Users/hanaazizah/Developer/sentiment-analysis-emoji/hasil_audit_label.csv"
TEMP_FILE = "/Users/hanaazizah/Developer/sentiment-analysis-emoji/audit_temp.jsonl"

SYSTEM_PROMPT = """Kamu adalah analis sentimen ahli untuk bahasa Indonesia.
Tugasmu menganalisis tweet berbahasa Indonesia dan menentukan label sentimen yang tepat.

Panduan analisis:
- Pertimbangkan KONTEKS KESELURUHAN kalimat, bukan hanya kata per kata
- Kenali SARKASME & SINDIRAN (misalnya "bagus banget" yang sebenarnya negatif)
- Pahami SLANG & BAHASA GAUL Indonesia: wkwk/wkwkwk (tawa/netral), anjir/anjay (ekspresi kuat), gabut (bosan), baper (bawa perasaan), bucin (budak cinta), lebay (berlebihan), auto (otomatis), dll
- Pahami SINGKATAN MEDSOS: woy/woi (panggilan), mksd (maksud), blm (belum), gpp (gapapa), dll
- Emoji bisa memperkuat atau membalik makna teks
- "wkwkwk" di konteks marah = ironi/negatif; di konteks biasa = netral/positif
- Keluhan bisa disampaikan dengan humor (tetap negatif)
- Pujian sarkastis = negatif
- Ekspresi campuran = ambiguous

Label yang valid:
- positif: ekspresi senang, gembira, kagum, cinta, harapan, dukungan, terima kasih tulus
- netral: informasi, pertanyaan tanpa emosi kuat, pernyataan biasa
- negatif: ekspresi sedih, marah, kecewa, benci, frustrasi, keluhan
- ambiguous: benar-benar tidak jelas atau campuran yang seimbang"""


def load_all_tweets():
    """Muat semua tweet dari ketiga file CSV."""
    all_tweets = []
    for label, filepath in INPUT_FILES.items():
        print(f"Membaca {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                all_tweets.append({
                    "nomor_baris": i + 1,  # nomor baris dalam file asli (tidak termasuk header)
                    "file_sumber": label,
                    "tweet": row.get("tweet_emoji", "").strip(),
                    "label_saat_ini": label,
                })
    print(f"Total tweet dimuat: {len(all_tweets)}")
    return all_tweets


def load_progress():
    """Muat progress yang sudah disimpan."""
    hasil = {}
    if os.path.exists(TEMP_FILE):
        print(f"Ditemukan file progress: {TEMP_FILE}")
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        key = f"{item['file_sumber']}_{item['nomor_baris']}"
                        hasil[key] = item
                    except json.JSONDecodeError:
                        pass
        print(f"Progress dimuat: {len(hasil)} tweet sudah dianalisis")
    return hasil


def save_progress(hasil_dict):
    """Simpan progress ke file sementara."""
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        for item in hasil_dict.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def analyze_batch(client, tweets_batch, max_retries=3):
    """Analisis batch tweet dengan Claude API."""
    # Buat prompt untuk batch
    tweets_json = []
    for item in tweets_batch:
        tweets_json.append({
            "index": item["index"],
            "tweet": item["tweet"],
            "label_saat_ini": item["label_saat_ini"]
        })

    prompt = f"""Analisis sentimen tweet-tweet berikut. Perhatikan konteks bahasa Indonesia, sarkasme, slang, dan makna keseluruhan.

Tweet yang perlu dianalisis (dalam format JSON):
{json.dumps(tweets_json, ensure_ascii=False, indent=2)}

Berikan respons dalam format JSON array dengan field berikut untuk setiap tweet:
- index: (sama dengan index input)
- label_seharusnya: positif/netral/negatif/ambiguous
- tingkat_keyakinan: Tinggi/Sedang/Rendah
- alasan: penjelasan singkat (1-2 kalimat) mengapa label tersebut tepat

Contoh format respons:
[
  {{
    "index": 0,
    "label_seharusnya": "negatif",
    "tingkat_keyakinan": "Tinggi",
    "alasan": "Tweet mengungkapkan kekecewaan dan frustrasi meskipun menggunakan tanda tawa."
  }}
]

PENTING: Hanya kembalikan JSON array, tidak ada teks lain."""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text.strip()
            # Bersihkan jika ada markdown code block
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            results = json.loads(text)
            return results

        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Kembalikan hasil default jika gagal
                return [{"index": item["index"], "label_seharusnya": "ambiguous",
                         "tingkat_keyakinan": "Rendah", "alasan": "Gagal dianalisis"}
                        for item in tweets_batch]
            time.sleep(2 ** attempt)

        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit! Menunggu {wait} detik...")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            print(f"  API Error (attempt {attempt+1}/{max_retries}): {e.status_code}")
            if attempt == max_retries - 1:
                return [{"index": item["index"], "label_seharusnya": "ambiguous",
                         "tingkat_keyakinan": "Rendah", "alasan": f"API Error: {e.status_code}"}
                        for item in tweets_batch]
            time.sleep(2 ** attempt)

        except Exception as e:
            print(f"  Error tidak terduga (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return [{"index": item["index"], "label_seharusnya": "ambiguous",
                         "tingkat_keyakinan": "Rendah", "alasan": f"Error: {str(e)[:50]}"}
                        for item in tweets_batch]
            time.sleep(2 ** attempt)

    return []


def determine_status(label_saat_ini, label_seharusnya):
    """Tentukan status perbandingan label."""
    if label_seharusnya == "ambiguous":
        return "PERLU_REVIEW"
    elif label_saat_ini == label_seharusnya:
        return "SESUAI"
    else:
        return "TIDAK_SESUAI"


def main():
    print("=" * 60)
    print("AUDIT LABEL SENTIMEN TWEET BAHASA INDONESIA")
    print("=" * 60)

    # Inisialisasi client - API key dari environment atau prompt
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Masukkan ANTHROPIC_API_KEY: ").strip()
    client = anthropic.Anthropic(api_key=api_key)

    # Muat semua tweet
    all_tweets = load_all_tweets()

    # Muat progress sebelumnya
    hasil_dict = load_progress()

    # Filter tweet yang belum dianalisis
    pending = []
    for tweet in all_tweets:
        key = f"{tweet['file_sumber']}_{tweet['nomor_baris']}"
        if key not in hasil_dict:
            pending.append(tweet)

    print(f"\nTweet yang perlu dianalisis: {len(pending)}")
    print(f"Tweet yang sudah dianalisis: {len(hasil_dict)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)

    # Proses dalam batch
    total_processed = len(hasil_dict)

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]

        # Tambahkan index untuk tracking
        for i, item in enumerate(batch):
            item["index"] = i

        # Analisis batch
        results = analyze_batch(client, batch)

        # Simpan hasil
        results_by_index = {r["index"]: r for r in results}

        for item in batch:
            idx = item["index"]
            result = results_by_index.get(idx, {
                "label_seharusnya": "ambiguous",
                "tingkat_keyakinan": "Rendah",
                "alasan": "Tidak ada hasil"
            })

            key = f"{item['file_sumber']}_{item['nomor_baris']}"
            hasil_dict[key] = {
                "nomor_baris": item["nomor_baris"],
                "file_sumber": item["file_sumber"],
                "tweet": item["tweet"],
                "label_saat_ini": item["label_saat_ini"],
                "label_seharusnya": result.get("label_seharusnya", "ambiguous"),
                "tingkat_keyakinan": result.get("tingkat_keyakinan", "Rendah"),
                "alasan": result.get("alasan", ""),
                "status": determine_status(
                    item["label_saat_ini"],
                    result.get("label_seharusnya", "ambiguous")
                )
            }

        total_processed += len(batch)

        # Progress reporting
        if total_processed % PROGRESS_INTERVAL < BATCH_SIZE or total_processed == len(all_tweets):
            pct = total_processed / len(all_tweets) * 100
            print(f"Progress: {total_processed}/{len(all_tweets)} tweet ({pct:.1f}%)")

        # Simpan progress secara berkala
        if total_processed % SAVE_INTERVAL < BATCH_SIZE:
            print(f"  Menyimpan progress ({total_processed} tweet)...")
            save_progress(hasil_dict)

        # Rate limit protection - kecil delay antar batch
        time.sleep(0.5)

    # Simpan progress akhir
    save_progress(hasil_dict)

    # Tulis output CSV
    print("\nMenulis hasil ke CSV...")
    fieldnames = ["nomor_baris", "file_sumber", "tweet", "label_saat_ini",
                  "label_seharusnya", "tingkat_keyakinan", "alasan", "status"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Tulis dalam urutan file asli
        for tweet in all_tweets:
            key = f"{tweet['file_sumber']}_{tweet['nomor_baris']}"
            if key in hasil_dict:
                row = {k: hasil_dict[key].get(k, "") for k in fieldnames}
                writer.writerow(row)

    print(f"Hasil disimpan ke: {OUTPUT_FILE}")

    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    total = len(hasil_dict)
    sesuai = sum(1 for v in hasil_dict.values() if v["status"] == "SESUAI")
    tidak_sesuai = sum(1 for v in hasil_dict.values() if v["status"] == "TIDAK_SESUAI")
    perlu_review = sum(1 for v in hasil_dict.values() if v["status"] == "PERLU_REVIEW")

    print(f"Total tweet dianalisis : {total:,}")
    print(f"SESUAI                 : {sesuai:,} ({sesuai/total*100:.1f}%)")
    print(f"TIDAK_SESUAI           : {tidak_sesuai:,} ({tidak_sesuai/total*100:.1f}%)")
    print(f"PERLU_REVIEW           : {perlu_review:,} ({perlu_review/total*100:.1f}%)")

    estimasi_akurasi = sesuai / total * 100
    print(f"\nEstimasi Akurasi Label : {estimasi_akurasi:.1f}%")

    # Per-file breakdown
    print("\nBreakdown per file:")
    for label in ["negatif", "netral", "positif"]:
        file_items = [v for v in hasil_dict.values() if v["file_sumber"] == label]
        if file_items:
            f_total = len(file_items)
            f_sesuai = sum(1 for v in file_items if v["status"] == "SESUAI")
            f_tidak = sum(1 for v in file_items if v["status"] == "TIDAK_SESUAI")
            f_review = sum(1 for v in file_items if v["status"] == "PERLU_REVIEW")
            print(f"  {label:8s}: {f_total:5,} total | {f_sesuai:5,} sesuai ({f_sesuai/f_total*100:.1f}%) | "
                  f"{f_tidak:4,} tidak sesuai | {f_review:4,} perlu review")

    # Tingkat keyakinan breakdown
    print("\nDistribusi Tingkat Keyakinan:")
    for tk in ["Tinggi", "Sedang", "Rendah"]:
        count = sum(1 for v in hasil_dict.values() if v["tingkat_keyakinan"] == tk)
        print(f"  {tk:6s}: {count:,} ({count/total*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SELESAI!")
    print("=" * 60)

    # Hapus temp file jika berhasil
    if os.path.exists(TEMP_FILE) and total == len(all_tweets):
        os.remove(TEMP_FILE)
        print(f"File sementara dihapus: {TEMP_FILE}")


if __name__ == "__main__":
    main()
