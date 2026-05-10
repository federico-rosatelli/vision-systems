
import os, shutil, random, hashlib
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from icrawler.builtin import (
    GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler
)

DATASET_ROOT     = Path("dataset")
CLASSES          = ["human", "robot"]
IMAGES_PER_CLASS = 300

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

MIN_WIDTH, MIN_HEIGHT = 100, 100

SEARCH_QUERIES = {
    "human": [
        "person walking street",
        "human face portrait",
        "people crowd",
        "human body full length",
        "man woman standing",
    ],
    "robot": [
        "humanoid robot",
        "industrial robot machine",
        "robot artificial intelligence",
        "android robot science",
        "robot technology future",
    ],
}

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()

def is_valid_image(path):
    try:
        with Image.open(path) as img: img.verify()
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_WIDTH and h >= MIN_HEIGHT
    except Exception: return False

def deduplicate(folder):
    seen, removed = set(), 0
    for p in list(folder.iterdir()):
        if p.is_file():
            d = file_hash(p)
            if d in seen: p.unlink(); removed += 1
            else: seen.add(d)
    return removed

def clean_folder(folder):
    removed = 0
    for p in list(folder.iterdir()):
        if p.is_file() and not is_valid_image(p):
            p.unlink(); removed += 1
    return removed

def scrape_images(class_name, target, dest):
    dest.mkdir(parents=True, exist_ok=True)
    queries   = SEARCH_QUERIES[class_name]
    per_query = max(1, target // len(queries)) + 10

    crawlers = [
        ("Bing",       BingImageCrawler),
        ("Baidu",      BaiduImageCrawler)
    ]

    for query in queries:
        existing = len([f for f in dest.iterdir() if f.is_file()])
        if existing >= int(target * 1.3): break

        for engine_name, CrawlerClass in crawlers:
            print(f"[{class_name}] query={query}  engine={engine_name}")
            try:
                crawler = CrawlerClass(
                    storage={"root_dir": str(dest)}
                )
                crawler.crawl(
                    keyword=query,
                    max_num=per_query,
                    file_idx_offset="auto", # auto-increment indexing
                )
                break
            except Exception as exc:
                print(f"Error with {engine_name}: {exc}")

def split_class(class_name, raw_dir, split_dirs):
    images = [p for p in raw_dir.iterdir() if p.is_file()]
    random.shuffle(images)

    train_imgs, temp_imgs = train_test_split(
        images, test_size=(VALID_RATIO + TEST_RATIO), random_state=42
    )
    valid_ratio_adj = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    valid_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=(1 - valid_ratio_adj), random_state=42
    )

    counts = {}
    for split_name, split_images in [
        ("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs),
    ]:
        dest = split_dirs[split_name] / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(split_images):
            ext = src.suffix.lower() or ".jpg"
            dst = dest / f"{class_name}_{split_name}_{i:04d}{ext}"
            shutil.copy2(src, dst)
        counts[split_name] = len(split_images)
    return counts

def build_dataset():
    raw_root   = DATASET_ROOT / "_raw"
    split_dirs = {
        "train": DATASET_ROOT / "train",
        "valid": DATASET_ROOT / "valid",
        "test":  DATASET_ROOT / "test",
    }


    for cls in CLASSES:
        raw_dir = raw_root / cls
        print(f"\n[SCRAPING] Class: {cls.upper()}")
        scrape_images(cls, IMAGES_PER_CLASS, raw_dir)
        print(f"  Cleaning corrupted: {clean_folder(raw_dir)} removed")
        print(f"  Deduplication: {deduplicate(raw_dir)} removed")
        total = len([f for f in raw_dir.iterdir() if f.is_file()])
        print(f"  Valid images: {total}")

    print("\n[SPLIT] Splitting train / valid / test...")
    summary = {}
    for cls in CLASSES:
        counts = split_class(cls, raw_root / cls, split_dirs)
        summary[cls] = counts
        print(f"  %-8s  train=%d  valid=%d  test=%d",
                 cls, counts["train"], counts["valid"], counts["test"])


if __name__ == "__main__":
    random.seed(42)
    build_dataset()
