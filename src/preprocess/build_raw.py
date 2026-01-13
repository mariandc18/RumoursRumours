import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataset.convert_veracity_annotations import convert_annotations

DATA_RAW = Path('./dataset/all-rnr-annotated-threads')
OUTPUT_FILE = Path('./data/raw/new.csv')


def flatten_json(data, parent_key='', sep='_'):
    items = {}
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            items[new_key] = len(v)
        else:
            items[new_key] = v
    return items


def parse_json_file(filepath: Path):
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with filepath.open('r', encoding=encoding) as f:
                return flatten_json(json.load(f))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception as e:
            print(f"Error en {filepath}: {e}")
            return None
    print(f"No se pudo leer {filepath}")
    return None


def get_veracity_label(annotation_file: Path):
    if annotation_file.exists():
        try:
            with annotation_file.open() as f:
                return convert_annotations(json.load(f), string=True)
        except Exception as e:
            print(f"Error leyendo {annotation_file}: {e}")
    return None


def process_tweet(json_file: Path, event_name, category, thread_id, thread_veracity):
    tweet_data = parse_json_file(json_file)
    if tweet_data is None:
        return None

    tweet_veracity = get_veracity_label(json_file.parent / json_file.name.lstrip('._'))

    tweet_data.update({
        'event': event_name,
        'category': category,
        'thread_id': thread_id,
        'subfolder': json_file.parent.name,
        'thread_veracity': thread_veracity,
        'tweet_veracity': tweet_veracity
    })
    return tweet_data


def process_thread(thread_path: Path, event_name, category):
    thread_veracity = get_veracity_label(thread_path / 'annotation.json')
    thread_id = thread_path.name
    rows = []

    for subfolder in ['source-tweets', 'reactions']:
        folder = thread_path / subfolder
        if not folder.is_dir():
            continue
        for json_file in folder.glob('*.json'):
            if json_file.name.startswith('._'):
                continue
            row = process_tweet(json_file, event_name, category, thread_id, thread_veracity)
            if row:
                rows.append(row)
    return rows


def process_category(category_path: Path, event_name: str, category_name: str):
    if not category_path.is_dir():
        return []
    threads = [d for d in category_path.iterdir() if d.is_dir()]
    all_rows = []
    for thread_path in tqdm(threads, desc=f"{event_name}/{category_name}"):
        all_rows.extend(process_thread(thread_path, event_name, category_name))
    return all_rows


def process_event(event_path: Path):
    event_name = event_path.name
    all_rows = []
    for category in ['rumours', 'non-rumours']:
        all_rows.extend(process_category(event_path / category, event_name, category))
    return all_rows


def main():
    if OUTPUT_FILE.exists():
        print(f"{OUTPUT_FILE} ya existe, cargando...")
        data = pd.read_csv(OUTPUT_FILE)
        print(f"Cargado {len(data)} tweets")
        return data

    all_rows = []
    events = [d for d in DATA_RAW.iterdir() if d.is_dir() and not d.name.startswith('._')]

    for event_path in events:
        print(f"Procesando evento: {event_path.name}")
        all_rows.extend(process_event(event_path))

    data = pd.DataFrame(all_rows)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Guardado CSV: {OUTPUT_FILE}")
    return data


if __name__ == "__main__":
    data = main()
