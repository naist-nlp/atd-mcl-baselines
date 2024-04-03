import codecs
import json
import time

import h5py
import numpy as np


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_path',
        default='data/test.json',
        type=str,
        help='path to input json file'
    )
    parser.add_argument(
        '-iv', '--input_vec',
        default='data/test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5',
        type=str,
        help='path to input json file'
    )
    parser.add_argument(
        '-ei', '--entry_id',
        default='data/20230620_all_extnames.ids.txt',
        type=str,
        help='path to ids of OSM database entries'
    )
    parser.add_argument(
        '-ev', '--entry_vec',
        default='data/20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5',
        type=str,
        help='path to vectors of OSM database entries'
    )
    parser.add_argument(
        '-o', '--output_path',
        default=None,
        type=str,
        help='path to output json file'
    )
    parser.add_argument(
        '-s', '--entry_size',
        default=None,
        type=int,
        help='entry size')
    args = parser.parse_args()
    if args.output_path is None:
        output_path = args.input_path.replace(
            ".json", f".ed_results.json"
        )
    else:
        output_path = args.output_path

    disambiguate(
        args.input_path, args.input_vec,
        args.entry_id, args.entry_vec,
        output_path, args.entry_size
    )


def load_json(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def write_json(filename, data):
    with codecs.open(filename, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def disambiguate(
        input_path: str, input_vec_path: str,
        entry_id_path: str, entry_vec_path: str,
        output_path: str, entry_size: int
):
    prev_time = time.time()
    entry_hdf5 = h5py.File(entry_vec_path, "r")
    entry_vecs = []
    for entry_key in range(len(entry_hdf5.keys())):
        entry_vecs.append(entry_hdf5[str(entry_key)])
        if entry_size and len(entry_vecs) == entry_size:
            break
    print(f'{len(entry_vecs)} entry vectors have been loaded', flush=True)
    print('- {:.2f} seconds'.format(time.time() - prev_time), flush=True)
    entry_vecs = np.asarray(entry_vecs)

    with open(entry_id_path, "r") as f:
        entry_ids = [line.rstrip("\n") for line in f]
    entry_ids = entry_ids[:len(entry_vecs)]
    if entry_size:
        entry_ids = entry_ids[:entry_size]
    print(f'{len(entry_ids)} entry ids have been loaded', flush=True)

    assert len(entry_ids) == len(entry_vecs)

    input_hdf5 = h5py.File(input_vec_path, "r")
    input_vecs = [input_hdf5[str(input_key)]
                  for input_key in range(len(input_hdf5.keys()))]
    input_vecs = np.asarray(input_vecs)
    print(f'{len(input_vecs)} input vectors have been loaded', flush=True)

    results = {}
    entity_index = 0
    docs: dict = load_json(input_path)
    for doc_id, doc in docs.items():
        doc_result = {}
        for ent_id, entity in doc["entities"].items():
            if entity["has_name"]:
                entity_name = entity["normalized_name"]
                entry_group_ids: list[str] = retrieve_top_entries(
                    query_vec=input_vecs[entity_index],
                    db_entry_ids=entry_ids,
                    db_entry_vecs=entry_vecs,
                )
            else:
                entity_name = None
                entry_group_ids = []
            doc_result[ent_id] = {
                "name": entity_name,
                "member_mention_ids": entity["member_mention_ids"],
                "entry_group_ids": entry_group_ids
            }
            entity_index += 1
        results[doc_id] = doc_result
    write_json(output_path, results)
    print(f"Save the results in {output_path}")


def retrieve_top_entries(
        query_vec: np.ndarray,
        db_entry_ids: list[str],
        db_entry_vecs: np.ndarray,
        k=1000
) -> list[str]:
    scores = np.dot(db_entry_vecs, query_vec)
    arg_sort = np.argsort(scores)[::-1][:k]
    top_ranked_entries = []
    for rank, arg in enumerate(arg_sort):
        top_ranked_entries.append(db_entry_ids[arg])
    return top_ranked_entries


if __name__ == '__main__':
    main()
