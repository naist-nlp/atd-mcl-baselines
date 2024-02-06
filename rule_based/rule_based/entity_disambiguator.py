import argparse
from collections import Counter
import json
import os

from logzero import logger

from eval_scripts.constants import MENS, ENTS, MEM_MEN_IDS, TXT, ENT_TYPE, NORM_NAME
from eval_scripts.data_io import load_data_from_filepath, save_as_json
from eval_scripts.entity_disambiguation_database import Database, TARGET_KEYS, load_entry_database


NAME = 'name'
GIDS = 'entry_group_ids'


def normalize_mention_text(text: str) -> str:
    return text.replace("「", "").replace("」", "")


def determine_entity_name_from_mention_texts(men_texts: list[str]) -> str:
    if men_texts:
        max_len = max([len(text) for text in men_texts])
        texts_with_max_len = [text for text in men_texts if len(text) == max_len]
        return texts_with_max_len[0]
    else:
        return None


def select_db_entries_for_entities(
        data: dict, 
        db: Database, 
        use_norm_name: bool = False,
        top_k: int = 100,
) -> dict:

    n_entities = 0
    n_entities_noname = 0
    n_entities_hit = 0
    n_entities_hit_multi = 0
    max_n_hit = 0

    pred_data = {}

    for docid, doc in data.items():
        pred_doc = {ENTS: {}}
        pred_data[docid] = pred_doc

        mentions = doc[MENS]
        for ent_id, entity in doc[ENTS].items():
            member_ids  = entity[MEM_MEN_IDS]

            if (use_norm_name 
                and NORM_NAME in entity and entity[NORM_NAME] != None
            ):
                entity_name = entity[NORM_NAME].split(";")[0]

            else:
                men_texts = []
                for men_id in member_ids:
                    mention   = mentions[men_id]
                    men_label = mention[ENT_TYPE]
                    if men_label.endswith("NAME"):
                        men_text  = normalize_mention_text(mention[TXT])
                        men_texts.append(men_text)

                entity_name = determine_entity_name_from_mention_texts(men_texts)
                if not men_texts:
                    n_entities_noname += 1

            gids = db.get_gids_with_exact_name(entity_name)

            n_entities += 1
            if len(gids) == 0:
                pred_gids = None
            else:
                max_n_hit = max(max_n_hit, len(gids))

                if len(gids) == 1:
                    pred_gids = list(gids)
                    n_entities_hit += 1
                else:
                    pred_gids = sorted(gids)[:top_k]
                    n_entities_hit += 1
                    n_entities_hit_multi += 1

            pred_doc[ENTS][ent_id] = {
                NAME: entity_name,
                MEM_MEN_IDS: member_ids,
                GIDS: pred_gids,
            }

    logger.info(f'Num of documents: {len(data)}')
    logger.info(f'Num of entities: {n_entities}')
    logger.info(f'Num of entities with name: {n_entities-n_entities_noname}')
    logger.info(f'Num of entities with one or more predicted DB entries: {n_entities_hit}')
    logger.info(f'Num of entities with two or more predicted DB entries: {n_entities_hit_multi}')
    logger.info(f'Max num of found candidate DB entries: {max_n_hit}')

    return pred_data


def output_results(pred_data: dict, out_path: str = None):
    if out_path:
        with open(out_path, 'w') as fw:
            fw.write(json.dumps(pred_data, ensure_ascii=False, indent=2))
            fw.write('\n')
        logger.info(f'Write: {out_path}')

    else:
        print(json.dumps(pred_data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--database_path', '-d',
        type=str,
        required=True
    )
    parser.add_argument(
        '--input_path', '-i',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--top_k', '-k',
        default=100,
        type=int
    )
    parser.add_argument(
        '--use_normalized_name', '-n',
        action='store_true'
    )
    args = parser.parse_args()

    db = load_entry_database(args.database_path)
    gold_data = load_data_from_filepath(args.input_path, rename_sentence_id=False)

    pred_data = select_db_entries_for_entities(
        gold_data, db, use_norm_name=args.use_normalized_name)
    
    save_as_json(pred_data, args.output_path)
