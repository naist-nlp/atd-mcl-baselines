import argparse
from collections import Counter
import json

from logzero import logger

from eval_scripts.constants import (
    ENTS, ORIG_ENT_ID,HAS_REF, BEST_REF_URL, BEST_REF_IS_OVERSEAS, OSM_URL_PREFIX
)
from eval_scripts.data_io import (
    save_as_json, load_data_from_paths_per_doc, load_data_from_single_file_path,
)
from eval_scripts.entity_disambiguation_database import Database, load_entry_database


NAME = 'name'

ENTRY_GROUP_IDS = 'entry_group_ids'
ENTRY_IDS       = 'entry_ids'
ENTRY_URLS      = 'entry_urls'

TYPE_GROUP_ID = 'group_id'
TYPE_ENT_URL  = 'entry_url'


# for given entiteis
class CountForGroupED:
    n_instances: int
    match_counter: Counter

    def __init__(self):
        self.n_instances = 0
        self.match_counter = Counter()


    def increment_num_instances(self) -> None:
        self.increment_num_instances_by(1)


    def increment_num_instances_by(self, value: int) -> None:
        self.n_instances += value


    def update(self, top_k: int, matches: bool) -> None:
        if matches:
            self.match_counter[top_k] += 1


    def calc_scores(self) -> dict:
        scores = {}

        for top_k in sorted(self.match_counter.keys()):
            n_match = self.match_counter[top_k]
            acc = n_match*1.0 / self.n_instances
            scores[top_k] = {
                'n_gold': self.n_instances,
                'n_corr': n_match,
                'recall': acc,
            }

        return scores


def get_scores_str(scores: dict) -> str:
    res = '------------------------------\n'
    res += f'k\tn_gold\tn_corr\trecall@k\n'
    res += '------------------------------\n'
    for k, scores_for_k in scores.items():
        n_gold = scores_for_k['n_gold']
        n_corr = scores_for_k['n_corr']
        r      = scores_for_k['recall']
        res += f'{k}\t{n_gold}\t{n_corr}\t{r:.3f}\n'
    return res.strip('\n')


def evaluate_for_gold_entities(
        gold_data: dict,
        pred_data: dict,
        db: Database,
        k_values_for_recall: list[int],
        use_orig_ent_id: bool = False,
        pred_output_type: str = TYPE_GROUP_ID,
) -> CountForGroupED:

    debug = False

    count = CountForGroupED()

    # for docid, gold_doc in gold_data.items():
    for docid in sorted(gold_data.keys()):
        gold_ents = gold_data[docid][ENTS]
        if use_orig_ent_id:
            gold_ents = {ent[ORIG_ENT_ID]: ent for ent in gold_ents.values()}

        if docid in pred_data:
            pred_ents = pred_data[docid][ENTS]
            if use_orig_ent_id:
                pred_ents = {ent[ORIG_ENT_ID] if ORIG_ENT_ID in ent else ent_id: ent 
                             for ent_id, ent in pred_ents.items()}

        else:
            # TODO check
            pred_ents = {}

        for ent_id, g_entity in gold_ents.items():
            g_entity = gold_ents[ent_id]
            if not g_entity[HAS_REF]:
                # Out of DB
                continue

            if BEST_REF_IS_OVERSEAS in g_entity and g_entity[BEST_REF_IS_OVERSEAS]:
                # overseas entry
                continue

            # TODO: take SECOND_A/B_REF_URL into consideration
            g_urls = g_entity[BEST_REF_URL].split(';')
            if not g_urls[0].startswith(OSM_URL_PREFIX):
                # Out of OSM
                continue

            g_entry_ids  = [url.split(OSM_URL_PREFIX)[1] for url in g_urls]

            p_entity = pred_ents[ent_id]

            if pred_output_type == TYPE_GROUP_ID:
                # expect list
                p_gids = p_entity[ENTRY_GROUP_IDS]

                count.increment_num_instances()
                for top_k in k_values_for_recall:
                    res = db.are_entry_ids_in_gids(g_entry_ids, p_gids, top_k=top_k)
                    count.update(top_k, res)
                    if debug and top_k == 1 and res == True:
                        print(count.match_counter[top_k], docid, ent_id, p_entity['name'], p_entity['member_mention_ids'], p_entity['entry_group_ids'][0])

            elif pred_output_type == TYPE_ENT_URL:
                p_entry_urls_str = p_entry_urls = None

                if BEST_REF_URL in p_entity:
                    # expect ',' or ';' sparated string
                    p_entry_urls_str = p_entity[BEST_REF_URL]

                elif ENTRY_URLS in p_entity:
                    # expect ',' or ';' sparated string
                    p_entry_urls_str = p_entity[ENTRY_URLS]

                if ',' in p_entry_urls_str:
                    p_entry_urls = p_entry_urls_str.split(',')

                elif ';' in p_entry_urls_str:
                    p_entry_urls = p_entry_urls_str.split(';')

                else:
                    p_entry_urls = [p_entry_urls_str]

                if not p_entry_urls:
                    continue

                p_entry_ids = [url.split(OSM_URL_PREFIX)[1] for url in p_entry_urls]

                count.increment_num_instances()
                for top_k in k_values_for_recall:
                    res = False
                    for g_entry_id in g_entry_ids:
                        if g_entry_id in p_entry_ids[:top_k]:
                            res = True
                            break
                    count.update(top_k, res)

    return count
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--database_path', '-d',
        type=str,
        required=True
    )
    parser.add_argument(
        '--gold_paths', '-g',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--pred_paths', '-p',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--gold_single_input_file', '-gs',
        action='store_true'
    )
    parser.add_argument(
        '--pred_single_input_file', '-ps',
        action='store_true'
    )
    parser.add_argument(
        '--target_docids', '-t',
        type=str,
    )
    parser.add_argument(
        '--k_values_for_recall', '-k',
        type=str,
        default='1,5,10,100',
    )
    parser.add_argument(
        '--use_predicted_entities',
        action='store_true',
    )
    parser.add_argument(
        '--use_orig_ent_id',
        action='store_true',
    )
    parser.add_argument(
        '--output_score_path', '-o',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.use_predicted_entities:
        # To be implemented: alignment entities based on member exact match is necessary
        logger.info('The process for --use_predicted_entities is not implemented.')
        exit()

    if args.k_values_for_recall:
        k_values_for_recall = [int(k) for k in args.k_values_for_recall.split(',')]
    else:
        k_values_for_recall = None

    db = load_entry_database(args.database_path)

    if args.gold_single_input_file:
        gold_data = load_data_from_single_file_path(
            args.gold_paths, args.target_docids, rename_sentence_id=False)
    else:
        gold_data = load_data_from_paths_per_doc(
            args.gold_paths, set({'json'}), args.target_docids, rename_sentence_id=False)

    if args.pred_single_input_file:
        pred_data = load_data_from_single_file_path(
            args.pred_paths, args.target_docids, rename_sentence_id=False)
    else:
        pred_data = load_data_from_paths_per_doc(
            args.pred_paths, set({'json'}), args.target_docids, rename_sentence_id=False)

    count = evaluate_for_gold_entities(
        gold_data, pred_data, db,
        k_values_for_recall=k_values_for_recall,
        use_orig_ent_id=args.use_orig_ent_id,
    )

    scores = count.calc_scores()
    res = get_scores_str(scores)
    print(res)

    if args.output_score_path:
        save_as_json(scores, args.output_score_path)
