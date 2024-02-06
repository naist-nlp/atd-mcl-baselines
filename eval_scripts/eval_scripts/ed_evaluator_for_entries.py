import argparse
from collections import Counter
from typing import Tuple

import logzero
from logzero import logger
from sklearn.metrics import cohen_kappa_score

from eval_scripts.constants import (
    ENTS, ORIG_ENT_ID, MEM_MEN_IDS, ENT_TYPE_MRG, NORM_NAME,
    HAS_REF, BEST_REF_URL, BEST_REF_TYPE, PARTOF, NOT_FOUND, OSM_URL_PREFIX
)
from eval_scripts.data_io import (
    save_as_json, load_data_from_paths_per_doc, load_data_from_single_file_path, load_urlmap,
)
from eval_scripts.util import GOLD, PRED, CORRECT
from eval_scripts.util import get_PRF_scores, get_PRF_scores_str, get_simple_scores_str


ALL       = 'ALL'
InKB      = 'InKB'
OOKB      = 'OutOfKB'
InOSM     = 'InOSM'
OOOSM     = 'OOOSM'
ENT_NONEX = 'ent_nonex'         # entity nonexist


# for cluster-level F1
class CountForED:
    def __init__(self):
        self.counter = Counter()


    def update(self, gold_url: str, pred_url: str) -> None:
        if gold_url != ENT_NONEX:
            if gold_url != NOT_FOUND:
                self.counter[(InKB, GOLD)] += 1
            else:
                self.counter[(OOKB, GOLD)] += 1

        if pred_url != ENT_NONEX:
            if pred_url != NOT_FOUND:
                self.counter[(InKB, PRED)] += 1
            else:
                self.counter[(OOKB, PRED)] += 1

        if gold_url != ENT_NONEX and pred_url != ENT_NONEX:
            if gold_url == pred_url:
                if gold_url != NOT_FOUND:
                    self.counter[(InKB, CORRECT)] += 1
                else:
                    self.counter[(OOKB, CORRECT)] += 1


def get_url_and_type(
        ent: dict,
        osm_only: bool = False,
        use_binary: bool = False,
        urlmap: dict = {},
) -> Tuple[str, str]:

    if osm_only:
        has_ref = ent[HAS_REF] and ent[BEST_REF_URL].startswith(OSM_URL_PREFIX)
    else:
        has_ref = ent[HAS_REF]

    if use_binary:
        return has_ref, None, None

    if has_ref:
        url      = ent[BEST_REF_URL]
        url_type = ent[BEST_REF_TYPE]                

        if urlmap and url in urlmap:
            url = urlmap[url]
        if url_type.endswith(PARTOF):
           url = f'{PARTOF}#{url}'

    else:
        url      = NOT_FOUND
        url_type = ''

    return url, url_type


def evaluate_for_f1(
        gold_data: dict,
        pred_data: dict,
        urlmap: dict = {},
        use_orig_ent_id: bool = False,
        include_nonex: bool = False,
        osm_only: bool = False,
        debug: bool = False,
) -> CountForED:

    count = CountForED()

    for docid, gold_doc in gold_data.items():
        gold_ents = gold_doc[ENTS]
        if use_orig_ent_id:
            gold_ents = {ent[ORIG_ENT_ID]: ent for ent in gold_ents.values()}

        if docid in pred_data:
            pred_ents = pred_data[docid][ENTS]
            if use_orig_ent_id:
                pred_ents = {ent[ORIG_ENT_ID]: ent for ent in pred_ents.values()}

        else:
            # TODO check
            pred_ents = {}

        for entid, gold_ent in gold_ents.items():
            gold_members = gold_ent[MEM_MEN_IDS]  # for debug
            gold_name    = gold_ent[NORM_NAME]    # for debug
            gold_elabel  = gold_ent[ENT_TYPE_MRG] # for debug
            gold_url, gold_url_type = get_url_and_type(
                gold_ent, osm_only=osm_only, urlmap=urlmap)
                
            if entid in pred_ents:
                pred_entid   = entid                  # for debug
                pred_ent     = pred_ents[entid]
                pred_name    = pred_ent[NORM_NAME]    # for debug
                pred_elabel  = pred_ent[ENT_TYPE_MRG] # for debug
                pred_members = gold_ent[MEM_MEN_IDS]  # for debug
                pred_url, pred_url_type = get_url_and_type(
                    pred_ent, osm_only=osm_only, urlmap=urlmap)
            else:
                if not include_nonex:
                    continue

                pred_entid    = None      # for debug
                pred_name     = ''        # for debug
                pred_elabel   = None      # for debug
                pred_members  = None      # for debug
                pred_url      = ENT_NONEX
                pred_url_type = ''        # for debug

            count.update(gold_url, pred_url)

            if debug:
                gold_mem_str = ';'.join(gold_members) if gold_members != None else '' 
                pred_mem_str = ';'.join(pred_members) if pred_members != None else '' 
                print(f'doc_id:{docid}\tsame_members:{gold_members==pred_members}\tsame_url:{gold_url==pred_url}\tprac_same:')
                print(f'- {entid}\t{gold_url_type}\t{gold_url}\t{gold_name}\t{gold_mem_str}')
                print(f'- {pred_entid}\t{pred_url_type}\t{pred_url}\t{pred_name}\t{pred_mem_str}')

        if not include_nonex:
            continue

        for entid, pred_ent in pred_ents.items():
            if entid in gold_ents: # already checked
                continue
            else:
                gold_entid    = None      # for debug
                gold_name     = ''        # for debug
                gold_members  = None
                gold_url      = ENT_NONEX
                gold_url_type = ''        # for debug

                pred_name    = pred_ent[NORM_NAME]    # for debug
                pred_elabel  = pred_ent[ENT_TYPE_MRG] # for debug
                pred_members = pred_ent[MEM_MEN_IDS]  # for debug
                pred_url, pred_url_type = get_url_and_type(
                    pred_ent, osm_only=osm_only, urlmap=urlmap)

                count.update(gold_url, pred_url)

            if debug:
                gold_mem_str = ';'.join(gold_members) if gold_members != None else '' 
                pred_mem_str = ';'.join(pred_members) if pred_members != None else '' 
                print(f'doc_id:{docid}\tsame_members:{gold_members==pred_members}\tsame_url:{gold_url==pred_url}\tprac_same:')
                print(f'- {gold_entid}\t{gold_url_type}\t{gold_url}\t{gold_name}\t{gold_mem_str}')
                print(f'- {entid}\t{pred_url_type}\t{pred_url}\t{pred_name}\t{pred_mem_str}')

    return count


def evaluate_for_kappa(
        gold_data: dict,
        pred_data: dict,
        urlmap: dict = {},
        use_orig_ent_id: bool = False,
        osm_only: bool = False,
        use_binary: bool = False, # cast values as FOUND or NOT_FOUND
        include_nonex: bool = False,
) -> dict:
    g_seq, p_seq = get_gold_and_pred_url_sequences(
        gold_data, pred_data, urlmap=urlmap, 
        osm_only=osm_only, 
        use_binary=use_binary,
        include_nonex=include_nonex,
    )
    k_score = cohen_kappa_score(g_seq, p_seq)
    n_match = sum([1 if g == p else 0 for g, p in zip(g_seq, p_seq)])
    scores = {
        'n_ins': len(g_seq),
        'n_match': n_match,
        'kappa': float(k_score),
    }
    return scores


def get_gold_and_pred_url_sequences(
        gold_data: dict,
        pred_data: dict,
        urlmap: dict = {},
        use_orig_ent_id: bool = True,
        osm_only: bool = False,
        use_binary: bool = False, # cast values as FOUND or NOT_FOUND
        include_nonex: bool = False,
) -> Tuple[list, list]:

    gold_url_seq = []
    pred_url_seq = []

    for docid, gold_doc in gold_data.items():
        gold_ents_orig = gold_doc[ENTS]
        if use_orig_ent_id:
            gold_ents = {ent[ORIG_ENT_ID]: ent for ent in gold_ents_orig.values()}
        else:
            gold_ents = gold_ents_orig

        if docid in pred_data:
            pred_ents_orig = pred_data[docid][ENTS]
            if use_orig_ent_id:
                pred_ents = {ent[ORIG_ENT_ID]: ent for ent in pred_ents_orig.values()}
            else:
                pred_ents = pred_ents_orig

        else:
            # TODO check
            pred_ents = {}

        for entid, gold_ent in gold_ents.items():
            gold_url, _ = get_url_and_type(gold_ent, osm_only, use_binary, urlmap)
                
            if entid in pred_ents:
                pred_ent = pred_ents[entid]
                pred_url, _ = get_url_and_type(pred_ent, osm_only, use_binary, urlmap)

                gold_url_seq.append(gold_url)
                pred_url_seq.append(pred_url)

            else:
                logger.debug(f'Entity {docid}-{entid} does only exist in gold results.')

                if include_nonex:
                    pred_url = False if use_binary else ENT_NONEX
                    gold_url_seq.append(gold_url)
                    pred_url_seq.append(pred_url)

        for entid, pred_ent in pred_ents.items():
            if entid in gold_ents: # already checked
                continue

            else:
                logger.debug(f'Entity {docid}-{entid} does only exist in pred results.')

                if include_nonex:
                    gold_url = False if use_binary else ENT_NONEX
                    pred_url, _, _ = get_url_and_type(pred_ent, osm_only, urlmap)
                    gold_url_seq.append(gold_url)
                    pred_url_seq.append(pred_url)

    return gold_url_seq, pred_url_seq


if __name__ == '__main__':
    logzero.loglevel(20)

    parser = argparse.ArgumentParser()
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
        '--urlmap_path', '-u',
        type=str,
    )
    parser.add_argument(
        '--use_orig_ent_id',
        action='store_true',
    )
    parser.add_argument(
        '--eval_one_side_entity',
        action='store_true',
        help='evaluate entity in either of pred and gold results'
    )
    parser.add_argument(
        '--output_score_path', '-o',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    urlmap = load_urlmap(args.urlmap_path) if args.urlmap_path else {}

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

    count = evaluate_for_f1(
        gold_data, pred_data, urlmap=urlmap,
        use_orig_ent_id=args.use_orig_ent_id,
        include_nonex=args.eval_one_side_entity,
        osm_only=False,
        debug=False,
    )
    f_scores = get_PRF_scores(count.counter, label_display_order=[InKB, OOKB])
    f_res = get_PRF_scores_str(f_scores)

    k_scores = evaluate_for_kappa(
        gold_data, pred_data, urlmap=urlmap, osm_only=False, use_binary=False,
        include_nonex=args.eval_one_side_entity)
    k_res = get_simple_scores_str(k_scores)

    print('\n[KB=Any]')
    print(f_res)
    print(k_res)

    count_osm = evaluate_for_f1(
        gold_data, pred_data, urlmap=urlmap,
        use_orig_ent_id=args.use_orig_ent_id,
        include_nonex=args.eval_one_side_entity, 
        osm_only=True,
        debug=False)
    f_scores_osm = get_PRF_scores(count_osm.counter, label_display_order=[InOSM, OOOSM])
    f_res_osm = get_PRF_scores_str(f_scores_osm)

    k_scores_osm = evaluate_for_kappa(
        gold_data, pred_data, urlmap=urlmap, osm_only=True, use_binary=False,
        include_nonex=args.eval_one_side_entity)
    k_res_osm = get_simple_scores_str(k_scores_osm)

    print('\n[KB=OSM]')
    print(f_res_osm)
    print(k_res_osm)

    if args.output_score_path:
        scores = {}
        scores.update(f_scores)
        scores['kappa'] = k_scores
        save_as_json(scores, args.output_score_path)
