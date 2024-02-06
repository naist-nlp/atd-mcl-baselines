import argparse

from logzero import logger

from eval_scripts.constants import (
    SENS, MENS, ENTS, SEN_ID, ENT_ID, TXT, MEM_MEN_IDS, SPAN, ENT_TYPE, GENERIC, SPEC_AMB, HAS_NAME
)
from eval_scripts.data_io import (
    save_as_json, load_data_from_paths_per_doc, load_data_from_single_file_path,
)
from eval_scripts.util import GOLD, PRED, CORRECT
from eval_scripts.util import is_overlap, merge_sets, calc_PRF, get_coref_scores_str
from eval_scripts.coref_metrics import (
    Evaluator, get_mention_assignments, muc, b_cubed, ceafe, lea, mentions,
)


class CountForCoref:
    def __init__(self):
        self.n_cls_ex_match = 0
        self.n_sin_ex_match = 0
        self.n_gold_nom2name = 0
        self.n_pred_nom2name = 0
        self.n_corr_nom2name = 0


    def update_nom_to_name_relation(self, g_nom_men2name_men_set, p_nom_men2name_men_set):
        self.n_gold_nom2name += len(g_nom_men2name_men_set)
        self.n_pred_nom2name += len(p_nom_men2name_men_set)

        for p_nom_men, p_name_men_set in p_nom_men2name_men_set.items():
            if p_nom_men in g_nom_men2name_men_set:
                g_name_men_set = g_nom_men2name_men_set[p_nom_men]
                if len(p_name_men_set & g_name_men_set):
                    self.n_corr_nom2name += 1


    def update_cluster_exact_match(self, g_clusters, p_clusters):
        # num of clusters that exactly matches between gold and pred cluster sets
        for g_cls in g_clusters:
            if g_cls in p_clusters:
                self.n_cls_ex_match += 1
                if len(g_cls) == 1:
                    self.n_sin_ex_match += 1


def get_nom_to_name_relations(ents, mens):
    nom_men2name_men_set = {}

    for ent_id, ent in ents.items():
        cluster = []
        has_name = ent[HAS_NAME] if HAS_NAME in ent else False
        mem_mens = ent[MEM_MEN_IDS]
        if not has_name or len(mem_mens) == 1:
            continue

        men_nids_name  = set()
        men_nids_other = set()
        for men_id in mem_mens:
            men_dict = mens[men_id]
            sen_id   = men_dict[SEN_ID]
            begin    = men_dict[SPAN][0]
            end      = men_dict[SPAN][1]
            men_nid  = f'{sen_id}-{begin}-{end}'
            elab     = men_dict[ENT_TYPE]
            if elab.endswith('NAME'):
                men_nids_name.add(men_nid)
            else:
                men_nids_other.add(men_nid)

        if len(men_nids_name) > 0 and len(men_nids_other) > 0:
            for men_nid in men_nids_other:
                nom_men2name_men_set[men_nid] = men_nids_name

    return nom_men2name_men_set


def get_clusters_for_eval(ents, mens, exclude_singletons=False, name_mention_only=False):
    clusters = []

    for ent_id, ent in ents.items():
        cluster = []
        mem_mens = ent[MEM_MEN_IDS]
        if exclude_singletons and len(mem_mens) == 1:
            continue

        for men_id in mem_mens:
            men_dict = mens[men_id]
            elab     = men_dict[ENT_TYPE]
            if name_mention_only and not elab.endswith('NAME'):
                continue

            sen_id   = men_dict[SEN_ID]
            begin    = men_dict[SPAN][0]
            end      = men_dict[SPAN][1]
            men      = (sen_id, begin, end)            
            cluster.append(men)

        if len(cluster) > 0:
            clusters.append(cluster)

    return clusters


def modify_pred_mentions_using_gold_mentions(pred_ents, pred_mens, gold_ents, gold_mens, sens=None):
    pred_mens_new = {}
    pred_ents_new = {}

    g_span2mid = {}

    for g_ent in gold_ents.values(): # mentions with specificity tag do not comprise entities
        for g_men_id in g_ent[MEM_MEN_IDS]:
            g_men = gold_mens[g_men_id]
            g_span = (g_men[SEN_ID], g_men[SPAN][0], g_men[SPAN][1])
            g_span2mid[g_span] = g_men_id

    debug = False
    if debug:
        for p_ent_id, p_ent in pred_ents.items():
            p_member_ids = p_ent[MEM_MEN_IDS]
            print(f'- {p_ent_id}')
            for p_men_id in p_member_ids:
                p_men = pred_mens[p_men_id]
                sid   = p_men[SEN_ID]
                b_idx = p_men[SPAN][0]
                e_idx = p_men[SPAN][1]
                label    = p_men[ENT_TYPE]
                generic  = 'GEN ' if GENERIC in p_men and p_men['generic'] else ''
                spec_amb = 'SPC ' if SPEC_AMB in p_men and p_men['ref_spec_amb'] else ''
                men_text = p_men[TXT]
                sen_text = sens[sid][TXT] if sens else ''
                p_span = (sid, b_idx, e_idx)
                match = 'o' if p_span in g_span2mid else 'x'
                print(f'  {match} {p_men_id}: {generic}{spec_amb}{b_idx} {e_idx} {label} {men_text} | {sen_text}')
    
    # retain predicted mentions that are also in gold mentions
    spans_added = set()
    for p_men_id, p_men in pred_mens.items():
        p_span = (p_men[SEN_ID], p_men[SPAN][0], p_men[SPAN][1])

        if p_span in g_span2mid:
            g_men_id = g_span2mid[p_span]
            pred_mens_new[p_men_id] = p_men.copy()
            pred_mens_new[p_men_id][ENT_TYPE] = gold_mens[g_men_id][ENT_TYPE]
            spans_added.add(p_span)

    # add gold mentions that had not been in original prediction to predicted mentions and
    # add them to predicted entities as singletons 
    p_men_id_new = 1
    p_ent_id_new = 1
    for g_span, g_men_id in g_span2mid.items():
        if g_span in spans_added:
            continue

        g_men = gold_mens[g_men_id]
        p_men_id_new_str = f'X{p_men_id_new}'
        p_ent_id_new_str = f'Y{p_ent_id_new}'

        # add mention
        p_men_new = {
            SEN_ID: g_men[SEN_ID],
            ENT_ID: p_ent_id_new_str,
            SPAN: g_men[SPAN],
            ENT_TYPE: g_men[ENT_TYPE],
            TXT: g_men[TXT],
        }
        pred_mens_new[p_men_id_new_str] = p_men_new

        # add entity
        p_ent_new = {
            HAS_NAME: p_men_new[ENT_TYPE].endswith('NAME'),
            MEM_MEN_IDS: [p_men_id_new_str]
        }
        pred_ents_new[p_ent_id_new_str] = p_ent_new

        p_men_id_new += 1
        p_ent_id_new += 1

    # add multi member clusters
    for p_ent_id, p_ent in pred_ents.items():
        p_member_ids = p_ent[MEM_MEN_IDS]
        p_member_ids_new = []
        for p_men_id in p_member_ids:
            if p_men_id in pred_mens_new:
                p_member_ids_new.append(p_men_id)

        if len(p_member_ids_new) == 0:
            continue

        p_ent_new = p_ent.copy()
        if ','.join(p_member_ids) == ','.join(p_member_ids_new):
            pass
        else:
            p_ent_new[MEM_MEN_IDS] = p_member_ids_new

        p_ent_new[HAS_NAME] = (
            True in [pred_mens_new[men_id][ENT_TYPE].endswith('NAME') 
                     for men_id in p_ent_new[MEM_MEN_IDS]]
        )
        pred_ents_new[p_ent_id] = p_ent_new

    # check
    g_all_mentions = set(g_span2mid.keys())
    p_all_mentions = set()
    for p_men_id, p_men in pred_mens_new.items():
        p_span = (p_men[SEN_ID], p_men[SPAN][0], p_men[SPAN][1])
        p_all_mentions.add(p_span)
    assert g_all_mentions == p_all_mentions

    p_all_mentions_in_ents = set()
    for p_ent in pred_ents_new.values():
        for p_men_id in p_ent[MEM_MEN_IDS]:
            p_men = pred_mens_new[p_men_id]
            p_span = (p_men[SEN_ID], p_men[SPAN][0], p_men[SPAN][1])
            p_all_mentions_in_ents.add(p_span)

    assert p_all_mentions == p_all_mentions_in_ents

    return pred_ents_new, pred_mens_new


def modify_pred_mentions_using_gold_mentions_considering_overlap(
        pred_ents, pred_mens, gold_ents, gold_mens, docid=None):
    pred_mens_new = {}
    pred_ents_new = {}

    p_span2mid = {}
    g_span2mid = {}
    g_mids_isolated = set()
    p_mid2ovlp_g_mid = {}    # p -> g;   p is overlapped (often is included) by g
    list_p_eids_to_merged = []

    # create p_span2mid
    for p_men_id, p_men in pred_mens.items():
        p_span = (p_men[SEN_ID], p_men[SPAN][0], p_men[SPAN][1])
        p_span2mid[p_span] = p_men_id

    # create g_span2mid
    for g_ent in gold_ents.values(): # mentions with specificity tag do not comprise entities
        for g_men_id in g_ent[MEM_MEN_IDS]:
            g_men = gold_mens[g_men_id]
            g_span = (g_men[SEN_ID], g_men[SPAN][0], g_men[SPAN][1])
            g_span2mid[g_span] = g_men_id

    for g_span in sorted(g_span2mid.keys()):
        g_men_id = g_span2mid[g_span]
        g_men = gold_mens[g_men_id]

        p_spans_overlapped = set()
        p_eids_to_merged = set()

        # check overlapping between g_span and each p_span (-> p_spans_overlapped)
        #   and obtain list_p_eids_to_merged
        for p_span in sorted(p_span2mid.keys()):
            p_men_id = p_span2mid[p_span]
            if (g_span[0] == p_span[0]
                and is_overlap(g_span[1], g_span[2], p_span[1], p_span[2])
            ):
                p_spans_overlapped.add(p_span)
                p_ent_id = pred_mens[p_men_id][ENT_ID]
                p_eids_to_merged.add(p_ent_id)

                list_p_eids_to_merged = merge_sets(list_p_eids_to_merged, p_eids_to_merged)

        # set p_mid2ovlp_g_mid
        if p_spans_overlapped:
            p_spans_to_exclude = set()

            for p_span in p_spans_overlapped:
                p_men_id = p_span2mid[p_span]

                if p_men_id in p_mid2ovlp_g_mid: # already found other overlapping glod spans
                    p_spans_to_exclude.add(p_span)
                    g_men_id_prev = p_mid2ovlp_g_mid[p_men_id]
                    g_men_prev = gold_mens[g_men_id_prev]
                    g_span_prev = (g_men_prev[SEN_ID], g_men_prev[SPAN][0], g_men_prev[SPAN][1])
                    logger.warning(f'pred mention {p_men_id} in {docid} is overlapped by more than two gold mentions, and second or later gold mentions are treated as isolated spans.: {p_men_id} {p_span} -> {g_men_id_prev} {g_span_prev}, {g_men_id} {g_span}')

            for p_span in p_spans_overlapped - p_spans_to_exclude:
                p_men_id = p_span2mid[p_span]
                p_mid2ovlp_g_mid[p_men_id] = g_men_id

            if not p_spans_overlapped - p_spans_to_exclude:
                # add isolated g_mid
                g_mids_isolated.add(g_men_id)

        else:
            # add isolated g_mid
            g_mids_isolated.add(g_men_id)


    # add mentions with overlap to pred_ents_new and pred_mens_new
    list_p_eids = list_p_eids_to_merged
    for p_eids in list_p_eids:
        p_men_ids_new = set()

        for p_ent_id in p_eids:
            p_ent = pred_ents[p_ent_id]
            p_men_ids = p_ent[MEM_MEN_IDS]

            for p_men_id in p_men_ids:
                if p_men_id in p_mid2ovlp_g_mid:
                    g_men_id = p_mid2ovlp_g_mid[p_men_id]
                    p_men_ids_new.add(g_men_id)

        p_men_ids_new = list(p_men_ids_new)
        
        # add entity        
        p_ent_id_new_str = '+'.join(sorted(p_eids))

        # add mention
        for g_men_id in p_men_ids_new:
            g_men = gold_mens[g_men_id]
            p_men_new = {
                SEN_ID: g_men[SEN_ID],
                ENT_ID: p_ent_id_new_str,
                SPAN: g_men[SPAN],
                ENT_TYPE: g_men[ENT_TYPE],
                TXT: g_men[TXT],
            }
            pred_mens_new[g_men_id] = p_men_new

        p_ent_new = {
            HAS_NAME: (
                True in [pred_mens_new[men_id][ENT_TYPE].endswith('NAME')
                         for men_id in p_men_ids_new]
            ),
            MEM_MEN_IDS: p_men_ids_new,
        }
        pred_ents_new[p_ent_id_new_str] = p_ent_new

    # add gold mentions with no overlap to pred_ents_new and pred_mens_new
    for g_men_id in g_mids_isolated:
        g_men = gold_mens[g_men_id]
        p_ent_id_new_str = f'Z{g_men_id[1:]}'

        # add mention
        p_men_new = {
            SEN_ID: g_men[SEN_ID],
            ENT_ID: p_ent_id_new_str,
            SPAN: g_men[SPAN],
            ENT_TYPE: g_men[ENT_TYPE],
            TXT: g_men[TXT],
        }
        pred_mens_new[g_men_id] = p_men_new

        # add entity        
        p_ent_new = {
            HAS_NAME: p_men_new[ENT_TYPE].endswith('NAME'),
            MEM_MEN_IDS: [g_men_id]
        }
        pred_ents_new[p_ent_id_new_str] = p_ent_new

    # check
    p_all_mentions = []
    for p_men_id, p_men in pred_mens_new.items():
        p_span = (p_men[SEN_ID], p_men[SPAN][0], p_men[SPAN][1])
        p_all_mentions.append(p_span)
    dup = [m for m in set(p_all_mentions) if p_all_mentions.count(m) > 1]
    assert len(dup) == 0, f'duplicated in p_all_mentions: {dup}'
    p_all_mentions_set = set(p_all_mentions)

    p_all_mentions_in_ents = []
    for p_ent_id, p_ent in pred_ents_new.items():
        for g_men_id in p_ent[MEM_MEN_IDS]:
            g_men = gold_mens[g_men_id]
            g_span = (g_men[SEN_ID], g_men[SPAN][0], g_men[SPAN][1])
            p_all_mentions_in_ents.append(g_span)
    dup = [m for m in set(p_all_mentions_in_ents) if p_all_mentions_in_ents.count(m) > 1]
    assert len(dup) == 0, f'duplicated in p_all_mentions_in_ents: {dup}'
    p_all_mentions_in_ents_set = set(p_all_mentions_in_ents)

    assert p_all_mentions_set == p_all_mentions_in_ents_set
    assert p_all_mentions_in_ents_set == set(g_span2mid.keys())

    return pred_ents_new, pred_mens_new


def evaluate(
        gold_data: dict,
        pred_data: dict,
        exclude_singletons: bool = False,
        name_mention_only: bool = False,
        constrain_by_gold_mentions: str = None,
) -> dict:

    doc_coref_info = {}

    n_gold_cls = 0
    n_pred_cls = 0
    n_gold_sin = 0              # number of singletons
    n_pred_sin = 0
    count           = CountForCoref()
    evaluator_muc   = Evaluator(muc)
    evaluator_b3    = Evaluator(b_cubed)
    evaluator_ceafe = Evaluator(ceafe)
    evaluator_lea   = Evaluator(lea)
    evaluator_men   = Evaluator(mentions)

    for docid, gold_doc in gold_data.items():
        sens = gold_doc[SENS]
        gold_ents = gold_doc[ENTS]
        gold_mens = gold_doc[MENS]

        if docid in pred_data:
            pred_doc = pred_data[docid]
        else:
            # TODO check
            pred_doc = {MENS: {}, ENTS: {}}

        pred_ents = pred_doc[ENTS]
        pred_mens = pred_doc[MENS]

        if constrain_by_gold_mentions == 'exact_match':
            pred_ents, pred_mens = modify_pred_mentions_using_gold_mentions(
                pred_ents, pred_mens, gold_ents, gold_mens, sens=sens)
        elif constrain_by_gold_mentions == 'overlap':
            pred_ents, pred_mens = modify_pred_mentions_using_gold_mentions_considering_overlap(
                pred_ents, pred_mens, gold_ents, gold_mens, docid=docid)
        else:
            pass            

        # clusters: list of clusters (cluster: list of mentions)
        p_clusters = get_clusters_for_eval(
            pred_ents, pred_mens, exclude_singletons, name_mention_only)
        g_clusters = get_clusters_for_eval(
            gold_ents, gold_mens, exclude_singletons, name_mention_only)
        count.update_cluster_exact_match(g_clusters, p_clusters)

        if name_mention_only:
            n_gold_cls += len(g_clusters)
            n_pred_cls += len(p_clusters)
            n_gold_sin += sum([1 for cls in g_clusters if len(cls) == 1])
            n_pred_sin += sum([1 for cls in p_clusters if len(cls) == 1])

        else:
            n_gold_cls += len(g_clusters)
            n_pred_cls += len(p_clusters)
            n_gold_sin += sum([1 for cls in g_clusters if len(cls) == 1])
            n_pred_sin += sum([1 for cls in p_clusters if len(cls) == 1])

            # pair from NOM mention to NAME mention
            g_nom_men2name_men_set = get_nom_to_name_relations(gold_ents, gold_mens)
            p_nom_men2name_men_set = get_nom_to_name_relations(pred_ents, pred_mens)
            count.update_nom_to_name_relation(g_nom_men2name_men_set, p_nom_men2name_men_set)

        p_mention_g_cluster = get_mention_assignments(p_clusters, g_clusters)  
        g_mention_p_cluster = get_mention_assignments(g_clusters, p_clusters)
        coref_info = (g_clusters, p_clusters, g_mention_p_cluster, p_mention_g_cluster)

        evaluator_men.update(coref_info)
        evaluator_muc.update(coref_info)
        evaluator_b3.update(coref_info)
        evaluator_ceafe.update(coref_info)
        evaluator_lea.update(coref_info)

    # evaluate nom to name relation
    p_n2n, r_n2n, f_n2n = calc_PRF(
        count.n_gold_nom2name, count.n_pred_nom2name, count.n_corr_nom2name)

    # evaluate coreference
    p_muc, r_muc, f_muc       = evaluator_muc.get_PRF()
    p_b3, r_b3, f_b3          = evaluator_b3.get_PRF()
    p_ceafe, r_ceafe, f_ceafe = evaluator_ceafe.get_PRF()
    p_lea, r_lea, f_lea       = evaluator_lea.get_PRF()
    p_men, r_men, f_men       = evaluator_men.get_PRF()

    scores = {
        'stats': {
            'n_g_cls'        : n_gold_cls,
            'n_g_mcls'       : n_gold_cls-n_gold_sin, # multi member clusters
            'n_p_cls'        : n_pred_cls,
            'n_p_mcls'       : n_pred_cls-n_pred_sin, # multi member clusters
            'n_exmatch_cls'  : count.n_cls_ex_match,  # exact match
            'n_exmatch_mcls' : count.n_cls_ex_match-count.n_sin_ex_match
        },
        'muc': {
            'precision' : p_muc,
            'recall'    : r_muc,
            'f1'        : f_muc,
        },
        'b_cubed': {
            'precision' : p_b3,
            'recall'    : r_b3,
            'f1'        : f_b3,
        },
        'ceaf_e': {
            'precision' : p_ceafe,
            'recall'    : r_ceafe,
            'f1'        : f_ceafe,
        },
        'conll': {
            'precision' : (p_muc + p_b3 + p_ceafe) / 3.0,
            'recall'    : (r_muc + r_b3 + r_ceafe) / 3.0,
            'f1'        : (f_muc + f_b3 + f_ceafe) / 3.0,
        },
        'lea': {
            'precision' : p_lea,
            'recall'    : r_lea,
            'f1'        : f_lea,
        },
        'mention': {
            'precision' : p_men,
            'recall'    : r_men,
            'f1'        : f_men,
        },
        'nom2name_coref': {
            'n_gold'    : count.n_gold_nom2name,
            'n_pred'    : count.n_pred_nom2name,
            'n_corr'    : count.n_corr_nom2name,
            'precision' : p_n2n,
            'recall'    : r_n2n,
            'f1'        : f_n2n,
        },
    }

    return scores


if __name__ == '__main__':
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
        '--target_docids', '-d',
        type=str,
    )
    parser.add_argument(
        '--no_rename_sentence_id',
        action='store_true'
    )
    parser.add_argument(
        '--constrain_by_gold_mentions', '-c', 
        type=str,
        choices=('exact_match', 'overlap')
    )
    parser.add_argument(
        '--name_mention_only',
        action='store_true'
    )
    parser.add_argument(
        '--output_score_path', '-o',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    rename_sentence_id = not args.no_rename_sentence_id
    if args.gold_single_input_file:
        gold_data = load_data_from_single_file_path(
            args.gold_paths, args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )
    else:
        gold_data = load_data_from_paths_per_doc(
            args.gold_paths, set({'json'}), args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )

    if args.pred_single_input_file:
        pred_data = load_data_from_single_file_path(
            args.pred_paths, args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )
    else:
        pred_data = load_data_from_paths_per_doc(
            args.pred_paths, set({'json'}), args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )

    scores = evaluate(
        gold_data, pred_data,
        exclude_singletons=False,
        name_mention_only=args.name_mention_only, 
        constrain_by_gold_mentions=args.constrain_by_gold_mentions,
    )

    scores_multi = evaluate(
        gold_data, pred_data,
        exclude_singletons=True,
        name_mention_only=args.name_mention_only, 
        constrain_by_gold_mentions=args.constrain_by_gold_mentions,
    )
    del scores_multi['stats']
    
    res = get_coref_scores_str(scores)
    print('\n[For all clusters]')
    print(res)

    res_multi = get_coref_scores_str(scores_multi)
    print('\n[For multi-member clusters]')
    print(res_multi)

    if args.output_score_path:
        save_as_json(scores, args.output_score_path)
