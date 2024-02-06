import argparse
from collections import Counter

from logzero import logger

from eval_scripts.constants import SENS, MENS, SEN_ID, SPAN, ENT_TYPE
from eval_scripts.data_io import (
    load_json, save_as_json,
    load_data_from_paths_per_doc,
    load_data_from_single_file_path,
)
from eval_scripts.util import GOLD, PRED, CORRECT
from eval_scripts.util import get_PRF_scores, get_PRF_scores_str


POLICY_LOC_FAC_LINE_IDENTIFY  = 'IDENTIFY'
POLICY_LOC_FAC_LINE_DUPLICATE = 'DUPLICATE'

OVERALL  = 'OVERALL'
NAME_ALL = 'NAME_ALL'
NOM_ALL  = 'NOM_ALL'
ORG_ALL  = 'ORG_ALL'

ANY_NAME = 'ANY_NAME'
ANY_NOM  = 'ANY_NOM'
ANY_ORG  = 'ANY_ORG'

LOC_FAC_LINE_NAME      = 'LOC_FAC_LINE_NAME'
LOC_FAC_LINE_NAME_LIST = ['LOC_NAME', 'FAC_NAME', 'LINE_NAME']


class CountForNER:
    def __init__(
            self, 
            use_abstract_labels: bool = False, 
            label_map: dict = None,
            label_conversion_policy: str = None,
    ):
        self.counter = Counter()
        self.use_abstract_labels = use_abstract_labels
        self.label_map = label_map
        self.label_conversion_policy = label_conversion_policy


    def update_for_dataset(
            self,
            gold_data: dict,
            pred_data: dict,
    ) -> None:

        for doc_id, gold_doc in gold_data.items():
            sen_ids = gold_doc[SENS].keys()
            sen_id2gold_spans = {sen_id: [] for sen_id in sen_ids}
            sen_id2pred_spans = {sen_id: [] for sen_id in sen_ids}

            for men_id, men in gold_doc[MENS].items():
                if not men[SEN_ID] in sen_id2gold_spans:
                    sen_id2gold_spans[men[SEN_ID]] = []

                sen_id2gold_spans[men[SEN_ID]].append(
                    (men[SPAN][0], men[SPAN][1], men[ENT_TYPE]))

            if doc_id in pred_data:
                pred_doc = pred_data[doc_id]
                for men_id, men in pred_doc[MENS].items():
                    if not men[SEN_ID] in sen_id2pred_spans:
                        sen_id2pred_spans[men[SEN_ID]] = []

                    sen_id2pred_spans[men[SEN_ID]].append(
                        (men[SPAN][0], men[SPAN][1], men[ENT_TYPE]))

            else:
                logger.warning(f'Prediction for {doc_id} is not available.')

            for sen_id in sen_id2gold_spans.keys():
                gold_spans = []
                if sen_id in sen_id2gold_spans:
                    gold_spans = sen_id2gold_spans[sen_id]

                pred_spans = []
                if sen_id in sen_id2pred_spans:
                    pred_spans = sen_id2pred_spans[sen_id]
                    
                self.update(gold_spans, pred_spans)


    def update(
            self,
            gold_spans,
            pred_spans
    ) -> None:

        if self.label_map is not None:
            pred_spans = [(pb, pe, self.label_map[plabel]) for pb, pe, plabel in pred_spans
                          if plabel in self.label_map]

        if self.label_conversion_policy == POLICY_LOC_FAC_LINE_IDENTIFY:
            gold_spans = [(gb, ge, LOC_FAC_LINE_NAME 
                           if glabel in LOC_FAC_LINE_NAME_LIST else glabel)
                          for gb, ge, glabel in gold_spans]

            pred_spans = [(pb, pe, LOC_FAC_LINE_NAME 
                           if plabel in LOC_FAC_LINE_NAME_LIST else plabel)
                          for pb, pe, plabel in pred_spans]

        elif self.label_conversion_policy == POLICY_LOC_FAC_LINE_DUPLICATE:
            pred_spans = duplicate_spans(pred_spans, LOC_FAC_LINE_NAME_LIST)

        if self.use_abstract_labels:
            gold_spans = [(gb, ge, get_abstract_label(glabel)) for gb, ge, glabel in gold_spans]
            pred_spans = [(pb, pe, get_abstract_label(plabel)) for pb, pe, plabel in pred_spans]

        for gspan in gold_spans:
            gb, ge, glabel = gspan
            self.counter[(glabel, GOLD)] += 1

            if gspan in pred_spans:
                self.counter[(glabel, CORRECT)] += 1

        for pspan in pred_spans:
            pb, pe, plabel = pspan
            self.counter[(plabel, PRED)] += 1


def get_abstract_label(label: str) -> str:
    if label.endswith('NAME'):
        return ANY_NAME
    elif label.endswith('NOM') or label == 'LOC_OR_FAC':
        return ANY_NOM
    elif label.endswith('ORG'):
        return ANY_ORG
    else:
        return label


def duplicate_spans(
        spans: list,
        label_list: list[str],
) -> list:

    new_spans = []
    for span in spans:
        begin, end, label = span
        if label in label_list:
            for label_ in label_list:
                new_spans.append((begin, end, label_))
        else:
            new_spans.append((begin, end, label))
            
    return new_spans


def get_summary(counter: Counter) -> Counter:
    sum_counter = Counter()

    for (label, vtype), value in counter.items():
        sum_counter[(OVERALL, vtype)] += value

        if label.endswith('NAME'):
            sum_counter[(NAME_ALL, vtype)] += value

        elif label.endswith('NOM') or label == 'LOC_OR_FAC':
            sum_counter[(NOM_ALL, vtype)] += value

        elif label.endswith('ORG'):
            sum_counter[(ORG_ALL, vtype)] += value

    return sum_counter


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
        '--use_abstract_labels', '-a',
        action='store_true'
    )
    parser.add_argument(
        '--label_conversion_map_path', '-l',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--label_conversion_policy',
        type=str,
        default=None,
        choices=(None, 'IDENTIFY', 'DUPLICATE'),
    )
    parser.add_argument(
        '--label_display_order',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--output_score_path', '-o',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    rename_sentence_id = not args.no_rename_sentence_id

    label_map = None
    if args.label_conversion_map_path:
        label_map = load_json(args.label_conversion_map_path)

    label_order = []
    if args.label_display_order:
        for label in args.label_display_order.split(','):
            label_order.append(label)

    count = CountForNER(
        use_abstract_labels=args.use_abstract_labels, 
        label_map=label_map,
        label_conversion_policy=args.label_conversion_policy,
    )

    if args.gold_single_input_file:
        gold_data = load_data_from_single_file_path(
            args.gold_paths, args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )
    else:
        gold_data = load_data_from_paths_per_doc(
            args.gold_paths, set({'tsv', 'json'}), args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )

    if args.pred_single_input_file:
        pred_data = load_data_from_single_file_path(
            args.pred_paths, args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )
    else:
        pred_data = load_data_from_paths_per_doc(
            args.pred_paths, set({'tsv', 'json'}), args.target_docids,
            rename_sentence_id=rename_sentence_id,
        )

    count.update_for_dataset(gold_data, pred_data)

    label_order_sum = [OVERALL, NAME_ALL, NOM_ALL, ORG_ALL]
    scores_sum = get_PRF_scores(
        get_summary(count.counter), calc_overall=False, label_display_order=label_order_sum)

    if not label_order:
        label_order = [
            ANY_NAME, ANY_NOM, ANY_ORG, 'LOC_NAME', 'FAC_NAME', 'LINE_NAME', 'TRANS_NAME',
            'LOC_NOM', 'FAC_NOM', 'LINE_NOM', 'TRANS_NOM', 'LOC_OR_FAC', 'DEICTIC',
            'LOC_ORG', 'FAC_ORG',
        ]
    scores_each = get_PRF_scores(
        count.counter, calc_overall=False, label_display_order=label_order)

    scores = scores_sum['each_label']
    scores.update(scores_each)    
    res = get_PRF_scores_str(scores)
    print(res)

    if args.output_score_path:
        save_as_json(scores, args.output_score_path)
