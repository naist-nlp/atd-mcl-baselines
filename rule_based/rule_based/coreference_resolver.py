import argparse
import copy

from logzero import logger

from eval_scripts.constants import (
    SENS, MENS, ENTS, SEN_ID, ENT_ID, TXT, SPAN, ENT_TYPE, GENERIC, SPEC_AMB, HAS_NAME, MEM_MEN_IDS,
)
from eval_scripts.data_io import load_data_from_filepath, save_as_json


def generate_predictions(
        data: dict,
        no_merge: bool = False,
        use_ent_label: bool = False
) -> dict:


    pred_data = {}

    for docid, doc in data.items():
        # set key2spans
        key2spans = {}
        menids_to_remove = set()

        for men_id, mention in doc[MENS].items():
            sen_id   = mention[SEN_ID]
            text     = mention[TXT]
            begin    = mention[SPAN][0]
            end      = mention[SPAN][1]
            label    = mention[ENT_TYPE]
            generic  = GENERIC in mention and mention[GENERIC]
            spec_amb = SPEC_AMB in mention and mention[SPEC_AMB]

            if (label.startswith('TRANS')
                or label.endswith('ORG')
                or generic
                or spec_amb
            ):
                menids_to_remove.add(men_id)
                continue

            if no_merge:
                key = (text, label, f'{sen_id}:{begin}-{end}')
            elif use_ent_label:
                key = (text, label)
            else:
                key = (text)

            if not key in key2spans:
                key2spans[key] = []
            key2spans[key].append((sen_id, men_id, begin, end, label))

        # create a new document with predicted entities
        doc_new = {SENS: {}, MENS: copy.deepcopy(doc[MENS]), ENTS: {}}
        pred_data[docid] = doc_new

        for men_id in menids_to_remove:
            del doc_new[MENS][men_id]

        ent_idx = 1
        for key, spans in key2spans.items():
            ent_id    = f'E{ent_idx:03d}'
            ent_text  = key[0]
            ent_label = key[1] if len(key) > 1 else None

            mem_men_ids = []
            for span in spans:
                sen_id, men_id, begin, end, men_label = span
                doc_new[MENS][men_id][ENT_ID] = ent_id
                mem_men_ids.append(men_id)

            doc_new[ENTS][ent_id] = {
                HAS_NAME: (
                    True in [doc_new[MENS][men_id][ENT_TYPE].endswith('NAME') 
                             for men_id in mem_men_ids]
                ),
                MEM_MEN_IDS: mem_men_ids,
            }
            ent_idx += 1

    return pred_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--no_merge', '-n',
        action='store_true'
    )
    parser.add_argument(
        '--use_ent_label', '-l',
        action='store_true'
    )
    args = parser.parse_args()

    gold_data = load_data_from_filepath(args.input_path, rename_sentence_id=False)
    pred_data = generate_predictions(
        gold_data,
        no_merge=args.no_merge, 
        use_ent_label=args.use_ent_label)

    save_as_json(pred_data, args.output_path)
