from collections import Counter
import json
import os

from logzero import logger

from eval_scripts.constants import SENS, MENS, ENTS, SEN_ID, SPAN, ENT_TYPE


def save_as_json(
        data: dict,
        output_path: str,
) -> None:

    with open(output_path, 'w', encoding='utf-8') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)
    logger.info(f'Saved: {output_path}')


def load_json(
        input_path: str,
) -> dict:

    with open(input_path, encoding='utf-8') as f:
        logger.info(f'Read: {input_path}')
        data = json.load(f)
    return data


def load_jsonl(
        input_path: str,
) -> dict:

    data = {}
    with open(input_path, encoding='utf-8') as f:
        logger.info(f'Read: {input_path}')
        for line in f:
            data_one_doc = json.loads(line.strip('\n'))
            data.update(data_one_doc)
    return data


def load_urlmap(urlmap_path: str) -> dict[str, list[str]]:
    # format: URL -> set of URLs that should be merged
    # example:
    # https://www.openstreetmap.org/node/6396868334	[https://www.openstreetmap.org/node/6396868334;https://www.openstreetmap.org/node/6397207541]

    urlmap = {}
    with open(urlmap_path, encoding='utf-8') as f:
        logger.info(f'Read: {urlmap_path}')
        for line in f:
            line = line.rstrip('\n')
            array = line.split('\t')
            urlmap[array[0]] = array[1]
    return urlmap


def rename_sentence_ids_in_dict(
        data: dict,
) -> dict:

    new_data = {}
    for docid, doc in data.items():
        senid_old2new = {}
        for senid in doc[SENS].keys():
            senid_old2new[senid] = len(senid_old2new) + 1

        new_doc = {SENS: {}, MENS: {}, ENTS: doc[ENTS]}
        new_data[docid] = new_doc

        for senid, sen in doc[SENS].items():
            new_doc[SENS][senid] = sen

        for menid, men in doc[MENS].items():
            new_men = {}
            new_men.update(men)
            new_men[SEN_ID] = senid_old2new[men[SEN_ID]]
            new_doc[MENS][menid] = new_men

    return new_data


def load_tsv_for_ner(
        input_path: str,
        rename_sentence_id: bool = True,
) -> dict:

    data = {}
    docid_to_sent_num = Counter()

    with open(input_path) as f:
        logger.info(f'Read: {input_path}')

        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            array = line.split('\t')
            docid = array[0]

            if rename_sentence_id:
                docid_to_sent_num[docid] += 1
                senid = docid_to_sent_num[docid]
            else:
                senid = array[1]

            if not docid in data:
                data[docid] = {SENS: {}, MENS: {}}
            data[docid][SENS][senid] = None

            metainfo = array[3]
            if metainfo != '':
                span_instances = []

                for mention_info in metainfo.split(';'):
                    array = mention_info.split(',')
                    men_id       = array[0]
                    indexes_str  = array[1]
                    entity_label = array[2]
                    begin, end   = indexes_str.split(':')
                    begin        = int(begin)
                    end          = int(end)
                    men_id       = f'{senid}_{begin}_{end}'
                    span_instances.append((men_id, begin, end, entity_label))
                    # generic      = array[3] if len(array) > 3 else None
                    # spec_amb     = array[4] if len(array) > 4 else None
                    # hie_amb      = array[5] if len(array) > 5 else None
                    # cid          = array[6] if len(array) > 6 else None

                for (men_id, begin, end, entity_label) in span_instances:
                    data[docid][MENS][men_id] = {
                        SEN_ID: senid,
                        SPAN: [begin, end],
                        ENT_TYPE: entity_label,
                    }

    return data


# For a file containing one document.
def load_data_from_single_file_path(
        file_path: str,
        target_docids: set[str] = None,
        rename_sentence_id: bool = True,
) -> dict[str, object]:

    data = {}

    if os.path.isfile(file_path):
        file_name = file_path.split('/')[-1]
        data_tmp = load_data_from_filepath(file_path, rename_sentence_id)
        for docid, doc in data_tmp.items():
            if target_docids and not docid in target_docids:
                continue
            data[docid] = doc

    logger.info(f'Read: {len(data)} documents.')
    return data


# For a file containing one or multiple documents.
# Expect file names in the format of of "(document ID).(extension)"
def load_data_from_paths_per_doc(
        input_paths: str,
        exts: set[str],
        target_docids: set[str],
        rename_sentence_id: bool = True,
) -> dict[str, object]:

    data = {}

    for input_path in input_paths.split(','):
        if os.path.isdir(input_path):
            dir_path = input_path

            for file_name in sorted(os.listdir(dir_path)):
                ext = file_name.split('.')[-1]
                if not ext in exts:
                    continue

                docid = '.'.join(file_name.split('.')[:-1])
                if target_docids and not docid in target_docids:
                    continue

                file_path = os.path.join(dir_path, file_name)
                data_doc = load_data_from_filepath(file_path, rename_sentence_id)
                data.update(data_doc)

        elif os.path.isfile(input_path):
            file_path = input_path
            file_name = file_path.split('/')[-1]

            ext = file_name.split('.')[-1]
            if not ext in exts:
                continue

            docid = '.'.join(file_name.split('.')[:-1])
            if target_docids and not docid in target_docids:
                continue

            data_doc = load_data_from_filepath(file_path, rename_sentence_id)
            data.update(data_doc)

    logger.info(f'Read: {len(data)} documents.')
    return data


def load_data_from_filepath(
        file_path: str,
        rename_sentence_id: bool = True,
) -> dict:

    if file_path.endswith('.json'):
        data = load_json(file_path)
        if rename_sentence_id:
            return rename_sentence_ids_in_dict(data)
        else:
            return data

    elif file_path.endswith('.jsonl'):
        data = load_jsonl(file_path)
        if rename_sentence_id:
            return rename_sentence_ids_in_dict(data)
        else:
            return data

    elif file_path.endswith('.tsv'):
        return load_tsv_for_ner(file_path, rename_sentence_id=rename_sentence_id)
