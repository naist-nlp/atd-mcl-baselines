import argparse
import json
from typing import Tuple

from logzero import logger
from kyoto_reader import KyotoReader, Document, Sentence
from pyknp.juman.morpheme import Morpheme


def get_sentence_reader(text_path):
    with open(text_path) as f:
        logger.info(f'Read: {text_path}')
        for line in f:
            line = line.rstrip('\n')
            yield line
        StopIteration


def read_sentences(text_path: str) -> list[str]:
    res = []

    with open(text_path) as f:
        logger.info(f'Read: {text_path}')
        for line in f:
            line = line.rstrip('\n')
            res.append(line)

    return res


def read_doc(knp_path: str) -> Document:
    logger.info(f'Read: {knp_path}')
    reader = KyotoReader(knp_path)
    doc = reader.process_document(reader.doc_ids[0])
    return doc


def get_knp_morpheme_generator(doc: Document) -> Tuple[int, Morpheme]:
    for mid, mrph in enumerate(doc.mrph_list()):
        yield mid, mrph
    StopIteration


def get_mrph_list_repr(mrph_list: list[Morpheme]) -> str:
    return ' '.join([f'{m[0]}:{m[1].midasi}' for m in mrph_list])


def restore_orig_midasi(
        midasi: str,
        conv_exclamation: bool = False,
        conv_asterisk : bool = False,
        conv_plus: bool = False,
        conv_double_quote: bool = False,
) -> str:

    midasi = midasi.replace('␣', '　')
    if conv_exclamation:
        midasi = midasi.replace('！', '!')
    if conv_asterisk:
        midasi = midasi.replace('＊', '*')
    if conv_plus:
        midasi = midasi.replace('＋', '+')
    if conv_double_quote:
        midasi = midasi.replace("”", '"')

    return midasi


def get_sid2ne_list(doc: Document) -> dict:
    sid2ne_list = {}
    for ne in doc.named_entities:
        if not ne.sid in sid2ne_list:
            sid2ne_list[ne.sid] = []

        sen = doc.sid2sentence[ne.sid]
        mid_list = list(ne.mid_range)
        mid_begin = mid_list[0]
        mid_end = mid_list[-1] + 1
        char_span = get_char_span(sen, mid_begin, mid_end)
        sid2ne_list[ne.sid].append((ne.name, char_span, ne.category))

    return sid2ne_list


def get_char_span(
        sen: Sentence,
        mid_begin: int,
        mid_end: int,
) -> Tuple[int, int]:

    char_begin = 0
    char_end = 0

    for mid, mrph in enumerate(sen.mrph_list()):
        # print('*', mid, mrph.midasi)
        if mid < mid_begin:
            char_begin += len(mrph.midasi)
            char_end += len(mrph.midasi)
            # print('+a', char_begin, char_end)

        elif mid < mid_end:
            char_end += len(mrph.midasi)
            # print('+b', char_begin, char_end)

        elif mid == mid_end:
            break

    return (char_begin, char_end)


def get_sentences_aligned_with_original(
        knp_doc: Document,
        orig_sent_strs: list[str],
        debug: bool = False,
) -> list[list]:

    mrph_generator = get_knp_morpheme_generator(knp_doc)
    knp_sentences = []
    mrph_next = None

    for i, orig_sen_str in enumerate(orig_sen_strs):
        if debug:
            print(f'OrgSen-{i}:', orig_sen_str)

        knp_sen_current = ''
        knp_sentences.append([])

        while orig_sen_str != knp_sen_current:
            if mrph_next:
                mrph = mrph_next
                mid = mrph.mrph_id
                mrph_next = None
            else:
                mid, mrph = mrph_generator.__next__()
                mrph.midasi = restore_orig_midasi(
                    mrph.midasi,
                    conv_exclamation=(mrph.midasi == '！' and not '！' in orig_sen_str),
                    conv_asterisk=(mrph.midasi == '＊' and not '＊' in orig_sen_str),
                    conv_plus=(mrph.midasi == '＋' and not '＋' in orig_sen_str),
                    conv_double_quote=('"' in orig_sen_str),
                )

            if len(orig_sen_str) < len(knp_sen_current) + len(mrph.midasi):
                mrph_orig = mrph
                idx_split = len(orig_sen_str)-len(knp_sen_current)
                midasi1 = mrph_orig.midasi[:idx_split]
                midasi2 = mrph_orig.midasi[idx_split:]
                mrph = Morpheme(midasi1, mrph_id=mid)        # TMP: mid is tentative
                mrph_next = Morpheme(midasi2, mrph_id=mid+1) # TMP: mid is tentative

                if debug:
                    print('*', mrph_orig.midasi[:idx_split], mrph_orig.midasi[idx_split:])

            knp_sentences[-1].append((mid, mrph))
            knp_sen_current += mrph.midasi

            if debug:
                print('mrph:', mrph.midasi, '->', knp_sen_current)

        sen_repr = ' '.join([f'{m[0]}:{m[1].midasi}' for m in knp_sentences[-1]])

        if debug:
            print(f'{len(knp_sentences)-1}\t{sen_repr}')

        assert orig_sen_str == knp_sen_current

    return knp_sentences


def get_coref_results(
        knp_doc: Document, 
        knp_sentences_aligned: list[list], 
        debug: bool = False,
):
    ret_sentences = {}
    ret_entities = {}
    ret_mentions = {}
    men_id = 1

    dmid2sidx = {}

    for sidx, sen_mrph_list in enumerate(knp_sentences_aligned):
        ret_sid = f'{sidx+1:03d}'
        sen_str = ''.join([m[1].midasi for m in sen_mrph_list])
        ret_sentences[ret_sid] = {'text': sen_str, 'mention_ids': []}

        for mid, mrph in sen_mrph_list:
            dmid2sidx[mid] = sidx

    doc_mrph_list = knp_doc.mrph_list()

    if debug:
        print('====')
        print(' '.join([f'{i}:{mrph.midasi}' for i, mrph in enumerate(doc_mrph_list)]))
        print('====')

    for eid, ent in knp_doc.entities.items():
        if not ent.taigen:
            continue

        ret_eid = f'E{eid:03d}'
        ret_member_mids = []

        if debug:
            print(eid, ent, list(ent.mentions)[0])

        for men in ent.mentions:
            mrph = doc_mrph_list[men.content_dmid]
            dmid = men.content_dmid
            surf = men.core

            sidx = dmid2sidx[dmid]
            sen_mrph_list = knp_sentences_aligned[sidx]
            sen_str = ''.join([m[1].midasi for m in sen_mrph_list])
            if not surf in sen_str:
                logger.warning('Skipped a mention crossing sentence boundaries')
                continue

            if debug:
                print('-', men.sid, dmid, surf, mrph.midasi, '|', 
                      get_mrph_list_repr(sen_mrph_list))

            def get_span(tgt_dmid, tgt_surf, sen_mrph_list):
                surf_len_remain = len(tgt_surf)
                span_started = False
                begin_idx = end_idx = 0
                for mid, mrph in sen_mrph_list:
                    if surf_len_remain <= 0:
                        break

                    if mid < tgt_dmid:
                        end_idx += len(mrph.midasi)
                        if debug:
                            print('*', begin_idx, end_idx, surf_len_remain, 
                                  mid, mrph.midasi, mrph.hinsi)

                        begin_idx = end_idx

                    else: # mid >= tgt_dmid
                        end_idx += len(mrph.midasi)

                        if not span_started:
                            if (mrph.hinsi in ('助詞', '特殊', '判定詞', '') 
                                and not surf.startswith(mrph.midasi)
                            ):
                                begin_idx = end_idx
                                if debug:
                                    print('@*', begin_idx, end_idx, surf_len_remain, 
                                          mid, mrph.midasi, mrph.hinsi)
                            else:
                                span_started = True
                                surf_len_remain -= len(mrph.midasi)
                                if debug:
                                    print('@', begin_idx, end_idx, surf_len_remain, 
                                          mid, mrph.midasi, mrph.hinsi)
                        else:
                            surf_len_remain -= len(mrph.midasi)
                            if debug:
                                print('$', begin_idx, end_idx, surf_len_remain, 
                                      mid, mrph.midasi, mrph.hinsi)
                return [begin_idx, end_idx]

            span = get_span(dmid, surf, sen_mrph_list)
            assert sen_str[span[0]:span[1]] == surf, f"{sen_str}\n'{sen_str[span[0]:span[1]]}' != '{surf}'"

            ret_mid = f'M{men_id:03d}'
            ret_sid = f'{sidx+1:03d}'
            ret_mentions[ret_mid] = {
                'sentence_id'  : ret_sid,
                'entity_id'    : ret_eid,
                'span'         : span,
                'entity_label' : None,
                'text'         : surf,
            }

            ret_member_mids.append(ret_mid)

            if not ret_mid in ret_sentences[ret_sid]['mention_ids']:
                ret_sentences[ret_sid]['mention_ids'].append(ret_mid)

            men_id += 1

        if ret_member_mids:
            ret_entities[ret_eid] = {
                'normalized_name': ret_mentions[ret_member_mids[0]]['text'],
                'entity_label_merged': None,
                'member_mention_ids': ret_member_mids,
            }

    doc_json = {
        'sentences': ret_sentences,
        'mentions' : ret_mentions,
        'entities' : ret_entities,
    }
    
    return doc_json


def get_ne_results(
        knp_doc: Document,
        knp_sentences_aligned: list[list],
        debug: bool = False,
) -> list[list]:

    n_sens = len(knp_sentences_aligned)

    sen_idx = 0
    ne_seq_list = [[]]
    mrph_seq = knp_sentences_aligned[sen_idx]
    head_mid = mrph_seq[0][0]
    tail_mid = mrph_seq[-1][0]

    for ne in knp_doc.named_entities:
        dmid_list = list(ne.dmid_range)
        dmid_begin = dmid_list[0]
        dmid_end = dmid_list[-1] + 1
        ne.name = restore_orig_midasi(ne.name)

        if debug:
            print(ne.name, dmid_begin, dmid_end, head_mid, tail_mid, tail_mid <= dmid_begin)

        while tail_mid < dmid_begin and sen_idx < n_sens:
            sen_idx += 1
            ne_seq_list.append([])
            mrph_seq = knp_sentences_aligned[sen_idx]
            head_mid = mrph_seq[0][0]
            tail_mid = mrph_seq[-1][0]

        sen_repr = ' '.join([f'{m[0]}:{m[1].midasi}' for m in mrph_seq])

        if debug:
            print(head_mid, tail_mid, '|', sen_repr)

        char_begin = sum([len(m[1].midasi) for m in mrph_seq if m[0] < dmid_begin])
        char_end = (char_begin 
                    +sum([len(m[1].midasi) for m in mrph_seq if dmid_begin <= m[0] < dmid_end]))
        sen_str = ''.join([m[1].midasi for m in mrph_seq])
        ne_reext = sen_str[char_begin:char_end]

        if debug:
            print(f'Sen-{sen_idx}', sen_str)
            print('NE ', ne.name, ne_reext)

        if ne.name.startswith(ne_reext) and len(ne.name) > len(ne_reext):
            # TENTATIVE: 原文境界をまたぐNEを分割して登録
            ne_name_orig = ne.name
            ne.name = ne_reext
            ne_seq_list[sen_idx].append((ne.name, (char_begin, char_end), ne.category))

            ne_name_remain = ne_name_orig[len(ne.name):]
            sen_idx += 1            
            ne_seq_list.append([])
            mrph_seq = knp_sentences_aligned[sen_idx]
            head_mid = mrph_seq[0][0]
            tail_mid = mrph_seq[-1][0]
            char_begin = sum([len(m[1].midasi) for m in mrph_seq if m[0] < dmid_begin])
            char_end = (char_begin 
                        +sum([len(m[1].midasi) for m in mrph_seq if dmid_begin <= m[0] < dmid_end]))
            sen_str = ''.join([m[1].midasi for m in mrph_seq])

            if debug:
                print(f'Sen-{sen_idx}', sen_str)
                print(f'NE ', ne_name_remain, sen_str[char_begin:char_end], '$')

            assert ne_name_remain == sen_str[char_begin:char_end]
            ne_seq_list[sen_idx].append((ne_name_remain, (char_begin, char_end), ne.category))
        else:
            assert ne.name == sen_str[char_begin:char_end]
            ne_seq_list[sen_idx].append((ne.name, (char_begin, char_end), ne.category))

    return ne_seq_list


def output_sentence_with_nes(
        output_path: str,
        knp_sentences_aligned: list[list],
        ne_seq_list: list[list],
        doc_id: str = None,
):
    # TODO 括弧表現の調整
    with open(output_path, 'w', encoding='utf-8') as fw:
        for i, (mrph_seq, ne_seq) in enumerate(zip(knp_sentences_aligned, ne_seq_list)):
            sen_str = ''.join([m[1].midasi for m in mrph_seq])        
            ne_seq_str = ';'.join([f'_,{ne[1][0]}:{ne[1][1]},{ne[2]}' for ne in ne_seq])
            line = doc_id if doc_id else ''
            line = f'{line}\t{i+1}\t{sen_str}\t{ne_seq_str}\n'
            fw.write(line)

    logger.info(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text_path', '-t',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--input_knp_path', '-k',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--mode', '-m', 
        type=str,
        required=True,
        choices=('ne', 'coref'),
    )
    args = parser.parse_args()
    doc_id = args.input_text_path.split('/')[-1].split('.txt')[0]
    
    orig_sen_strs = read_sentences(args.input_text_path)
    knp_doc = read_doc(args.input_knp_path)
    knp_sentences_aligned = get_sentences_aligned_with_original(knp_doc, orig_sen_strs) #, True)
    assert len(orig_sen_strs) == len(knp_sentences_aligned)

    if args.mode == 'ne':
        ne_seq_list = get_ne_results(knp_doc, knp_sentences_aligned)
        output_sentence_with_nes(
            args.output_path, knp_sentences_aligned, ne_seq_list, doc_id=doc_id)

    elif args.mode == 'coref':
        doc = get_coref_results(knp_doc, knp_sentences_aligned)
        data = {doc_id: doc}

        with open(args.output_path, 'w', encoding='utf-8') as fw:
            json.dump(data, fw, ensure_ascii=False, indent=2)
        logger.info(f'Saved: {args.output_path}')
