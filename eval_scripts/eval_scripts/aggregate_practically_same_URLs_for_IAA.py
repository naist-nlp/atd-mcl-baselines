import argparse


def read_and_write(input_path: str, output_path: str):
    f_prac_same = False
    org_url1 = org_url2 = None
    merged_urls = []

    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if line.startswith('doc_id'):
                array = line.split('\t')
                f_prac_same_str = array[3].split(':')[1]
                f_prac_same = f_prac_same_str == 'T'
                org_ur1 = org_url2 = None

            elif line.startswith('-'):
                if not f_prac_same:
                    continue

                array = line.split('\t')
                url = array[1]
                utype = array[2]

                if org_url1 is None:
                    org_url1 = url
                    org_url2 = None

                elif org_url2 is None:
                    org_url2 = url
                    urls = set([org_url1, org_url2]) | set(org_url1.split(';')) | set(org_url2.split(';'))
                    flag_to_add = False
                    for i, url_set in enumerate(merged_urls):
                        for u in urls:
                            if u in url_set:
                                flag_to_add = True
                                break
                        if flag_to_add:
                            url_set |= urls
                            break
  
                    if not flag_to_add:
                        merged_urls.append(urls)
  
                    org_url1 = org_url2 = None

    with open(output_path, 'w', encoding='utf-8') as fw:
        for murls in sorted(merged_urls):
            to_url = f'[{";".join(sorted(murls))}]'
            for url in murls:
                fw.write(f'{url}\t{to_url}\n')


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
        default=None,
    )
    args = parser.parse_args()

    read_and_write(args.input_path, args.output_path)
