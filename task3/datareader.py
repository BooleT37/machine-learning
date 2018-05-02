import re

import pandas as pd


def read_csv(filename, train, train_lines):
    with open(filename, encoding='utf-8') as f:
        df = pd.DataFrame(columns=['name', 'isOrg'])
        line_num = -1
        for line in f:
            if line_num == -1:
                line_num += 1
                continue
            if train_lines is not None and line_num >= train_lines:
                break
            match = re.match(r'^\d+', line)
            if match is None:
                raise ParsingError(f"Cannot parse line {line_num}: ""{line}""")
            if line_num != int(match[0]):
                raise ParsingError(f"line numbers are inconsistent: {line_num} vs {int(match[1])}")
            line_rest = line[len(match[0]):].strip()
            if train:
                name, is_org = get_is_org(line_rest, line_num)
                name = parse_quotes(name)
                df = df.append({'name': name, 'isOrg': is_org}, ignore_index=True)
            else:
                name = line_rest
                df = df.append({'name': name}, ignore_index=True)
            line_num += 1

            if line_num % 1000 == 0:
                print(f"Reading line {line_num}...")

    return df


def get_is_org(line, line_num):
    if line[-4:] == 'True':
        is_org = True
        name = line[:-4]
    elif line[-5:] == 'False':
        is_org = False
        name = line[:-5]
    else:
        raise ParsingError(f"line {line_num}: End of line should be either 'True' or 'False'!")
    name = name.rstrip()
    return name, is_org


def parse_quotes(s):
    if s[0] == '"':
        return s[1:-1].replace('""', '"')
    return s


class ParsingError(Exception):
    pass
