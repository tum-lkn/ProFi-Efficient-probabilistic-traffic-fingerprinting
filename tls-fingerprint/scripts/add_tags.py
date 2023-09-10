import os
import pandas as pd
import json

import sqlalchemy
from typing import List, Tuple

with open("CONFIG.json", 'r') as fh:
    CONFIG = json.laod(fh)

SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:{CONFIG["db_password"]}' + \
                          f'{CONFIG["db_host"]}:3306/gatherer_upscaled'


def _insert_tags(url_ids: List[int], tag_id: int):
    cmd = 'INSERT INTO tag_assignment (tag_fk, trace_fk) VALUES '
    values = ', '.join([f'({tag_id}, {url_id})' for url_id in url_ids])
    cmd += values
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        connection.execute(cmd)
    print(cmd)


def get_url_ids(url_part: str) -> List[Tuple[str, int]]:
    q = f"SELECT url, id FROM urls WHERE urls.url like '%%{url_part}%%'"
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        res = connection.execute(q).fetchall()
    return res


def insert_tags(url_start: int, url_end: int, tag_id: int):
    _insert_tags(list(range(url_start, url_end)), tag_id)


def get_hostname(url: str) -> str:
    """
    Extract the hostname from an URL. URL has to start with https://.

    Args:
        url: URL that has been queried.

    Returns:
        hostname
    """
    # assert url.startswith('https://'), f"URL expected to start with https://, URL is {url}"
    hostname = url.lstrip('https://')
    hostname = hostname.lstrip('http://')
    idx = hostname.find('/')
    # assert idx > 0, f"Could not find slash in url {hostname}."
    if idx < 0:
        return hostname
    else:
        return hostname[:idx]


def get_hostnames():
    q = "SELECT id, url FROM urls"
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        urls = connection.execute(q).fetchall()
    hostnames = {}
    for url_id, url in urls:
        hostname = get_hostname(url)
        if hostname not in hostnames:
            hostnames[hostname] = {
                'hostname': hostname,
                'urls': [],
                'hoster': '',
                'adultery': False
            }
        hostnames[hostname]['urls'].append(url_id)
    lines = ['{};{};{};{}'.format(
        x['hostname'],
        x['hoster'],
        x['adultery'],
        ','.join([str(a) for a in x['urls']])) for x in hostnames.values()]
    with open('tbl.csv', 'w') as fh:
        fh.write(os.linesep.join(lines))


def _get_tag_id(tagname, connection):
    tag_id = None
    while tag_id is None:
        tag_id = connection.execute(f"SELECT id FROM tags WHERE tag like \"{tagname}\"").fetchall()
        if len(tag_id) > 0:
            tag_id = tag_id[0][0]
        else:
            connection.execute(f"INSERT INTO tags (tag) VALUES (\"{tagname}\")")
            tag_id = None
    return tag_id


def _insert_tag(url_ids: List[str], tag_id: int, connection):
    cmd = 'INSERT INTO tag_assignment (tag_fk, trace_fk) VALUES '
    values = ', '.join([f'({tag_id}, {url_id})' for url_id in url_ids])
    cmd += values
    connection.execute(cmd)


def insert_meta_table():
    df = pd.read_csv(
        '/opt/project/scripts/tbl_meta.csv',
        sep=';',
        names=['hostname', 'hoster', 'adult', 'url_ids']
    )
    engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        for i in range(df.shape[0]):
            url_ids = df.iat[i, 3].split(',')
            hostname_tag = _get_tag_id(df.iat[i, 0], connection)
            hoster_tag = _get_tag_id(df.iat[i, 1], connection)
            print(hoster_tag, df.iat[i, 0])
            empty_tag = 23
            adult_tag = 38 if df.iat[i, 2] else None
            _insert_tag(url_ids, empty_tag, connection)
            _insert_tag(url_ids, hostname_tag, connection)
            _insert_tag(url_ids, hoster_tag, connection)
            if adult_tag is not None:
                _insert_tag(url_ids, adult_tag, connection)


if __name__ == '__main__':
    pass
    # get_hostnames()
    # insert_meta_table()
    # step = 50
    # url_start, tag1, tag2 = 601, 24, 41
    # insert_tags(url_start, url_start + step, tag1)
    # insert_tags(url_start, url_start + step, tag2)
    # insert_tags(url_start, url_start + step, 38)
