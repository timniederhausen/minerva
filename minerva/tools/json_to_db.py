import sys
from contextlib import closing

from minerva.metadata.db import MetadataConnection
from minerva.metadata.json import load_all


def main(path, uri):
    with closing(MetadataConnection(uri)) as conn:
        for ds in load_all(path):
            conn.save(ds)


# usage: python -m minerva.tools.json_to_db _meta postgresql://minerva:1@localhost/minerva
main(*sys.argv[1:])
