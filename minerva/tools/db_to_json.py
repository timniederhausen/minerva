import sys
import os
from contextlib import closing

from minerva.metadata.db import MetadataConnection
from minerva.metadata.json import save


def main(uri, path):
    try:
        os.makedirs(path)
    except OSError:
        pass

    with closing(MetadataConnection(uri)) as conn:
        for ds in conn.load_all():
            save(ds, os.path.join(path, ds['name'] + '.json'))


# usage: python -m minerva.tools.db_to_json postgresql://minerva:1@localhost/minerva _meta2
main(*sys.argv[1:])
