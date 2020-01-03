import json
import os
import collections
import typing

from ConfigSpace.configuration_space import Configuration
from smac.utils.io.traj_logging import TrajLogger

from minerva import calculate_metafeatures
from minerva.autoencoder import make_config_space
from minerva.util import LoadedDataset, filename_to_dataset_name


def _serialize(v):
    if isinstance(v, Configuration):
        return v.get_dictionary()
    return str(v)


def _deserialize_mf(mf: list):
    # format: name, type_, fold, repeat, value, time, comment
    value = mf[4]
    if value is None:
        pass
    elif value == '?':
        value = None
    elif isinstance(value, collections.Iterable):
        value = [float(v) for v in value]
    else:
        value = float(value)
    mf[4] = value
    return mf


def build_metadata_file(dataset: LoadedDataset, filename: str, smac_dir: str,
                        modifier: typing.Callable[[dict], None] = None):
    try:
        with open(filename) as fp:
            metadata: dict = json.load(fp)
    except FileNotFoundError:
        metadata = {}

    name = filename_to_dataset_name(dataset.filename)
    metadata['name'] = name
    metadata['x_shape'] = dataset.x.shape
    metadata['y_shape'] = dataset.y.shape

    if 'autoencoder' not in metadata:
        try:
            cs = make_config_space(dataset.x.shape)
            trajectory = TrajLogger.read_traj_aclib_format(
                fn=os.path.join(smac_dir, name, 'run_1', 'traj_aclib2.json'),
                cs=cs)
            metadata['autoencoder'] = dict(trajectory=trajectory)
            print('Loaded trajectory for {}: n = {}'.format(name, len(trajectory)))
        except FileNotFoundError:
            print('Couldn\'t find SMAC run info for {} yet!'.format(name))

    if 'metafeatures' not in metadata:
        print('Calculating metafeatures for ' + name)
        metafeatures = calculate_metafeatures(name, dataset.x, dataset.y)
        metadata['metafeatures'] = metafeatures.dumps()

    # Fix name if it's broken (i.e. filename instead of dataset name)
    metadata['metafeatures']['relation'] = name

    if modifier:
        modifier(metadata)

    save(metadata, filename)


def load(filename_or_uri: str) -> dict:
    if filename_or_uri.startswith('file:///'):
        filename_or_uri = filename_or_uri[8:]

    with open(filename_or_uri) as fp:
        metadata = json.load(fp)
        if 'metafeatures' in metadata:
            metafeatures = metadata['metafeatures']
            metafeatures['data'] = list(map(_deserialize_mf, metafeatures['data']))
        return metadata


def load_all(filename_or_uri: str) -> typing.List[dict]:
    if filename_or_uri.startswith('file:///'):
        filename_or_uri = filename_or_uri[8:]

    if os.path.isdir(filename_or_uri):
        return [load(os.path.join(filename_or_uri, f)) for f in os.listdir(filename_or_uri)]
    else:
        return [load(filename_or_uri)]


def save(metadata: dict, filename_or_uri: str):
    if filename_or_uri.startswith('file:///'):
        filename_or_uri = filename_or_uri[8:]

    with open(filename_or_uri, 'w') as f:
        json.dump(metadata, f, default=_serialize, indent=2)
