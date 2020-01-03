import typing

from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from minerva.util import get_iterable

Base = declarative_base()


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_rows = Column(Integer, nullable=False)
    x_columns = Column(Integer, nullable=False)
    y_columns = Column(Integer, nullable=False)

    metafeatures = relationship('DatasetMetafeature', backref='dataset')  # type: typing.List[DatasetMetafeature]
    autoencoders = relationship('AutoencoderConfiguration',
                                backref='dataset')  # type: typing.List[AutoencoderConfiguration]

    # Some tools might want to store additional data.
    # This field allows them to do so.
    # (In case of this bachelor thesis, certain data mining statistics are stored here)
    extra = Column(JSONB)

    def __repr__(self):
        return '<Dataset(name="{name}", |x|=({x_rows},{x_columns}), |y|={y_columns}>'.format(**self.__dict__)


class AutoencoderConfiguration(Base):
    __tablename__ = 'autoencoder_config'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.id'))
    wallclock_training_time = Column(Float)
    cost = Column(Float)
    origin = Column(String)
    incumbent = Column(JSONB)

    def __repr__(self):
        return '<AE(ds={dataset.name}, origin={origin}, cost={cost}, incumbent={incumbent}>'.format(**self.__dict__)


class DatasetMetafeature(Base):
    __tablename__ = 'dataset_metafeature'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.id'))
    name = Column(String)
    value = Column(ARRAY(Float), nullable=True)
    wallclock_time = Column(Float)
    use_for_comparison = Column(Boolean)

    def __repr__(self):
        return '<MF(ds={dataset.name}, name={name}, value={value}, is_mf={use_for_comparison}>'.format(**self.__dict__)


Session = sessionmaker()


def _convert_dataset(ds: Dataset):
    metadata = dict(name=ds.name,
                    x_shape=(ds.x_rows, ds.x_columns),
                    y_shape=(ds.x_rows, ds.y_columns),
                    autoencoder=dict(trajectory=[]),
                    metafeatures=dict(data=[]))
    if ds.extra:
        metadata.update(ds.extra)

    for ae in ds.autoencoders:
        metadata['autoencoder']['trajectory'].append(dict(
            # Set those to zero, we never got that info anyway (not supported on all our target OSes)
            cpu_time=0,
            total_cpu_time=None,
            wallclock_time=ae.wallclock_training_time,
            evaluations=0,  # not needed
            cost=ae.cost,
            origin=ae.origin,
            incumbent=ae.incumbent
        ))

    for mf in ds.metafeatures:
        metadata['metafeatures']['data'].append((
            mf.name,
            'METAFEATURE' if mf.use_for_comparison else 'HELPERFUNCTION',
            0, 0,
            mf.value if not mf.value or len(mf.value) > 1 else mf.value[0],
            mf.wallclock_time, ''
        ))

    return metadata


class MetadataConnection(object):
    r"""Persistent connection to a single PostgreSQL database.

    This class allows the user to re-use one session for different operations (load all, save one, ...)
    """

    def __init__(self, uri: str):
        r"""Construct a new persistent MetaDB connection.

        If the schema doesn't exist yet, it will be created.
        Only the database itself must exist when calling this function.

        :param uri: The URI of the PostgreSQL database to use.
        """
        assert uri.startswith('postgresql://') or uri.startswith('postgresql+')

        self.engine = create_engine(uri, client_encoding='utf8')

        # Ensure our schema exists and has all the necessary objects.
        Base.metadata.create_all(self.engine)

        self.session = Session(bind=self.engine)

    def close(self):
        r"""Close the session, releasing its resources"""
        if self.session:
            self.session.close()
            self.session = None

        self.engine.dispose()

    def save(self, metadata: dict):
        r"""Save a single dataset metadata dict to the database.

        Note that this does not override any existing entries.

        :param metadata: The dataset's metadata object.
        """
        ds = Dataset()
        ds.name = metadata.pop('name')

        x_shape = metadata.pop('x_shape')
        ds.x_rows = x_shape[0]
        ds.x_columns = x_shape[-1]

        y_shape = metadata.pop('y_shape')
        ds.y_columns = y_shape[-1] if len(y_shape) > 1 else 1

        autoencoder = metadata.pop('autoencoder')
        metafeatures = metadata.pop('metafeatures')
        ds.extra = metadata
        self.session.add(ds)

        for ae_dict in autoencoder['trajectory']:
            ae = AutoencoderConfiguration()
            ae.wallclock_training_time = ae_dict['wallclock_time']
            ae.cost = ae_dict['cost']
            ae.origin = ae_dict['origin']
            ae.incumbent = ae_dict['incumbent']
            ae.dataset = ds
            self.session.add(ae)

        for (name, type_, fold, repeat, value, time, comment) in metafeatures['data']:
            mf = DatasetMetafeature()
            mf.name = name
            mf.value = get_iterable(value if value != '?' else None)
            mf.wallclock_time = time
            mf.use_for_comparison = type_ == 'METAFEATURE'
            mf.dataset = ds
            self.session.add(mf)

        self.session.commit()

    def load_all(self):
        return [_convert_dataset(ds) for ds in self.session.query(Dataset)]
