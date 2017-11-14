from typing import Dict
import json
import logging

from overrides import overrides
import tqdm

import csv

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FieldPreparator():
    def __init__(self,
                 field: str = None,
                 mapping: Dict = {}):
        self._field = field
        self._mapping = mapping

    def transform(self, field, value) -> str:
        if field == self._field:
            return self._mapping.get(value, default=value)
        else:
            return value

    @classmethod
    def from_params(cls, params: Params) -> 'FieldPreparator':
        field = params.pop('field', None)
        mapping = params.pop('mapping', {}).as_dict()
        return FieldPreparator(field=field, mapping=mapping)


@DatasetReader.register("jsonl_classification_reader")
class JsonlClassificationReader(DatasetReader):
    """
    Reads a file from a classification dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 input: str,
                 gold_label: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 data_preparator: FieldPreparator = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._input = input
        self._gold_label = gold_label
        self._field_preparator = data_preparator
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as input_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(input_file):
                example = json.loads(line)
                input_text = example[self._input]
                # TODO: This ugly thing should be configurable or at least based on typechecking
                label = str(int(example[self._gold_label]))
                instances.append(self.text_to_instance(input_text, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         input_text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        if self._field_preparator:
            input_text = self._field_preparator.transform(self._input, input_text)
        input_tokens = self._tokenizer.tokenize(input_text)
        fields['tokens'] = TextField(input_tokens, self._token_indexers)
        if label:
            if self._field_preparator:
                label = self._field_preparator.transform(self._gold_label, label)
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'JsonlClassificationReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        input = params.pop('input', None)
        gold_label = params.pop('gold_label', None)
        data_preparator = FieldPreparator.from_params(params.pop('field_preparator', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return JsonlClassificationReader(tokenizer=tokenizer,
                                         token_indexers=token_indexers,
                                         input=input,
                                         gold_label=gold_label,
                                         data_preparator=data_preparator)


@DatasetReader.register("csv_classification_reader")
class CsvClassificationReader(DatasetReader):
    """
    Reads a file from a classification dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 input: int,
                 gold_label: int,
                 skip_header: bool = False,
                 delimiter: str = ",",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._input = input
        self._gold_label = gold_label
        self._skip_header = skip_header
        self._delimiter = delimiter
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as input_file:
            logger.info("Reading instances from CSV dataset at: %s", file_path)
            reader = csv.reader(input_file, delimiter=self._delimiter)
            if (self._skip_header):
                next(reader)
            # examples = [make_example(line, fields) for line in reader]
            for example in reader:
                input = example[self._input]
                label = example[self._gold_label]
                instances.append(self.text_to_instance(input, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         input: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        input_tokens = self._tokenizer.tokenize(input)
        fields['tokens'] = TextField(input_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'CsvClassificationReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        input = params.pop('pos_input', None)
        gold_label = params.pop('pos_gold_label', None)
        skip_header = params.pop('skip_header', None)
        delimiter = params.pop('delimiter', None)
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return CsvClassificationReader(tokenizer=tokenizer,
                                       token_indexers=token_indexers,
                                       skip_header=skip_header,
                                       delimiter=delimiter,
                                       input=input,
                                       gold_label=gold_label)
