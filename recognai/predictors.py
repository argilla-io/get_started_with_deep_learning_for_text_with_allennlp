from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('sequence_classifier')
class SequenceClassifierPredictor(Predictor):
    """
    Wrapper for the :class:`~custom.models.SequenceClassifier` model.
    """
    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"text": "..."}``.
        """
        input_text = json["text"]
        return self._dataset_reader.text_to_instance(input_text)
