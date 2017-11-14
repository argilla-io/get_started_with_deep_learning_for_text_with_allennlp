#!/usr/bin/env python
import logging
import os
import sys

from allennlp.commands import main  # pylint: disable=wrong-import-position


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

''' TODO: We should find a way to introspect Predictor and create this mapping automatically
    keeping naming consistent between model and predictor
'''
predictors = {
    'sequence_classifier': 'sequence_classifier'
}

main(predictor_overrides=predictors)