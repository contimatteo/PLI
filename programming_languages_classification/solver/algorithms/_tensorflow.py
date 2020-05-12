# /usr/bin/env python3

from ._base import _BaseAlgorithm
from utils import FileManager
from keras.models import load_model


class _TensorflowAlgorithm(_BaseAlgorithm):

    def importTrainedModel(self):
        self.model = None
        self.model = load_model(FileManager.getTrainedModelFileUrl(self.type))
        return self

    def exportTrainedModel(self):
        self.model.save(FileManager.getTrainedModelFileUrl(self.type))
        return self
