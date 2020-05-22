# /usr/bin/env python3

from ._base import _BaseAlgorithm
from utils import FileManager
import joblib


class _ScikitLearnAlgorithm(_BaseAlgorithm):

    def importTrainedModel(self):
        self.model = None
        self.model = joblib.load(FileManager.getTrainedModelFileUrl(self.type))
        return self

    def exportTrainedModel(self):
        joblib.dump(self.model, FileManager.getTrainedModelFileUrl(self.type))
        return self
