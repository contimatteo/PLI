
class PLCExamplesDataset:

    datasetFolderPath: str = "../../datasets"
    datasetFolderName: str = "rosetta-code"

    #

    def load(self):
        print(self.datasetFolderPath + "/" + self.datasetFolderName)
