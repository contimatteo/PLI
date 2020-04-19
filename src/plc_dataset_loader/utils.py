
class PLExamplesDataset:

    datasetFolderPath: str = "../../datasets"
    datasetFolderName: str = "rosetta-code"

    # A sample method
    def load(self):
        print(self.datasetFolderPath + "/" + self.datasetFolderName)
