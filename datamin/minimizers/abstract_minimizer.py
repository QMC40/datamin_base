from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset


class AbstractMinimizer:
    def __init__(self) -> None:
        pass

    def fit(self, dataset: FolktablesDataset) -> None:
        raise NotImplementedError("No fit implemented")

    def get_bucketization(self) -> Bucketization:
        raise NotImplementedError("No get_bucketization implemented")
