from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from datamin.bucketization import Bucketization
from datamin.classifiers.classifier import Classifier
from datamin.dataset import FolktablesDataset
from datamin.encoding_utils import encode_loader
from datamin.utils.utils import Stat


class SplitStrategy:
    def __init__(self, clf_iter: int, clf_lr: float, clf_model: str = "mlp2") -> None:
        self.clf_iter = clf_iter
        self.clf_lr = clf_lr
        self.clf_model = clf_model

    def get_next_split(
        self,
        bucketizations: List[Tuple[FolktablesDataset, Bucketization]],
        model: Classifier,
    ) -> None:

        accs = []
        stats = []
        # Basic strategy, split the feature with the highest uncertainty
        for data, bucketization in bucketizations:

            # Prepare model

            clf_train_loader = encode_loader(
                data.train_loader, bucketization, shuffle=True
            )
            clf_val_loader = encode_loader(data.val_loader, bucketization)

            nb_fts = clf_train_loader.dataset.tensors[0].shape[1]  # type: ignore[attr-defined]
            clf = Classifier(model.device, self.clf_model, nb_fts)
            clf.fit(
                clf_train_loader,
                self.clf_iter,
                self.clf_lr,
                tune_wd=True,
                val_loader=clf_val_loader,
            )

            total_acc, bucket_stats = self.stats_over_buckets(
                data, bucketization, model
            )

            accs.append(total_acc)
            stats.append(bucket_stats)

            # print(total_acc)
            # for feat, val in bucket_stats.items():
            #     str = ""
            #     for buck in val:
            #         str += f" FEAT: {feat} - Class 0: {buck[0].n} Class 1: {buck[1].n} || "
            #     print(str)
            # print("Done")

        # Updates
        # Emulate chosen split
        # curr_bucketization.add_cont("AGEP", sz=4, borders=[0.25, 0.375, 0.5])
        # curr_bucketization.add_cont("SCHL", sz=4, borders=[0.739, 0.8, 0.87])
        # curr_bucketization.add_cont("WKHP", sz=4, borders=[0.316, 0.35, 0.398])

    def stats_over_buckets(
        self, data: FolktablesDataset, bucketization: Bucketization, model: Classifier
    ) -> Tuple[float, Dict[str, List[Tuple[Stat, Stat]]]]:

        # Predict the data
        data_loader = encode_loader(data.train_loader, bucketization)
        with torch.no_grad():
            bucket_stats: Dict[str, List[Tuple[Stat, Stat]]] = {}
            for feat in data.feature_names:
                bucket_stats[feat] = []
                size = bucketization.buckets[feat]["size"]
                assert isinstance(size, int)
                for _ in range(size):
                    bucket_stats[feat].append((Stat(), Stat()))

            tot_clf_loss, tot_clf_acc = Stat(), Stat()
            for z, y in data_loader:
                z, y = z.to(model.device), y.to(model.device)
                clf_preds = model.predict(z)
                clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                clf_loss = F.cross_entropy(clf_preds, y)
                tot_clf_loss += (clf_loss.item(), z.shape[0])
                tot_clf_acc += (clf_acc.item(), z.shape[0])

                # Score over buckets
                bucket_mappings = bucketization.get_indices(
                    z
                )  # Tensor num_samples x num_features

                for (z_indices, y_i) in zip(bucket_mappings, y):
                    for feat, z_index in zip(data.feature_names, z_indices):
                        bucket_stats[feat][int(z_index)][int(y_i)] += 1  # type: ignore[index]

            # print(tot_clf_loss.avg(), tot_clf_acc.avg())
            return tot_clf_acc.avg(), bucket_stats
