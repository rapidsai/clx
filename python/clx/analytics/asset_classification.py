import cudf
from cuml.preprocessing.model_selection import train_test_split
from cuml.preprocessing import LabelEncoder
import torch
import torch.optim as torch_optim
import torch.nn.functional as F
import logging
from torch.utils.dlpack import from_dlpack
from clx.analytics.model.tabular_model import TabularModel

log = logging.getLogger(__name__)


class AssetClassification:
    """
    This class provides multiple functionalities such as build, train and evaluate the RNNClassifier model
    to distinguish legitimate and DGA domain names.
    """

    def __init__(self, n_cont=0, out_sz=6, layers=[200, 100], drops=[0.001, 0.01], emb_drop=0.04, is_reg=False, is_multi=True, use_bn=True):
        self._n_cont = n_cont
        self._out_sz = out_sz
        self._layers = layers
        self._drops = drops
        self._emb_drop = emb_drop
        self._is_reg = is_reg
        self._is_multi = is_multi
        self._use_bn = use_bn
        self._device = None
        self._model = None
        self._optimizer = None
        self._device = torch.device('cuda')

    def train_model(self, train_gdf, label_col, batch_size, epochs, lr=0.01, wd=0.0):
        """
        This function is used for training fastai tabular model with a given training dataset.

        :param train_gdf: training dataset with categorized int16 feature columns
        :type train_gdf: cudf.DataFrame
        :param label_col: column name of label column in train_gdf
        :type label_col: str
        :param batch_size: train_gdf will be partitioned into multiple dataframes of this size
        :type batch_size: int
        :param epochs: number of epochs to be adjusted depending on convergence for a specific dataset
        :type epochs: int
        :param lr: learning rate
        :type lr: float
        :param wd: wd
        :type wd: float

        Examples
        --------
        >>> from clx.analytics.asset_classification import AssetClassification
        >>> ac = AssetClassification()
        >>> ac.train_model(X_train, "label", batch_size, epochs, lr=0.01, wd=0.0)
        """

        X, val_X, Y, val_Y = train_test_split(train_gdf, label_col, train_size=0.9)
        val_X.index = val_Y.index

        X.index = Y.index

        embedded_cols = {}
        for col in X.columns:
            categories_cnt = X[col].max() + 2
            if categories_cnt > 1:
                embedded_cols[col] = categories_cnt

        X[label_col] = Y
        val_X[label_col] = val_Y

        # Embedding
        embedding_sizes = [(n_categories, min(100, (n_categories + 1) // 2)) for _, n_categories in embedded_cols.items()]

        # Partition dataframes
        train_part_dfs = self._get_partitioned_dfs(X, batch_size)
        val_part_dfs = self._get_partitioned_dfs(val_X, batch_size)

        self._model = TabularModel(embedding_sizes, self._n_cont, self._out_sz, self._layers, self._drops, self._emb_drop, self._is_reg, self._is_multi, self._use_bn)
        self._to_device(self._model, self._device)
        self._config_optimizer()
        for i in range(epochs):
            loss = self._train(self._model, self._optimizer, train_part_dfs, label_col)
            print("training loss: ", loss)
            self._val_loss(self._model, val_part_dfs, label_col)

    def predict(self, gdf):
        """
        Predict the class with the trained model

        :param gdf: prediction input dataset with categorized int16 feature columns
        :type gdf: cudf.DataFrame

        Examples
        --------
        >>> ac.predict(X_test).to_array()
        0       0
        1       0
        2       0
        3       0
        4       2
            ..
        8204    0
        8205    4
        8206    0
        8207    3
        8208    0
        Length: 8209, dtype: int64
        """
        xb_cont_tensor = torch.zeros(0, 0)
        xb_cont_tensor.cuda()

        df = df.to_dlpack()
        df = from_dlpack(df).long()

        out = self._model(df, xb_cont_tensor)
        preds = torch.max(out, 1)[1].view(-1).tolist()

        return cudf.Series(preds)

    def save_model(self, fname):
        """
        Save trained model

        :param save_to_path: directory path to save model
        :type save_to_path: str

        Examples
        --------
        >>> from clx.analytics.asset_classification import AssetClassification
        >>> ac = AssetClassification()
        >>> ac.train_model(X_train, "label", batch_size, epochs, lr=0.01, wd=0.0)
        >>> ac.save_model("ac.mdl")
        """
        torch.save(self._model, fname)

    def load_model(self, fname):
        """
        Load a saved model.

        :param fname: directory path to model
        :type fname: str

        Examples
        --------
        >>> from clx.analytics.asset_classification import AssetClassification
        >>> ac = AssetClassification()
        >>> ac.train_model(X_train, "label", batch_size, epochs, lr=0.01, wd=0.0)
        >>> ac.load_model("ac.mdl")
        """
        self._model = torch.load(fname)

    def categorize_columns(self, gdf):
        """
        Categorize feature columns of input dataset and convert to numeric (int16). This is required format for train_model input.

        :param gdf: dataset to be categorized
        :type gdf: cudf.DataFrame
        """
        cat_gdf = gdf.copy()
        for col in cat_gdf.columns:
            cat_gdf[col] = cat_gdf[col].astype('str')
            cat_gdf[col] = cat_gdf[col].fillna("NA")
            cat_gdf[col] = LabelEncoder().fit_transform(cat_gdf[col])
            cat_gdf[col] = cat_gdf[col].astype('int16')

        return cat_gdf

    def _config_optimizer(self, lr=0.001, wd=0.0):
        parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        self._optimizer = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)

    def _get_partitioned_dfs(self, df, batch_size):
        dataset_len = df.shape[0]
        prev_chunk_offset = 0
        partitioned_dfs = []
        while prev_chunk_offset < dataset_len:
            curr_chunk_offset = prev_chunk_offset + batch_size
            chunk = df.iloc[prev_chunk_offset:curr_chunk_offset:1]
            partitioned_dfs.append(chunk)
            prev_chunk_offset = curr_chunk_offset
        return partitioned_dfs

    def _train(self, model, optim, dfs, label_col):
        self._model.train()
        total = 0
        sum_loss = 0

        xb_cont_tensor = torch.zeros(0, 0)
        xb_cont_tensor.cuda()

        for df in dfs:
            batch = df.shape[0]
            train_set = df.drop(label_col).to_dlpack()
            train_set = from_dlpack(train_set).long()

            output = self._model(train_set, xb_cont_tensor)
            train_label = df[label_col].to_dlpack()
            train_label = from_dlpack(train_label).long()

            loss = F.cross_entropy(output, train_label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())

        return sum_loss / total

    def _val_loss(self, model, dfs, label_col):
        self._model.eval()
        total = 0
        sum_loss = 0
        correct = 0

        xb_cont_tensor = torch.zeros(0, 0)
        xb_cont_tensor.cuda()

        for df in dfs:
            current_batch_size = df.shape[0]

            val = df.drop(label_col).to_dlpack()
            val = from_dlpack(val).long()

            out = self._model(val, xb_cont_tensor)

            val_label = df[label_col].to_dlpack()
            val_label = from_dlpack(val_label).long()

            loss = F.cross_entropy(out, val_label)
            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size

            pred = torch.max(out, 1)[1]
            correct += (pred == val_label).float().sum().item()
        print("valid loss %.3f and accuracy %.3f" % (sum_loss / total, correct / total))

        return sum_loss / total, correct / total

    def _to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self._to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
