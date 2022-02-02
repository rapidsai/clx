import cudf
from cuml.model_selection import train_test_split
import torch
import torch.optim as torch_optim
import torch.nn.functional as F
import logging
from torch.utils.dlpack import from_dlpack
from clx.analytics.model.tabular_model import TabularModel

log = logging.getLogger(__name__)


class AssetClassification:
    """
    Supervised asset classification on tabular data containing categorical and/or continuous features.

    :param layers: linear layer follow the input layer
    :param drops: drop out percentage
    :param emb_drop: drop out percentage at embedding layers
    :param is_reg: is regression
    :param is_multi: is classification
    :param use_bn: use batch normalization
    """

    def __init__(self, layers=[200, 100], drops=[0.001, 0.01], emb_drop=0.04, is_reg=False, is_multi=True, use_bn=True):
        self._layers = layers
        self._drops = drops
        self._emb_drop = emb_drop
        self._is_reg = is_reg
        self._is_multi = is_multi
        self._use_bn = use_bn
        self._device = None
        self._model = None
        self._optimizer = None
        self._cat_cols = None
        self._cont_cols = None
        self._device = torch.device('cuda')

    def train_model(self, train_gdf, cat_cols, cont_cols, label_col, batch_size, epochs, lr=0.01, wd=0.0):
        """
        This function is used for training fastai tabular model with a given training dataset.

        :param train_gdf: training dataset with categorized and/or continuous feature columns
        :type train_gdf: cudf.DataFrame
        :param cat_cols: array of categorical column names in train_gdf
        :type label_col: array
        :param cont_col: array of continuous column names in train_gdf
        :type label_col: array
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
        >>> cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        >>> cont_cols = ["10"]
        >>> ac.train_model(X_train, cat_cols, cont_cols, "label", batch_size, epochs, lr=0.01, wd=0.0)
        """

        self._cat_cols = cat_cols
        self._cont_cols = cont_cols

        # train/test split
        X, val_X, Y, val_Y = train_test_split(train_gdf, label_col, train_size=0.9)
        val_X.index = val_Y.index
        X.index = Y.index

        embedded_cols = {}
        for col in cat_cols:
            if col != label_col:
                categories_cnt = X[col].max() + 2
                if categories_cnt > 1:
                    embedded_cols[col] = categories_cnt

        X[label_col] = Y
        val_X[label_col] = val_Y

        # Embedding
        embedding_sizes = [(n_categories, min(100, (n_categories + 1) // 2)) for _, n_categories in embedded_cols.items()]

        n_cont = len(cont_cols)
        out_sz = train_gdf[label_col].nunique()

        # Partition dataframes
        train_part_dfs = self._get_partitioned_dfs(X, batch_size)
        val_part_dfs = self._get_partitioned_dfs(val_X, batch_size)

        self._model = TabularModel(embedding_sizes, n_cont, out_sz, self._layers, self._drops, self._emb_drop, self._is_reg, self._is_multi, self._use_bn)
        self._to_device(self._model, self._device)
        self._config_optimizer()
        for i in range(epochs):
            loss = self._train(self._model, self._optimizer, train_part_dfs, cat_cols, cont_cols, label_col)
            print("training loss: ", loss)
            self._val_loss(self._model, val_part_dfs, cat_cols, cont_cols, label_col)

    def predict(self, gdf, cat_cols, cont_cols):
        """
        Predict the class with the trained model

        :param gdf: prediction input dataset with categorized int16 feature columns
        :type gdf: cudf.DataFrame
        :param cat_cols: array of categorical column names in gdf
        :type label_col: array
        :param cont_col: array of continuous column names in gdf
        :type label_col: array

        Examples
        --------
        >>> cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        >>> cont_cols = ["10"]
        >>> ac.predict(X_test, cat_cols, cont_cols).values_host
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
        cat_set = torch.zeros(0, 0)
        xb_cont_tensor = torch.zeros(0, 0)

        if cat_cols:
            cat_set = gdf[self._cat_cols].to_dlpack()
            cat_set = from_dlpack(cat_set).long()
        if cont_cols:
            xb_cont_tensor = gdf[self._cont_cols].to_dlpack()
            xb_cont_tensor = from_dlpack(xb_cont_tensor).float()

        out = self._model(cat_set, xb_cont_tensor)
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
        >>> cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        >>> cont_cols = ["10"]
        >>> ac.train_model(X_train, cat_cols, cont_cols, "label", batch_size, epochs, lr=0.01, wd=0.0)
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
        >>> ac.load_model("ac.mdl")
        """
        self._model = torch.load(fname)

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

    def _train(self, model, optim, dfs, cat_cols, cont_cols, label_col):
        self._model.train()
        total = 0
        sum_loss = 0

        cat_set = torch.zeros(0, 0)
        xb_cont_tensor = torch.zeros(0, 0)

        for df in dfs:
            batch = df.shape[0]
            if cat_cols:
                cat_set = df[cat_cols].to_dlpack()
                cat_set = from_dlpack(cat_set).long()
            if cont_cols:
                xb_cont_tensor = df[cont_cols].to_dlpack()
                xb_cont_tensor = from_dlpack(xb_cont_tensor).float()

            output = self._model(cat_set, xb_cont_tensor)
            train_label = df[label_col].to_dlpack()
            train_label = from_dlpack(train_label).long()

            loss = F.cross_entropy(output, train_label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())

        return sum_loss / total

    def _val_loss(self, model, dfs, cat_cols, cont_cols, label_col):
        self._model.eval()
        total = 0
        sum_loss = 0
        correct = 0

        val_set = torch.zeros(0, 0)
        xb_cont_tensor = torch.zeros(0, 0)

        for df in dfs:
            current_batch_size = df.shape[0]

            if cat_cols:
                val_set = df[cat_cols].to_dlpack()
                val_set = from_dlpack(val_set).long()
            if cont_cols:
                xb_cont_tensor = df[cont_cols].to_dlpack()
                xb_cont_tensor = from_dlpack(xb_cont_tensor).float()

            out = self._model(val_set, xb_cont_tensor)

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
