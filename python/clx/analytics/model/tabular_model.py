# Original code at https://github.com/spro/practical-pytorch
import torch
import torch.nn as nn


class TabularModel(nn.Module):
    "Basic model for tabular data"

    def __init__(self, emb_szs, n_cont, out_sz, layers, drops,
                 emb_drop, use_bn, is_reg, is_multi):
        super().__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True)] * (len(sizes) - 2) + [None]
        layers = []
        for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + drops, actns)):
            layers += self._bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.n_cont == 1:
                x_cont = x_cont.unsqueeze(1)
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        return x.squeeze()

    def _bn_drop_lin(self, n_in, n_out, bn, p, actn):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers
