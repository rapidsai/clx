from torch.utils.data import Dataset


class DGADataset(Dataset):
    def __init__(self):
        self.domains = None
        self.types = None
        self.len = 0
        self.type_list = None

    def __getitem__(self, index):
        return self.domains[index], self.types[index]

    def __len__(self):
        return self.len

    def get_type_id(self, type):
        return self.type_list.index(type)

    def get_type(self, id):
        return self.type_list[id]

    def set_attributes(self, rows):
        self.domains = [row[0] for row in rows]
        self.types = [row[1] for row in rows]
        self.len = len(self.types)
        self.type_list = list(sorted(set(self.types)))
