from torch.utils.data import Dataset

# movielens 1k

class trip(Dataset):

    def __init__(self,rt):
        super(Dataset,self).__init__()
        self.uId = list(rt['user_id'])
        self.iId = list(rt['listing_id'])
        self.rt = list(rt['rating'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])