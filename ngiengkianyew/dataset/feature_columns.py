from dataclasses import dataclass
import torch
from collections import OrderedDict
from itertools import chain
import pandas as pd
import numpy as np

def gen_dataset():
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    # behavior_length = np.array([3, 3, 2])
    
    data ={'uid': uid,
      'ugender': ugender,
      'iid': iid,
      'igenre': igender,
      'score': score, 
      'hist_iid':hist_iid.tolist(),
      'hist_igenre': hist_igender.tolist(),
      'hist_length': [3,3,2],} # hist_length contains number of unpadded values


    df = pd.DataFrame(data)
    return df
def gen_feature_columns():
    sparse_features = [SparseFeat(name='uid',
                                vocabulary_size=3,
                                embedding_dim=3),
                        SparseFeat(name='iid',
                                vocabulary_size=4,
                                embedding_dim=3),
                        SparseFeat(name='ugender',
                                vocabulary_size=3,
                                embedding_dim=3),
                        SparseFeat(name='igenre',
                                   vocabulary_size=3,
                                   embedding_dim=3)]

    var_len_sparse_feat = [
        VarLenSparseFeat(
            sparse_feat=SparseFeat(name='hist_iid',
                                    vocabulary_size=4,
                                    embedding_dim=3),
            max_len=4,
            length_name='hist_length'),
        VarLenSparseFeat(
            sparse_feat=SparseFeat(name='hist_igenre',
                                    vocabulary_size=3,
                                    embedding_dim=3),
            max_len=4,
            length_name='hist_length'),
        ]

    dense_features = [
        DenseFeat(name='score',
                dimension=1)
        ]
    feature_columns = sparse_features + var_len_sparse_feat + dense_features
    return feature_columns

@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    embedding_name: str = None
    dtype: torch.dtype = torch.long
    
    def __post_init__(self):
        if self.embedding_name is None:
            self.embedding_name = self.name
            
@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: torch.dtype = torch.float
    
    
@dataclass
class VarLenSparseFeat:
    sparse_feat: SparseFeat
    max_len: int
    length_name: str
    @property
    def name(self):
        return self.sparse_feat.name
    @property
    def vocabulary_size(self):
        return self.sparse_feat.vocabulary_size
    @property
    def embedding_dim(self):
        return self.sparse_feat.embedding_dim
    @property
    def embedding_name(self):
        return self.sparse_feat.embedding_name
    @property
    def dtype(self):
        return self.sparse_feat.dtype


def build_feature_positions(feature_columns):
    """Based on the ordering of the feature columns, get the position index"""
    feature_positions = OrderedDict()
    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if isinstance(feat, SparseFeat):
            feature_positions[feat_name] = (start, start+1)
            start += 1
            
        elif isinstance(feat, DenseFeat):
            feature_positions[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
            
        elif isinstance(feat, VarLenSparseFeat):
            feature_positions[feat_name] = (start, start + feat.max_len)
            start += feat.max_len
    
            if feat.length_name not in feature_positions and feat.length_name is not None:
                feature_positions[feat.length_name] = (start, start+1)
                start +=1 
        else:
            raise TypeError('Invalid feature columns type, got', type(feat))
    return feature_positions

def build_pytorch_dataset(df_pd, feature_columns):
    torch_df = {}
    for feat in feature_columns:
        feat_name = feat.name
        if isinstance(feat, SparseFeat) or isinstance(feat, DenseFeat):
            input_tensor = torch.tensor(df_pd[feat_name].values).reshape(-1,1)
            torch_df[feat_name] = input_tensor
            
        elif isinstance(feat, VarLenSparseFeat):
            input_tensor = torch.stack(list(map(lambda x: torch.tensor(x), df_pd[feat_name].values)))
            torch_df[feat_name] = input_tensor
            
            if feat.length_name is not None and feat.length_name not in torch_df:
                torch_df[feat.length_name] = torch.tensor(df_pd[feat.length_name].values).reshape(-1,1)
        else:
            raise TypeError('Invalid feature columns type, got,', type(feat))
    return torch.cat(list(torch_df.values()), dim=-1)

            
        
