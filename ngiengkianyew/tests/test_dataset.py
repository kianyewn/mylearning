from dataset.feature_columns import (SparseFeat,
                                     DenseFeat,
                                     VarLenSparseFeat,
                                     build_feature_positions,
                                     build_pytorch_dataset,
                                     gen_dataset,
                                     gen_feature_columns,
                                     )
import torch
import numpy as np
import pandas as pd
from itertools import chain

def test_sparse_feat():         
    sparse_feat = SparseFeat(name='cat1',
                            vocabulary_size=2,
                            embedding_dim=3)
    assert sparse_feat.embedding_name =='cat1'
    
    
def test_dense_feat():
    dense_feat = DenseFeat(name='age',
                           dimension=1)
    assert dense_feat.name == 'age'

    
def test_varlen_sparse_feat():
    var_len_sparse_feat = VarLenSparseFeat(sparse_feat= SparseFeat(name='hist_iid',
                                                                   vocabulary_size=4,
                                                                   embedding_dim=5,),
                                           max_len = 10,
                                           length_name='some_length_column')
    assert var_len_sparse_feat.name == 'hist_iid'
    assert var_len_sparse_feat.vocabulary_size == 4
    assert var_len_sparse_feat.embedding_dim == 5
    assert var_len_sparse_feat.dtype == torch.long
    assert var_len_sparse_feat.max_len == 10
    assert var_len_sparse_feat.length_name == 'some_length_column'
   


def test_build_feature_positions():
    
    sparse_features = [SparseFeat(name='cat1',
                                vocabulary_size=4,
                                embedding_dim=3),
                        SparseFeat(name='cat2',
                                vocabulary_size=4,
                                embedding_dim=3),
                        SparseFeat(name='cat3',
                                vocabulary_size=4,
                                embedding_dim=3),]

    var_len_sparse_feat = [
        VarLenSparseFeat(
            sparse_feat=SparseFeat(name='hist_iid',
                                    vocabulary_size=4,
                                    embedding_dim=3),
            max_len=10,
            length_name='length_iid'),
        VarLenSparseFeat(
            sparse_feat=SparseFeat(name='hist_uid',
                                    vocabulary_size=4,
                                    embedding_dim=3),
            max_len=10,
            length_name='length_iid'),
        ]

    dense_features = [
        DenseFeat(name='age',
                dimension=1),
        DenseFeat(name='salary',
                dimension=1)]
    feature_columns = sparse_features + var_len_sparse_feat + dense_features
    feature_positions = build_feature_positions(feature_columns)
    # assert order is increases and has no overlaps
    # {0, 1, 2, 3, 13, 14, 24, 25, 26}
    ordered_positions = list(chain.from_iterable(feature_positions.values()))
    assert ordered_positions == [0, 1, 1, 2, 2, 3, 3, 13, 13, 14, 14, 24, 24, 25, 25, 26]
    
def test_build_pytorch_dataset():
    df = gen_dataset()
    feature_columns = gen_feature_columns()
    pt_df = build_pytorch_dataset(df, feature_columns)
    assert pt_df.shape == (3,14)

    

    