import torch.nn as nn
import torch
from typing import List

class ResultLog:
    def __init__(self):
        self.res = {}
        self.DEBUG = False

    def __call__(self, key, value):
        self.res[key] = value
        if self.DEBUG:
            print(f'result: {key} with value: {value} logged')
        return

    def show_results(self):
        return self.res

    def reset():
        self.res = res
        print('results resetted.')
        return


log = ResultLog()


class MLM(nn.Module):
    """
    Summary:
        - For input data,
            1. we mask the input according to masking probability and remove any tokens that we do not want to mask,
                eg padding (full_mask)
            2. we mask the input according to unchanged probability (unchanged_mask)
            3. we mask the input according to randomize probability (randomize_mask)
            4. we combine these mask via full_mask & ~unchanged_mask & ~randomize_mask
            5. then fill the items inside the randomize mask with the randomized tokens
        - For the output data,
            1. we clone the raw input data
            2. pad the raw input data on indices where ~full_mask
        - For the loss function,
            1. we use CrossEntropyLoss(ignore_index= self.padding_token) to only accumulate loss on masked_tokens
    """
    def __init__(self,
                 vocab_size: int = 10,
                 masking_prob: float = 0.15,
                 no_change_prob: float = 0.1,
                 randomize_prob: float = 0.1,
                 no_mask_tokens: List[int] = []):
        ## the vocab size in your dictionary. This is inclusive of your [MASK] and [PAD] tokens
        self.vocab_size = vocab_size
        ## the index in vocabulary to represent the [MASK] token
        self.mask_token = vocab_size - 1
        ## the index in vocabulary to represent the [PAD] token
        self.padding_token = vocab_size - 2
        ## probability of masking the input tokens for mlm, aka masking probability
        self.masking_prob = masking_prob
        ## probability of keeping the masked tokens as randomized tokens
        self.randomize_prob = randomize_prob
        ## probability of keeping the masked tokens are unchanged tokens
        self.no_change_prob = no_change_prob
        ## tokens that you do not want to mask.
        ## You do not want to mask un-related tokens
        ## Sometimes, you also do not want to mask some important tokens
        self.no_mask_tokens = no_mask_tokens + [self.mask_token, self.padding_token]

    def gen_sample_data(self, num_seq: int = 3):
        """Generate `num_seq` sentences without any mask or padding tokens"""
        ## you do not want to add in the [MASK] or [PAD] tokens
        ## generate sample data of elements from 0 to self.vocab_size-2, of length 5 to 15
        single_sentence_generator = lambda: torch.randint(self.vocab_size - 2, size=(torch.randint(1, 10, size=(1,)),))
        return [single_sentence_generator() for _ in range(num_seq)]

    def truncate_data(self, sample_data: torch.Tensor, max_len: int = 3):
        """Truncate the tokenized data to max_len, data should be (T,B)
        where T is the max_len, and B is the batch_size

        if the length of the sentence is longer than max_len, then we pad the tokens
        """
        ## number of
        batch_data = torch.full((max_len, len(sample_data)), self.padding_token)
        for i, tokenized_sent in enumerate(sample_data):
            seq_len = min(max_len, len(tokenized_sent))
            ## `i` is broadcasted. tensor[(1,2), 1] -> tensor[(1,2), (1,1)]. Select elements (1,1) and (2,1)
            batch_data[:seq_len, i] = tokenized_sent[:seq_len]
        return batch_data

    def mask_tokens(self, batch_data: torch.Tensor):
        """Mask the batched data according to self.masking_prob"""
        masking = torch.randn(batch_data.shape) < self.masking_prob
        #         print()
        #         masked = batch_data.masked_fill_(masking, self.mask_token) # element-wise multiplication
        return masking

    def mask_tokens_omit_no_mask(self, batch_data: torch.Tensor, full_mask: torch.Tensor):
        """omit the tokens that should not be masked"""
        for tok in self.no_mask_tokens:
            ## full_mask &= batch_data != tok -> inplace operations, more neat
            full_mask = full_mask & (batch_data != tok)
        return full_mask

    def unchanged_mask_tokens(self, full_mask: torch.Tensor):
        unchanged_token_mask = full_mask & (torch.randn(full_mask.shape) < self.no_change_prob)
        return unchanged_token_mask

    def random_mask_tokens(self, full_mask: torch.Tensor):
        random_token_mask = full_mask & (torch.randn(full_mask.shape) < self.randomize_prob)
        return random_token_mask

    def combine_mask(self, full_mask: torch.Tensor, unchanged_token_mask: torch.Tensor, random_token_mask: torch.Tensor):
        """The final set of tokens that are going to be replaced by [MASK]
        This is where we make 10% of masking tokens to be unchanged,
        10% of masking tokens to be random. Important!!!
        """
        mask = full_mask & ~unchanged_token_mask & ~random_token_mask
        ### (masking_omit != random_token_mask) != unchanged_token_mask
        ### The above will not work. we want (True & True) = True.
        ### The above will set False != True = True.
        return mask

    def mask_fill(self, mask: torch.Tensor, batch_data: torch.Tensor):
        # remember that if you do masked_fill_, it will be an inplace operation
        batch_data = batch_data.clone()
        mask_filled_data = batch_data.masked_fill_(mask, self.mask_token)
        return mask_filled_data

    def fill_random_tokens(self, mask: torch.Tensor, random_token_mask: torch.Tensor):
        ## returns a tuple of tensors, where the first tuple represents indices in the first dim
        ## the second tuple represents indices in the second dimension, and so on if applicable
        # # (tensor([0, 0, 1, 2]), tensor([0, 1, 1, 1]))
        tuple_indices = torch.nonzero(random_token_mask, as_tuple=True)
        print(tuple_indices)
        random_tokens = torch.randint(self.vocab_size, size=(len(tuple_indices[0]),))
        log('random_tokens', random_tokens)
        # you can do indexing with indices isnide a tensor as opposed to a list or tuple
        # eg matrix[tensor[1,2], tensor[3,4]] will retrieve matrix[1,3] and matrix[2,4]
        mask[tuple_indices] = random_tokens
        return mask

    def get_labels(self, batch_data: torch.Tensor):
        return batch_data.clone()

    def format_labels_for_loss(self, y: torch.Tensor, full_mask: torch.Tensor):
        """Assign token [PAD] to all the other locations in the labels.
        The labels equal to [PAD] will not be used in the loss.
        """
        return y.masked_fill_(~full_mask, self.padding_token)


mlm = MLM(vocab_size=10,
          masking_prob=0.3)
sample_data = mlm.gen_sample_data(num_seq=3);
log('sample_data', sample_data)
# [tensor([2, 5, 0, 5, 4, 0]), tensor([2]), tensor([1, 0, 1, 1])]

batch_data = mlm.truncate_data(sample_data=sample_data);
log('batch_data', batch_data)
# tensor([[2, 2, 1],
#         [5, 8, 0],
#         [0, 8, 1]])
masking = mlm.mask_tokens(batch_data);
log('initial_mask', masking)
masking_omit = mlm.mask_tokens_omit_no_mask(batch_data, masking);
log('omitted_mask', masking_omit)
unchanged_token_mask = mlm.unchanged_mask_tokens(masking_omit);
log('unchanged_token', unchanged_token_mask)
random_token_mask = mlm.random_mask_tokens(masking_omit);
log('random_mask', random_token_mask)
mask = mlm.combine_mask(masking_omit, unchanged_token_mask, random_token_mask);
log('combined_mask', mask)
mask_filled_data = mlm.mask_fill(mask, batch_data);
log('mask_filled_data', mask_filled_data)
mask_random_filled = mlm.fill_random_tokens(mask_filled_data, random_token_mask);
log('mask_random_filled', mask_random_filled)
y = mlm.get_labels(batch_data)
formatted_y = mlm.format_labels_for_loss(y, masking_omit)
log.show_results()

# {'sample_data': [tensor([5, 0, 7, 4, 5, 0, 7]),
#   tensor([4, 2, 1, 5, 2, 3]),
#   tensor([2, 2, 5])],
#  'batch_data': tensor([[5, 4, 2],
#          [0, 2, 2],
#          [7, 1, 5]]),
#  'initial_mask': tensor([[False, False,  True],
#          [ True,  True,  True],
#          [ True,  True,  True]]),
#  'omitted_mask': tensor([[False, False,  True],
#          [ True,  True,  True],
#          [ True,  True,  True]]),
#  'unchanged_token': tensor([[False, False, False],
#          [False,  True, False],
#          [ True,  True, False]]),
#  'random_mask': tensor([[False, False, False],
#          [ True,  True,  True],
#          [False, False, False]]),
#  'combined_mask': tensor([[False, False,  True],
#          [False, False, False],
#          [False, False,  True]]),
#  'mask_filled_data': tensor([[5, 4, 9],
#          [4, 1, 4],
#          [7, 1, 9]]),
#  'random_tokens': tensor([4, 1, 4]),
#  'mask_random_filled': tensor([[5, 4, 9],
#          [4, 1, 4],
#          [7, 1, 9]])}