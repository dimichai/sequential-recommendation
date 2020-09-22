""" 
Created at 12/04/2019

@author: dimitris.michailidis
original: https://github.com/hungthanhpham94/GRU4REC-pytorch
"""
import torch

from helpers.dataset import Dataset
import numpy as np

from helpers.distance_calculator import vectorized_haversine, calculate_session_item_distance


class SessionParallelLoader:
    def __init__(self, dataset: Dataset, batch_size=50, n_sample=0, sample_alpha=0):
        """
            A class for creating session-parallel mini-batches.

            Args:
                 dataset (Dataset): the session dataset to generate the batches from
                 batch_size (int): size of the batch
                 n_sample (int): number of additional negative samples to add when training
                                (see https://arxiv.org/abs/1706.03847)

            """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha

        # prepare negative sampling - generate the list of popular items.
        if n_sample:
            self.pop = self.dataset.df.groupby('itemId').size()
            self.pop = self.pop[self.dataset.itemmap.itemId.values].values ** self.sample_alpha
            self.pop = self.pop.cumsum() / self.pop.sum()
            self.pop[-1] = 1

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        # initializations
        df = self.dataset.df
        session_offsets = self.dataset.session_offsets
        sorted_session_index = self.dataset.session_index

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        # In the clicks dataset, find the start and the end index of each session.
        start = session_offsets[sorted_session_index[iters]]
        end = session_offsets[sorted_session_index[iters] + 1]
        session_mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices (embedding representation) for clicks where the first sessions start
            # We set the 'start' values because the target of the previous iter is the input in the next
            target_index = df['item_idx'].values[start]
            self.curr_sess = df['sess_idx'].values[start]

            for i in range(minlen - 1):
                # Build inputs and targets. _index here means the item index that is created in dataset.py.
                # The index of the df does not matter since we retrieve the items by .values which returns a series of
                # the values.
                input_index = target_index
                target_index = df['item_idx'].values[start + i + 1]

                if self.n_sample:
                    # todo change back
                    # samples = self.generate_negative_samples(1)
                    samples = self.generate_negative_distance_samples(input_index)
                    output = np.hstack([target_index, samples])
                else:
                    output = target_index

                input = torch.LongTensor(input_index)
                target = torch.LongTensor(output)

                yield input, target, session_mask

            # click indices where a particular session meets second-to-last element
            start = start + minlen - 1
            # see how many sessions should terminate
            session_mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in session_mask:
                maxiter += 1
                if maxiter >= len(session_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = session_offsets[sorted_session_index[maxiter]]
                end[idx] = session_offsets[sorted_session_index[maxiter] + 1]

    def generate_negative_samples(self, length):
        """
        Generates extra negative samples to use while training.
        :param length:
        :return:
        """
        if self.sample_alpha:
            samples = np.searchsorted(self.pop, np.random.rand(self.n_sample * length))
        else:
            samples = np.random.choice(self.dataset.n_items, size=self.n_sample * length)
        if length > 1:
            samples = samples.reshape((length, self.n_sample))
        return samples

    def generate_negative_distance_samples(self, input_index):
        if self.sample_alpha:
            samples = np.searchsorted(self.pop, np.random.rand(self.n_sample))
        else:
            samples = np.random.choice(self.dataset.n_items, size=self.n_sample)

        samples = torch.LongTensor(samples)
        sample_loc = self.dataset.get_item_location(samples)
        sample_loc = self.dataset.get_item_location()
        session_loc = self.dataset.get_session_location(self.curr_sess)
        distance = calculate_session_item_distance(session_loc, sample_loc)

        print('test')
