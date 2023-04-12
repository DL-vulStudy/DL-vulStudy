import copy
import numpy
import sys
from tqdm import tqdm
import numpy as np
import torch
import json
from imblearn.over_sampling import SMOTE


class DataEntry_reveal:
    def __init__(self, dataset, feature_repr, label, filename=None, meta_data=None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet_reveal)
        self.features = copy.deepcopy(feature_repr)
        self.label = label
        self.meta_data = meta_data
        self.filename = filename

        pass

    def __repr__(self):
        return str(self.features) + '\t' + str(self.label)

    def __hash__(self):
        return str(self.features).__hash__

    def is_positive(self):
        return self.label == 1


class DataSet_reveal:
    def __init__(self,train_src, test_src, batch_size):
        self.train_entries = []
        self.valid_entries = []
        self.test_entries = []
        self.train_batch_indices = []
        self.valid_batch_indices = []
        self.test_batch_indices = []
        self.batch_size = batch_size
        self.hdim = 2
        self.positive_indices_in_train = []
        self.negative_indices_in_train = []
        
        if train_src is not None:
            features = []
            targets = []
            filename = []
            with open(train_src) as fp:
                path_list = fp.readlines()
                for path in tqdm(path_list):
                    json_data_file = open(path.strip(), "r")
                    data = json.load(json_data_file)
                    features.append(data['graph_feature'])
                    targets.append(data['target'])
                    filename.append(json_data_file.name.split('/')[-1].split(".")[0])
                    json_data_file.close()
                    del data
            train_x = numpy.array(features)
            train_y = numpy.array(targets)
            for _x, _y in zip(train_x, train_y):
                self.add_data_entry(_x.tolist(), _y.tolist(), 'train')
        if test_src is not None:
            features = []
            targets = []
            filename = []
            with open(test_src) as fp:
                path_list = fp.readlines()
                for path in tqdm(path_list):
                    json_data_file = open(path.strip(), "r")
                    data = json.load(json_data_file)
                    features.append(data['graph_feature'])
                    targets.append(data['target'])
                    filename.append(json_data_file.name.split('/')[-1].split(".")[0])
                    json_data_file.close()
                    del data
            test_x = numpy.array(features)
            test_y = numpy.array(targets)
            for _x, _y, _f in zip(test_x, test_y, filename):
                self.add_data_entry(_x.tolist(), _y.tolist(), _f, 'test')
        self.initialize_dataset()

    def initialize_dataset(self, balance=True, output_buffer=sys.stderr):
        if isinstance(balance, bool) and balance:
            entries = []
            train_features = []
            train_targets = []
            for entry in self.train_entries:
                train_features.append(entry.features)
                #train_features.append(np.array(entry.features))
                train_targets.append(entry.label)
                #train_targets.append(np.array(entry.label))
            train_features = np.array(train_features)
            train_targets = np.array(train_targets)
            smote = SMOTE(random_state=1000)
            if len(train_targets)!= 0:
                features, targets = smote.fit_resample(train_features, train_targets)
                for feature, target in zip(features, targets):
                #for feature, target in zip(train_features, train_targets):
                    entries.append(DataEntry_reveal(self, feature.tolist(), target.item()))
                self.train_entries = entries
        elif isinstance(balance, list) and len(balance) == 2:
            entries = []
            for entry in self.train_entries:
                if entry.is_positive():
                    for _ in range(balance[0]):
                        entries.append(
                            DataEntry_reveal(self, entry.features, entry.label, entry.meta_data)
                        )
                else:
                    if np.random.uniform() <= balance[1]:
                        entries.append(
                            DataEntry_reveal(self, entry.features, entry.label, entry.meta_data)
                        )
            self.train_entries = entries
            pass
        for tidx, entry in enumerate(self.train_entries):
            if entry.label == 1:
                self.positive_indices_in_train.append(tidx)
            else:
                self.negative_indices_in_train.append(tidx)
        self.initialize_train_batches()
        if output_buffer is not None:
            print('Number of Train Entries %d #Batches %d' % \
                  (len(self.train_entries), len(self.train_batch_indices)), file=output_buffer)
        self.initialize_valid_batches()
        if output_buffer is not None:
            print('Number of Valid Entries %d #Batches %d' % \
                  (len(self.valid_entries), len(self.valid_batch_indices)), file=output_buffer)
        self.initialize_test_batches()
        if output_buffer is not None:
            print('Number of Test  Entries %d #Batches %d' % \
                  (len(self.test_entries), len(self.test_batch_indices)), file=output_buffer)

    def add_data_entry(self, feature, label, filename = None,  part='train'):
        assert part in ['train', 'valid', 'test']
        entry = DataEntry_reveal(self, feature, label, filename)
        if part == 'train':
            self.train_entries.append(entry)
        elif part == 'valid':
            self.valid_entries.append(entry)
        else:
            self.test_entries.append(entry)

    def initialize_train_batches(self):
        self.train_batch_indices = self.create_batches(self.batch_size, self.train_entries)
        return len(self.train_batch_indices)
        pass

    def clear_test_set(self):
        self.test_entries = []

    def initialize_valid_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batch_indices = self.create_batches(batch_size, self.valid_entries)
        return len(self.valid_batch_indices)
        pass

    def initialize_test_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batch_indices = self.create_batches(batch_size, self.test_entries)
        return len(self.test_batch_indices)
        pass

    def get_next_train_batch(self):
        if len(self.train_batch_indices) > 0:
            indices = self.train_batch_indices.pop()
            if len(indices) <= 1: indices = self.train_batch_indices.pop()
            features, targets, _ = self.prepare_data(self.train_entries, indices)
            same_class_features = self.find_same_class_data(ignore_indices=indices)
            different_class_features = self.find_different_class_data(ignore_indices=indices)
            return features, targets, same_class_features, different_class_features
        raise ValueError('Initialize Train Batch First by calling dataset.initialize_train_batches()')
        pass

    def get_next_valid_batch(self):
        if len(self.valid_batch_indices) > 0:
            indices = self.valid_batch_indices.pop()
            return self.prepare_data(self.valid_entries, indices)
        raise ValueError('Initialize Valid Batch First by calling dataset.initialize_valid_batches()')
        pass

    def get_next_test_batch(self):
        if len(self.test_batch_indices) > 0:
            indices = self.test_batch_indices.pop()
            return self.prepare_data(self.test_entries, indices)
        raise ValueError('Initialize Test Batch First by calling dataset.initialize_test_batches()')
        pass

    def create_batches(self, batch_size, entries):
        _batches = []
        if batch_size == -1:
            batch_size = self.batch_size
        total = len(entries)
        indices = np.arange(0, total - 1, 1)
        np.random.shuffle(indices)
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            _batches.append(indices[curr:c_end])
            curr = c_end
        return _batches

    def prepare_data(self, _entries, indices):
        batch_size = len(indices)
        features = np.zeros(shape=(batch_size, self.hdim))
        targets = np.zeros(shape=(batch_size))
        filename = []
        # max_dim = -1
        # temp_features = []
        # for tidx, idx in enumerate(indices):
        #     entry = _entries[idx]
        #     assert isinstance(entry, DataEntry_reveal)
        #     temp_features.append(entry.features)
        #     targets[tidx] = entry.label
        # for feature_i in temp_features:
        #     max_dim = max(max_dim , np.array(feature_i).shape[0])
        # features = np.zeros(shape=(batch_size, max_dim, self.hdim))
        # for i, feature in enumerate(temp_features):
        #     features[i,:np.array(feature).shape[0],:np.array(feature).shape[1]] = feature
        for tidx, idx in enumerate(indices):
            entry = _entries[idx]
            assert isinstance(entry, DataEntry_reveal)
            targets[tidx] = entry.label
            filename.append(entry.filename)
            for feature_idx in range(self.hdim):
               features[tidx, feature_idx] = entry.features[feature_idx]
        return torch.FloatTensor(features), torch.LongTensor(targets), filename

    def find_same_class_data(self, ignore_indices):
        positive_indices_pool = set(self.positive_indices_in_train).difference(ignore_indices)
        negative_indices_pool = set(self.negative_indices_in_train).difference(ignore_indices)
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool)

    def find_different_class_data(self, ignore_indices):
        positive_indices_pool = set(self.negative_indices_in_train).difference(ignore_indices)
        negative_indices_pool = set(self.positive_indices_in_train).difference(ignore_indices)
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool)

    def find_triplet_loss_data(self, ignore_indices, negative_indices_pool, positive_indices_pool):
        indices = []
        for eidx in ignore_indices:
            if self.train_entries[eidx].is_positive():
                indices_pool = positive_indices_pool
            else:
                indices_pool = negative_indices_pool
            indices_pool = list(indices_pool)
            indices.append(np.random.choice(indices_pool))
        features, _, _ = self.prepare_data(self.train_entries, indices)
        return features


def create_dataset(train_file, valid_file=None, test_file=None, batch_size=32, output_buffer=sys.stderr):
    if output_buffer is not None:
        print('Reading Train data from %s' % train_file, file=output_buffer)
    train_data = json.load(open(train_file))
    # "target": 1, "graph_feature"
    hdim = len(train_data[0]["graph_feature"])
    dataset = DataSet_reveal(batch_size=batch_size, hdim=hdim)
    for data in train_data:
        dataset.add_data_entry(data["graph_feature"], min(int(data["target"]), 1), part='train')
    if valid_file is not None:
        if output_buffer is not None:
            print('Reading Valid data from %s' % valid_file, file=output_buffer)
        valid_data = json.load(open(valid_file))
        for data in valid_data:
            dataset.add_data_entry(data["graph_feature"], min(int(data["target"]), 1), part='valid')
    if test_file is not None:
        if output_buffer is not None:
            print('Reading Test data from %s' % test_file, file=output_buffer)
        test_data = json.load(open(test_file))
        for data in test_data:
            dataset.add_data_entry(data["graph_feature"], min(int(data["target"]), 1), part='test')
    # dataset.initialize_dataset()
    return dataset


    

