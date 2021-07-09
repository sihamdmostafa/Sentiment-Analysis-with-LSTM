import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader


    



def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

def datapreprocessing(reviews,labels):
    
    reviews = reviews.lower() # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])
    
    #Encoding the labels
    # 1=positive, 0=negative label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
    
    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
    
    seq_length = 200
    features = pad_features(reviews_ints, seq_length=seq_length)
    
    
    split_frac = 0.8

    ## split data into training, validation, and test data (features and labels, x and y)

    split_idx = int(len(features)*split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    batch_size = 50

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader,valid_loader,test_loader,vocab_to_int

    