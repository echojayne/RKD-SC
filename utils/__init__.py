import torch, datetime, os, sys
from os import path
from tqdm import tqdm
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def int_list(s):
    return [int(item) for item in s.split(',')]

def float_list(s):
    return [float(item) for item in s.split(',')]

def save_model(model, path):
    """
    Save the model

    Args:
        model (nn.Module): model to save
        path (str): path to save the model

    Returns:
        None

    """
    torch.save(model.state_dict(), path)

def save_data(data, path):
    """
    Save the data to test file

    Args:
        data (list): data to save
        path (str): path to save the data

    Returns:
        None

    """
    if not os.path.exists(path):
        with open(path, 'w') as f:
            for item in data:
                f.write("%s\n" % item)
    else:
        with open(path, 'a') as f:
            for item in data:
                f.write("%s\n" % item)

def get_features(model, dataloader, device):
    """
    Get the features from the model

    Args:
        model (nn.Module): model to get the features
        dataloader (DataLoader): data loader to get the features
        device (str): device to run the model
        ncols (int): number of columns for tqdm

    Returns:
        features (list): features of the data
        labels (list): labels of the data

    """

    features= []
    labels = []
    
    with torch.no_grad():
        for image, label in tqdm(dataloader, ncols=100):
            features_batch = model(image.to(device))
            features.append(features_batch)
            labels.append(label)

    return torch.cat(features).cpu().numpy(), torch.cat(labels).cpu().numpy()
                
def test_on_val(model, cla, dataloader, device):
    print('Testing on val...', end = ' \t')
    model.eval()
    cla.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            output = cla(feature)
            correct += (output.argmax(1) == target).sum().item()
    print('Accuracy: ', correct / len(dataloader.dataset))
    return correct / len(dataloader.dataset)