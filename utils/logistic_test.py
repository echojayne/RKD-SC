from utils import get_features
import torch, datetime
from sklearn.linear_model import LogisticRegression
from models import GenStudent, GenTeacher, GenClassifier, vit_parameters
import numpy as np
from utils.dataset import train_transform, test_transform
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

vit_parameters = {
    'image_resolution': 224,
    'vision_patch_size': 32,
    'vision_width': 384,
    'vision_layers': 9,
    'vision_heads': 384//16,
    'embed_dim': 512
}

def logic_test(model, train_loader, test_loader, device):
    
    print(f">> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running logistic regression on the features extracted from the model")
    # Calculate the image features
    train_features, train_labels = get_features(model, train_loader, device)
    test_features, test_labels = get_features(model, test_loader, device)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    train_acc = classifier.score(train_features, train_labels)
    test_acc = classifier.score(test_features, test_labels)

    print(f">> Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # only MSE loss
    # Teacher Model Accuracy = 88.762
    # Student Model Accuracy = 81.308

    return train_acc, test_acc, classifier

if __name__ == '__main__':
    device = 'cuda:2'
    # model = GenTeacher('ViT-B/32', device)  # size of teacher model is 87,849,216
    model = GenStudent(vit_parameters, device)  # size of student model is 87,849,216
    # 加载权重
    model.load_state_dict(torch.load('/home/ubuntu/users/dky/RKD-SC/results/KD/student_KD.pt'))
    BATCH_SIZE = 250
    train_dataset = datasets.CIFAR100('/home/ubuntu/users/dky/dataset', True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100('/home/ubuntu/users/dky/dataset', False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    train_acc, test_acc, classifier_sk = logic_test(model, train_loader, test_loader, device)
    classifier = GenClassifier(512,100,[],device)
    classifier.weights.data = torch.from_numpy(classifier_sk.coef_.T).float()
    classifier.bias.data = torch.from_numpy(np.array([classifier_sk.intercept_])).float()
    torch.save(classifier.state_dict(), '/home/ubuntu/users/dky/CLIP-KD-copy/results/weights/classifier_TEA.pt')


