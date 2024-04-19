import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from .clip import *
from .model import VisionTransformer, Classifier, KD_IB, featureFusion
from utils.network import NetworkManager


vit_parameters = {
    'image_resolution': 224,
    'vision_patch_size': 32,
    'vision_width': 384,
    'vision_layers': 9,
    'vision_heads': 384//16,
    'embed_dim': 512
}

classifier_layels_list = [] # 512×list[0] -> list[0]×list[1] -> list[1]×list[2] -> ... -> list[-1]×100

def GenTeacher(model_name, device = 'cuda'):
    """
    Generate teacher model from model_name

    Args:
        model_name (str): model name
        device (str): device to load model

    Returns:
        img_encoder (nn.Module): teacher model

    """
    model, _ = load(model_name, device)
    img_encoder = model.visual
    return img_encoder.float()

def GenStudent(vit_parameters = vit_parameters, device = 'cuda'):
    """
    Generate student model from model_name

    Args:
        vit_parameters (dict): parameters for VisionTransformer
        device (str): device to load model

    Returns:
        model (nn.Module): student model

    """
    model = VisionTransformer(
                input_resolution=vit_parameters['image_resolution'],
                patch_size=vit_parameters['vision_patch_size'],
                width=vit_parameters['vision_width'],
                layers=vit_parameters['vision_layers'],
                heads=vit_parameters['vision_heads'],
                output_dim=vit_parameters['embed_dim']
            ).to(device)
    return model

def GenClassifier(input_dim = 512, output_dim = 100, hidden_list=[256], device = 'cuda'):
    """
    Generate classifier model from model_name

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        hidden_list (list): list of hidden layer sizes
        device (str): device to load model

    Returns:
        classifier (nn.Module): classifier model

    """
    net = NetworkManager(input_dim, output_dim)
    classifier = net.create_network(hidden_list).to(device)
    return classifier

    