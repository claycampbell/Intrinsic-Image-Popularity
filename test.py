import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    return preds.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_1', type=str, required=True, help='path to the first image')
    parser.add_argument('--image_path_2', type=str, required=True, help='path to the second image')
    args = parser.parse_args()

    image1 = Image.open(args.image_path_1)
    image2 = Image.open(args.image_path_2)

    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=device)) 
    model.eval().to(device)

    score1 = predict(image1, model)
    score2 = predict(image2, model)

    if score1 > score2:
        print("Image 1 has a higher popularity score: %.2f" % score1)
    else:
        print("Image 2 has a higher popularity score: %.2f" % score2)
