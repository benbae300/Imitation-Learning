from dataset import LineFollowerDataset
from sim import Action
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision import transforms, datasets
from tqdm import tqdm
import os 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    val_acc_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    

                    outputs = model(inputs)
                    #print(outputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, 'hw4_model.pth')
                print("checkpoint saved at epoch", epoch)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    print('Best val Acc: {:4f}'.format(best_acc))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_size = 224
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./dataset"


    # Number of classes in the dataset
    num_classes = 4

    # Batch size for training (change depending on how much memory you have)
    batch_size = 5

    # Number of epochs to train for
    num_epochs = 20

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    # TODO: Load the dataset
    # - You might want to use a different transform than what is already provided
    #dataset = LineFollowerDataset(transform=transforms.ToTensor())

    # TODO: Prepare dataloaders
    # - Rnadomly split the dataset into the train validation dataset.
    #   * Hint: Checkout torch.utils.data.random_split
    # - Prepare train validation dataloaders
    # ========================================
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    
   
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    train_loader, val_loader = None, None
    # ========================================

    # TODO: Prepare model
    # - You might want to use a pretrained model like resnet18 and finetune it for your dataset
    # - See https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ========================================
    model_ft = resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, int(num_ftrs/2)),
        nn.ReLU(),
        nn.Linear(int(num_ftrs/2), num_classes),
    )
    
    
    #model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # 2 linear, 2 activation 
    #nn.sequential(linear layer activation function nn.Linear layer) --- put all the stuff in this funcgtion
    # last layer output should always be 4 
    # ========================================

    # TODO: Prepare loss and optimizer
    # ========================================
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.0005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()


    # TODO: Train model
    # - You might want to print training (and validation) loss and accuracy every epoch
    # - You might want to save your trained model every epoch
    # ========================================
    # ========================================
    train_model(model_ft, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception= False)
    

