import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import random

# Image preprocessing modules
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
hidden_size = 500
num_classes = 2
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.LeakyReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.tanh(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=4)
        self.conv3 = conv3x3(1, 16, stride=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 4)
        self.layer3 = self.make_layer(block, 64, layers[2], 4)
        self.layer4 = self.make_layer(block, 128, layers[3], 4)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.apply(self._init_weights)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(14*14*128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def create_batches(images,minibatch_size, y_val, train = True) :
    
    x = [images[i][0] for i in range(len(images))]
    y = [images[i][1] for i in range(len(images))]
    # Lets shuffle X and Y
    # shuffled index of examples
    y = convert_labels(y, y_val)

    if train :
        ratio = y.count(1) / y.count(0)
        ratio = 0.1 * (1/ratio)
        copy = x.copy()
        copy2 = y.copy()
        count = 0
        for i in range(len(copy)) :
            if copy2[i] == 1 :
                for i in range(int(ratio)) :
                    s = copy[i]
                    #gaus = transforms.GaussianBlur(kernel_size=(51,91), sigma=random.uniform(0.3,0.6))(s)
                    horz = transforms.RandomHorizontalFlip()(s)
                    #noisy = s+torch.randn_like(s) * random.uniform(0.05,0.1)
                    #x.append(gaus)
                    #x.append(noisy)
                    x.append(horz)
                    x.append(s)
                    y.append(1)
                    #y.append(1)
                    y.append(1)


            else :
                if count % 20 == 0 :
                    s = copy[i]
                    gaus = transforms.GaussianBlur(kernel_size=(51,91), sigma=random.uniform(0.1,0.6))(s)
                    horz = transforms.RandomHorizontalFlip()(s)
                    #noisy = s+torch.randn_like(s) * random.uniform(0.05,0.1)
                    x.append(gaus)
                    #x.append(noisy)
                    x.append(horz)
                    y.append(0)
                    #y.append(0)
                    y.append(0)
                count += 1

    m = len(y)
    permutation = list(np.random.permutation(m))            # shuffled index of examples
    shuffled_X = [x[permutation[i]] for i in range(len(permutation))]
    shuffled_Y = [y[permutation[i]] for i in range(len(permutation))]
    
    minibatches = []                                        # we will append all minibatch_Xs and minibatch_Ys to this minibatch list 
    number_of_minibatches = int(m/minibatch_size)           # number of mini batches 
    
    for k in range(number_of_minibatches):
        minibatch_X = shuffled_X[k*minibatch_size: (k+1)*minibatch_size ]
        minibatch_Y = shuffled_Y[k*minibatch_size: (k+1)*minibatch_size ]
        minibatch_pair = (minibatch_X , minibatch_Y)                        #tuple of minibatch_X and miinibatch_Y
        minibatches.append(minibatch_pair)
    if m%minibatch_size != 0 :
        last_minibatch_X = shuffled_X[(k+1)*minibatch_size: m ]
        last_minibatch_Y = shuffled_Y[(k+1)*minibatch_size: m ]
        last_minibatch_pair = (last_minibatch_X , last_minibatch_Y)
        minibatches.append(last_minibatch_pair)
    return minibatches

def train_test_files(train_path, test_path) :
    file = open(train_path, 'r')

    files = dict()

    line = file.readline()
    while line != '' :
        files[line[:-1]] = 1

        line = file.readline()

    file.close()
    file = open(test_path, 'r')
    line = file.readline()

    while line != '' :
        files[line[:-1]] = 0

        line = file.readline()

    return files

def get_labels(path) :
    file = open(path, 'r')

    labels = dict()

    data = csv.reader(file)
    classes = dict()
    num_classes = 0

    for i, line in enumerate(data) :
        if i == 0 : continue
        diseases = line[1].split('|')
        for disease in diseases :
            if classes.get(disease) == None :
                classes[disease] = num_classes
                num_classes += 1

        labels[line[0]] = [classes[diseases[i]] for i in range(len(diseases))]
    
    return labels, num_classes, classes

def get_images_labels2(images_path, file_types, labels, read_range=100000, transform=None,target_transform=None) :
    dirs = os.listdir(images_path)
    
    
    train = []
    test = []

    for dir in dirs :
        path = images_path + dir + '/images/'
        files = os.listdir(path)

        for i in range(min(read_range, len(files))) :
            file = files[i]
            image = torchvision.io.read_image(path + file)
            if image.shape[0] != 1 :
                continue
            image = image.float()
            if file_types[file] == 1 :
                train.append([image, labels[file]])
            else :
                test.append([image, labels[file]])

    return train, test

def get_images_labels(folder, read_range = 10000, transform=None,target_transform=None) :
    dirs = os.listdir(folder)
    images = []
    t = torchvision.transforms.CenterCrop(1024)
    for i in range(len(dirs)) :
        dir = dirs[i]
        files = os.listdir(folder + dir)
        cur_dir = folder + dir + '/'
        
        for j in range(min(len(files), read_range)) :
            file = files[j]
            image = torchvision.io.read_image(cur_dir + file)
            if image.shape[0] != 1 :
                continue
            image = torch.reshape(image, (image.shape[1], image.shape[2]))
            image = image.float()
            s = t(image)
            if transform:
                s = transform(s)
            s = torch.reshape(s, (1, s.shape[0], s.shape[1]))
            images.append([s, i])
    return images

def convert_labels(labels, y) :
    return [1 if y in labels[i] else 0 for i in range(len(labels))]

def show_distribution(labels, num_classes, classes) :
    freqs = [0] * num_classes
    c = {y: x for x, y in classes.items()}

    images = os.listdir('./images./images_001/images/') + os.listdir('./images./images_002/images/')
    d = dict()
    for image in images :
        d[image] = 0 

    label_items = list(labels.items())
    items = [c[label_items[i][1][j]] for i in range(len(label_items)) for j in range(len(label_items[i][1])) if d.get(label_items[i][0]) != None ]
    lbls = set(items)
    counts = [items.count(i) for i in lbls]
    plt.bar(list(lbls), counts, color='maroon',width=0.4)
    plt.xticks(rotation=45)
    plt.show()

    


def main() :
    # train_images = get_images_labels("./chest_xray/train/")
    # test_images = get_images_labels("./chest_xray/test/")
    dir = './images/'
    labels, num_classes, classes = get_labels('./Data_Entry_2017.csv')
    #show_distribution(labels, num_classes, classes)
    file_types = train_test_files('./train_val_list.txt', './test_list.txt')
    train_images, test_images = get_images_labels2(dir, file_types, labels, read_range=2000)
    
    

    models = []

    for i in range(num_classes) :
        train_batches = create_batches(train_images, 50, i)
        test_batches = create_batches(test_images, 50, i, train=False)

        # Loss and optimizer
        

        m = train(test_batches, train_batches, i)
        m = m.to(cpu)
        models.append(m)

    test(models, test_images)

def test(models, test_data) :
    outputs = []
    data = torch.stack([test_data[i][0] for i in range(len(test_data))]).to(device)
    labels = [test_data[i][1] for i in range(len(test_data))]
    with torch.no_grad():
        for model in models :
            model.eval()
            out = model(data)
            _, predicted = torch.max(out.data, 1)
            outputs.append(predicted)

    for i, label in enumerate(labels) :
        out = [outputs[j][i] for j in range(len(outputs))]
        correct = 0
        total_labels = len(label)

        for cur in label :
            if out[cur] == 1 :
                correct += 1

def train(test_batches, train_batches, y_val) :
    # Train the model_conv
    model_conv = ResNet(ResidualBlock, [4, 4, 4, 4]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)
    total_step = len(train_batches)
    test_acc_list, train_acc_list = [], []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_batches):
            #labels = convert_labels(labels, y_val)
            saved = np.array(labels)
            images = torch.stack(images)
            labels = torch.as_tensor(labels)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model_conv(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            s = predicted.to(cpu).numpy()
            t = labels.to(cpu).numpy()
            print('total:' + str(np.sum(saved == 1)) + ' have:' + str(np.sum((s == t))))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                

        # Test the model_conv
        model_conv.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_batches:
                #labels = convert_labels(labels, y_val)
                saved = np.array(labels)
                images = torch.stack(images)
                labels = torch.as_tensor(labels)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model_conv(images)
                _, predicted = torch.max(outputs.data, 1)
                s = predicted.to(cpu).numpy()
                t = labels.to(cpu).numpy()
                print('total:' + str(np.sum(saved == 1)) + ' have:' + str(np.sum((s == t))))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model_conv on the {} test images: {} %'.format(len(test_batches) * 100, 100 * correct / total))
            test_acc_list.append(100 * correct / total)

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_batches:
                #labels = convert_labels(labels, y_val)
                images = torch.stack(images)
                labels = torch.as_tensor(labels)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model_conv(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model_conv on the train images: {} %'.format(100 * correct / total))
            train_acc_list.append(100 * correct / total)
        
    plt.plot(train_acc_list, '-b', label='train acc')
    plt.plot(test_acc_list, '-r', label='test acc')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Accuracy ~ Epoch')
    # plt.savefig('assets/accr_{}.png'.format(cfg_idx))
    plt.show()


    # Save the model_conv checkpoint
    torch.save(model_conv.state_dict(), 'model_conv.ckpt')
    return model_conv

if __name__ == '__main__' :
    main()