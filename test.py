
dataier = iter(test_loader)
images, labels = dataier.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(Variable(images))
_, pred = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[pred[j]] for j in range(4)))

correct = 0.0
total = 0
for data in test_loader:
    images, labels = data
    outputs = net(Variable(images))
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum()
print('Accuracy of the network on the 10000 test images :%d %%' % (100 * correct / total))

def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()
    # np.transpose :按需求转置
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()