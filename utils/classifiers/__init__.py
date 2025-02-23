from classifiers import stacked_mnist, cifar, places, imagenet, mnist

classifier_dict = {
    'stacked_mnist': stacked_mnist.Classifier,
    'cifar': cifar.Classifier, 
    'places': places.Classifier,
    'imagenet': imagenet.Classifier,
    'mnist': mnist.Classifier
}
