import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def define_classifier(opt):
    """
    Defines and loads a pre-trained classifier model based on the given configuration options.

    This function modifies certain classifier-specific hyperparameters (like `rnn_size` and `g_dim`) in the
    options temporarily, loads a pre-trained model from a checkpoint, and restores the original hyperparameters.

    :param opt: A configuration object containing model parameters. Expected keys include:
                - `rnn_size`: The size of the recurrent neural network layers.
                - `g_dim`: The dimension of the latent feature space.
                - `cls_path`: The path to the pre-trained classifier's state dictionary.
    :return: A PyTorch model representing the classifier, loaded with pre-trained weights and set to evaluation mode.
    :raises NotImplementedError: If the method has not been implemented.
    """
    raise NotImplementedError("The 'define_classifier' method must be implemented by the user.")


def load_dataset(opt, mode='train'):
    """
    Loads the MUG dataset and returns data loaders for training and testing.

    This function applies necessary transformations and filtering to the dataset, sets up
    data loaders for efficient training and testing, and handles bad sample removal from the test set.

    :param opt: A configuration object that must include:
                - `batch_size`: The batch size to be used for both training and testing loaders.
                - `dataset_size`: Will be set to the size of the training dataset.
    :param mode: A string indicating the mode of the dataset loading process.
                 Use `'train'` for training and testing loaders, and `'eval'` for test loader only.
                 Default is `'train'`.
    :return:
        - If `mode='train'`: A tuple containing:
            - `test_loader`: A DataLoader object for the testing dataset.
            - `train_loader`: A DataLoader object for the training dataset.
        - If `mode='eval'`: Only the `test_loader` is returned.
    :raises NotImplementedError: If the method has not been implemented.
    """
    raise NotImplementedError("The 'load_dataset' method must be implemented by the user.")
