import torch
import shutil
import matplotlib.pyplot as plt


def plot_data_shape_dist(train_df, test_df):
    fig, ax = plt.subplots(1, 2)
    ax_idx = 0
    for df, title in zip([train_df, test_df], ["Train", "Test"]):
        df["shape"] = df.apply(lambda row: f"{row['height']}x{row['width']}", axis=1)
        value_counts = df["shape"].value_counts()

        ax[ax_idx].bar(value_counts.index.values, value_counts.values)
        ax[ax_idx].set_title(title)
        ax[ax_idx].set_xticklabels(value_counts.index.values, rotation=45)
        ax_idx += 1


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
