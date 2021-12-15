from shape_utils import *
from unsupervised_shells import *
from param import *
import scipy.io as sio
import os


def load_model_matching(model_address, num_epoch=None):
    model = UnsupervisedShells()
    model.param.reset_karray()
    model.load_self(save_path(model_address), num_epoch)
    model.feat_mod.eval()

    return model


def train_faustremeshed_train():
    dataset = Faustremeshed_train()
    model = UnsupervisedShells(dataset, save_path())

    model.train()

def run_pair(x, y):
    print('loading {} {}'.format(x, y))
    mat_dict_x = scipy.io.loadmat("./data/processed/csr00{}.mat".format(x))
    mat_dict_y = scipy.io.loadmat('./data/processed/csr00{}.mat'.format(y))
    print('shaping')
    shape_x = shape_from_dict(mat_dict_x["X"][0])
    shape_y = shape_from_dict(mat_dict_y["X"][0])
    print('model')
    model = load_model_matching("lnd_loss_60_2000_20")
    print('test')
    ass1, ass2 = model.test_model(shape_x, shape_y, plot_result=False)
    print('save')
    sio.savemat('outcome/{}{}/csr00{}.mat'.format(x, y, x), {'VERT': shape_x.vert.detach().cpu().numpy(), 'TRIV': shape_x.triv.detach().cpu().numpy(), 'MTCH': ass1.data.numpy()})
    sio.savemat('outcome/{}{}/csr00{}.mat'.format(x, y, y), {'VERT': shape_y.vert.detach().cpu().numpy(), 'TRIV': shape_y.triv.detach().cpu().numpy(), 'MTCH': ass2.data.numpy()})

if __name__ == "__main__":
    train_faustremeshed_train()
    # run_pair('07a', '07b')