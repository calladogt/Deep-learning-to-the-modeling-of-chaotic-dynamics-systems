import train_nns
import train_mgds
import train_lstms
import train_cnns
import nn_subsample
import mgd_subsample
import lstm_sub
import cnn_sub
import train_resnns
import resnet_mlps_sub

if __name__ == '__main__':
    train_nns.full_train()
    nn_subsample.full_train()
    train_mgds.full_train()
    mgd_subsample.full_train()
    train_cnns.full_train()
    cnn_sub.full_train()
    train_resnns.full_train()
    resnet_mlps_sub.full_train()
    train_lstms.full_train()
    lstm_sub.full_train()
