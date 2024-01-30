import matplotlib.pyplot as plt
import math
import os
    
def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp

def show_train_loss(epoch_loss, sub_dataset, lr):  #这个rmse的参数只是用于文件名词
        
    plt.xticks(range(1, len(epoch_loss)+1))
    plt.plot(epoch_loss, label='rmse')
    plt.title('train loss on CMAPSS Data %s'%(sub_dataset))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    if not os.path.exists('./result_para/%s/'%(sub_dataset)):
        os.makedirs('./result_para/%s/'%(sub_dataset))
    plt.savefig('./result_para/%s/lr=%f.png'%(sub_dataset, lr))
    plt.close()
    
def show_all_loss(train_loss, test_loss, test_score, sub_dataset):  #这个rmse的参数只是用于文件名词
        
    plt.plot(train_loss, label='train_rmse')
    plt.plot(range(5, len(train_loss)+1, 5), test_loss[:], label='test_rmse')
    # plt.plot(range(5, len(train_loss)+1, 5), test_score[:], label='test_score')
    plt.title('all loss on CMAPSS Data %s'%(sub_dataset))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if not os.path.exists('./result_para/%s/'%(sub_dataset)):
        os.makedirs('./result_para/%s/'%(sub_dataset))
    plt.savefig('./result_para/%s/all_loss.png'%(sub_dataset))
    plt.close()
    
def show_test_rmse(epochs, test_rmse, sub_dataset):  #这个rmse的参数只是用于文件名词
        
    #plt.plot(train_loss, label='train_rmse')
    plt.plot(range(5, epochs+1, 5), test_rmse[:], label='test_rmse')
    # plt.plot(range(5, len(train_loss)+1, 5), test_score[:], label='test_score')
    plt.title('test rmse on CMAPSS Data %s'%(sub_dataset))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('./result_para/%s/test_rmse.png'%(sub_dataset))
    plt.close()
    
def show_test_score(epochs, test_score, sub_dataset):  #这个rmse的参数只是用于文件名词
        
    #plt.plot(train_loss, label='train_rmse')
    #plt.plot(range(5, epochs+1, 5), test_loss[:], label='test_rmse')
    plt.plot(range(5, epochs+1, 5), test_score[:], label='test_score')
    plt.title('test score on CMAPSS Data %s'%(sub_dataset))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('./result_para/%s/test_score.png'%(sub_dataset))
    plt.close()


def visualize(result, rmse, score, sub_dataset, train=False):

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:2].to_numpy()

    if train:
        plt.plot(true_rul, label='Actual Data')
        pred_rul_del = pred_rul[:, 0]
        plt.scatter(x=range(1, len(pred_rul)+1, 100), y=pred_rul_del[0::100], s=1,
                    color='C1', label='Predicted Data')
        #plt.plot(pred_rul, label='Predicted Data')
    else:
        plt.plot(true_rul, label='Actual Data')
        #plt.plot(pred_rul, label='Predicted Data')
        plt.scatter(x=range(1, len(pred_rul)+1), y=pred_rul, s=10, color='C1', label='Predicted Data')
    
    plt.title('RUL Prediction on CMAPSS Data %s'%(sub_dataset))
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    
    if train:
        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        plt.savefig('./result/%s_train_%.2f_%.2f.png'%(sub_dataset, rmse, score))
        plt.close()
    else:
        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        plt.savefig('./result/%s_test_%.2f_%.2f.png'%(sub_dataset, rmse, score))
        plt.close()
    
    plt.show()