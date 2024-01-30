import pandas as pd
import numpy as np
from BERT_ft import *
from BERT_pre2 import *
from utils import *

from smtplib import SMTP_SSL
from email.mime.text import MIMEText

args = {}
min_rmse = 9999
min_score = 999999

min_rmse_np = np.array([])
min_score_np = np.array([])
#result = np.array([[0,0,0,0,0]])
result = np.array([[0,0,0]])

FD_list = ['FD001', 'FD002', 'FD003', 'FD004']
epochs_list = range(5, 101, 5)

#mlp_size_list = [8, 16, 32]


i = 1

for FD in FD_list:
    for epochs in epochs_list:
        args['FD'] = FD
        args['epochs'] = epochs
        print("%d running"%(i))
        rmse, score = train(args)
        print("%d finish"%(i))
        i = i + 1
        data = np.array([[epochs, rmse, score]])
        result = np.concatenate((result,data))
            
    result = result[1:]  
    ##############
    test_loss_array = result[:, 1]
    test_score_array = result[:, 2]
    
    show_test_rmse(epochs, test_loss_array, FD)
    show_test_score(epochs, test_score_array, FD)
    
    result_df = pd.DataFrame(result, columns=['epochs', 'rmse', 'score'])
    result_df.to_csv('./%s_para_result.csv'%(FD),index = False)
    ###############
    result = np.array([[0,0,0]])




message = '调参：BERT_ft的epoch'  #邮件内容
Subject = '调参：BERT_ft的epoch'  #邮件主题描述


to_addrs = '1520572977@qq.com'  #实际收件人
# 填写真实的发邮件服务器用户名、密码
sender = '1520572977@qq.com'
user = '1520572977'
password = 'zqofvxdssddigfaj'
# 邮件内容
msg = MIMEText(message, 'plain', _charset="utf-8")
# 邮件主题描述
msg["Subject"] = Subject
# 发件人显示
msg["From"] = 'BERT_ft.py'
# 收件人显示
msg["To"] = 'Bin'

with SMTP_SSL(host="smtp.qq.com",port=465) as smtp:
    # 登录发邮件服务器
    smtp.login(user = user, password = password)
    # 实际发送、接收邮件配置
    smtp.sendmail(from_addr = sender, to_addrs=to_addrs, msg=msg.as_string())
 
        
    

    
    
