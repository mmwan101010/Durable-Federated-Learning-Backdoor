# smtplib 用于邮件的发信动作
from asyncio import SendfileNotAvailableError
from calendar import EPOCH
import smtplib
# email 用于构建邮件内容
from email.mime.text import MIMEText
# 构建邮件头
from email.header import Header
from sre_parse import State

def sendEmail(title, state, emailcontant):
    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = '1580524241@qq.com'
    password = 'xoyiazfnbpwjigbh'
    # 收信方邮箱
    to_addr = '389002792@qq.com'
    # 发信服务器
    smtp_server = 'smtp.qq.com'
    
        
    # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
    msg = MIMEText(emailcontant, 'plain', 'utf-8')
    # 邮件头信息
    msg['From'] = Header('neurotoxin实验进程通知')  # 发送者
    msg['To'] = Header('李四')  # 接收者
    subject = f"{title} + {state}"
    msg['Subject'] = Header(subject, 'utf-8')  # 邮件主题
    
    try:
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        # 建立连接--qq邮箱服务和端口号（可百度查询）
        smtpobj.connect(smtp_server, 465)    
        # 登录--发送者账号和口令
        smtpobj.login(from_addr, password)   
        # 发送邮件
        smtpobj.sendmail(from_addr, to_addr, msg.as_string()) 
        print("邮件发送成功")
    except smtplib.SMTPException:
        print("无法发送邮件")
    finally:
        # 关闭服务器
        smtpobj.quit()






