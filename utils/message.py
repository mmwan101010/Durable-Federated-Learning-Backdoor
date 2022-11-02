# coding=utf-8
from unicodedata import name
import urllib
import urllib.request
import hashlib
import datetime

def finish(epoch, run_name):
    def md5(str):
        import hashlib
        m = hashlib.md5()
        m.update(str.encode("utf8"))
        return m.hexdigest()

    statusStr = {
        '0': '短信发送成功',
        '-1': '参数不全',
        '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
        '30': '密码错误',
        '40': '账号不存在',
        '41': '余额不足',
        '42': '账户已过期',
        '43': 'IP地址限制',
        '50': '内容含有敏感词'
    }

    strTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = 'imomoe'
    # 短信平台密码
    password = md5('a482f88f10084a2389cbe1986b7e48ea')
    # 要发送的短信内容
    content = '【neurotoxin】您的实验已完成，epochs：' + str(epoch) + '，runName：' + run_name + '\n' + strTime
    # 要发送短信的手机号码
    phone = '13098517357'


    print(content)
    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print (statusStr[the_page])

def will_finish(epoch, run_name):
    def md5(str):
        import hashlib
        m = hashlib.md5()
        m.update(str.encode("utf8"))
        return m.hexdigest()

    statusStr = {
        '0': '短信发送成功',
        '-1': '参数不全',
        '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
        '30': '密码错误',
        '40': '账号不存在',
        '41': '余额不足',
        '42': '账户已过期',
        '43': 'IP地址限制',
        '50': '内容含有敏感词'
    }

    strTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = 'imomoe'
    # 短信平台密码
    password = md5('a482f88f10084a2389cbe1986b7e48ea')
    # 要发送的短信内容
    content = '【neurotoxin】您的实验' + run_name + '，开始' + str(epoch) + 'epoch，即将完成，请注意。' + strTime
    # 要发送短信的手机号码
    phone = '13098517357'


    print(content)
    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print (statusStr[the_page])

