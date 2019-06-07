# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:38:55 2019

@author: eddy
"""

import re

phoneNumRegex = re.compile(r"(\d{3})-(\d\d)-\d") #r = 使後面的字串變成原始字串
mo = phoneNumRegex.search('ttt aaa sss 123-55-1')
print("find normalization = ", mo.group())
print("find normalization = ", mo.group(1))
print("find normalization = ", mo.group(2))



#batRegex = re.compile(r'Bat(man|ttt|sss)')
#mo = batRegex.search("Batttt fdsa")
#print("batRegex find normalization = ", mo.group())
#print("batRegex find normalization = ", mo.group(1))
#
#
#manyRegex = re.compile(r"many(wo)*man")  # (wo)* --> 比對零次或多次
#mo = manyRegex.search("the add manyman")
#print("many find normalization = ", mo.group())
#
#mo = manyRegex.search("the add manywowoman")
#print("many find normalization = ", mo.group())
#
#mo = manyRegex.search("the add manywowowowowowowowowoman")
#print("many find normalization = ", mo.group())
#
#
#muchRegex = re.compile(r"much(wo)+man")
#mo = muchRegex.search("test muchwoman")
#print("much regex find normalization = ", mo.group())
#
#mo = muchRegex.search("test muchman")
#print("much regex find normalization = ", mo == None)
#
#
#haRegex = re.compile(r"(ha){2}")
#mo = haRegex.search("hahaha")
#print("ha regex find normalization = ", mo.group())
#
#
#greedyRegex = re.compile(r"(ha){3,5}")  #貪婪
#mo = greedyRegex.search('hahahahaha')
#print("greedy regex = ", mo.group())
#
#nongreedyRegex = re.compile(r"(ha){3,5}?") #非貪婪
#mo = nongreedyRegex.search("hahahahaha")
#print("nongreedy regex = ", mo.group())
#
#
#phoneNumRegex = re.compile(r"\d\d\d-\d\d-\d")
#mo = phoneNumRegex.findall('ttt aaa sss 123-55-1 and 888-77-2')
#print("find normalization = ", mo)
#
#
#vowelRegex = re.compile(r'[aeiouAEIOU]')
#print("find vowelRegex = ", vowelRegex.findall('Robocop eats food. apple Air'))
#
#vowelRegex = re.compile(r'[^aeiouAEIOU]')
#print("find vowelRegex = ", vowelRegex.findall('Robocop eats food. apple Air'))
#
#beginWithHello = re.compile(r'^hello')
#mo = beginWithHello.search('hello world!!!!')
#print("begin have hello? " , mo.group())
#
#mo2 = beginWithHello.search('this is hello.')
#print("begin have hello? " , mo2 == None)
#
#
#atRegex = re.compile(r".at")
#mo = atRegex.findall("the cat in the hat sat on the flat mat.")
#print(". at regex find normalization = ", mo)
#
#
#nameRegex = re.compile(r"first name: (.*) last name = (.*)") #貪婪 模式
#mo = nameRegex.search("qqq first name: 111 last name = 123 sss")
#print("name regex = ", mo.group())
#
#nongreedyRegex = re.compile(r'<.*?>') #非貪婪 模式
#mo = nongreedyRegex.search("<To serve man > for dinner.>")
#print("nongreedy Regex = ", mo.group())
#
#greedyRegex = re.compile(r'<.*>') #貪婪 模式
#mo = greedyRegex.search("<To serve man > for dinner.>")
#print("greedy Regex = ", mo.group())
#
#
#noNewlineRegex = re.compile('.*') 
#mo = noNewlineRegex.search("serve the public truct.\nasdfasdf .\nfdsafdsafdsa")
#print("nongreedy Regex = ", mo.group())
#
#NewlineRegex = re.compile('.*', re.DOTALL) #會尋找所有的內容 包含換行符號
#mo = NewlineRegex.search("serve the public truct.\nasdfasdf .\nfdsafdsafdsa")
#print("greedy Regex = ", mo.group())
#
#Regex = re.compile(r"robocop", re.I) #re.I 當作參數傳入 進行不區分大小寫
#mo = Regex.findall("the cat in the hat Robocop sat on the flat mat.")
#print(". at regex find normalization = ", mo)
#
#namesRegex = re.compile(r'eddy')
#print("regex sub = ", namesRegex.sub('censored', 'eddy have the secret document to eddy bob.'))
#
#namesRegex = re.compile(r'eddy')
#print("regex sub = ", namesRegex.sub(r'1***', 'eddy have the secret document to eddy bob.'))


