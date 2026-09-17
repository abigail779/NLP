import jieba
#加载自定义词典
jieba.load_userdict('./Data/user_dict.txt')
text = "小明毕业于北京大学计算机系"
#cut返回一个生成器
#jieba.cut()
#精确模式
#lcut返回一个列表
word_list = jieba.lcut(text)
print(word_list)

#全模式
print(jieba.lcut(text,cut_all=True))

#搜索引擎模式
jieba.lcut_for_search(text)


#自定义词典
word_list = jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策",cut_all=True)
print(word_list)




