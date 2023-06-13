
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import csv
import time
import matplotlib.pyplot as plt

df = pd.read_csv("C:\dev/data.tsv", sep='\t')
df2 = pd.read_csv("C:\dev/data.tsv", sep='\t')

#with open('C:\dev/data2.tsv', 'wt') as out_file:
    #tsv_writer = csv.writer(out_file, delimiter='\t')
    #tsv_writer.writerow(['title', 'summary', 'score'])

#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

#Элемент предобработки. Заполняет Nan-ы
i = 0
while i < len(df['summary']):
    if type(df['summary'][i]) == float:
        df['summary'][i] = df['title'][i]
    i+=1

#Элемент предобработки. Заполняет Nan-ы
i = 0
while i < len(df2['summary']):
    if type(df2['summary'][i]) == float:
        df2['summary'][i] = df2['title'][i]
    i+=1

#Элемент предобработки. Убирает стоп-слова. Лемматизирует
morph = pymorphy2.MorphAnalyzer()


stop_words = set(stopwords.words('russian'))
stop_list = [',','.',':',';','"','.','-']
#,'«','»'
i = 0
while i < len(df['summary']):
    word_tokens = nltk.word_tokenize(df['summary'][i])
    word_tokens2 = nltk.word_tokenize(df['title'][i])

    filtered_sentence = ''
    filtered_sentence2 = ''

    for w in word_tokens:
        if (w not in stop_words) and (w not in stop_list):
            p = morph.parse(w)[0]
            filtered_sentence += ' ' + p.normal_form

    df['summary'][i] = filtered_sentence

    for w in word_tokens2:
        if w not in stop_words:
            p = morph.parse(w)[0]
            filtered_sentence2 += ' ' + p.normal_form

    df['title'][i] = filtered_sentence2
    i+=1

content = []
for text in df['title']:
    content.append(text)
for text in df2['title']:
    content.append(text)
for text in df['summary'].values.astype('U'):
    content.append(text)
for text in df2['summary'].values.astype('U'):
    content.append(text)

#Разбиение на тренировочный и тестовый наборы 
train, test = train_test_split(df, test_size=0.2, random_state=42)
train2, test2 = train_test_split(df2, test_size=0.2, random_state=42)

test_score_signs = []

for sc in test['score']:
    sign = -1
    if (sc) > 0:
        sign = 1
    test_score_signs.append(sign)


vectorizer = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()
vectorizer3 = TfidfVectorizer()
vectorizer4 = TfidfVectorizer()
vectorizerU = TfidfVectorizer()

#Векторизация входных данных
vectorizer.fit(df['title'])
vectorizer2.fit_transform(df['summary'].values.astype('U')) 

vectorizer3.fit(df2['title'])
vectorizer4.fit_transform(df2['summary'].values.astype('U'))
vectorizerU.fit_transform(content)

model = LinearRegression() #модель заголовок + предобработка + вект_заголовок
model2 = LinearRegression() #модель текст + предобработка + вект_текст

model3 = LinearRegression() #модель заголовок + предобработка + вект_текст
model4 = LinearRegression() #модель текст + предобработка + вект_заголовок

model5 = LinearRegression() #модель заголовок без предобработки + вект_заголовок
model6 = LinearRegression() #модель текст без предобработки + вект_текст

model7 = LinearRegression() #модель заголовок без предобработки + вект_текст
model8 = LinearRegression() #модель текст без предобработки + вект_заголовок


model9 = LinearRegression() #модель заголовок + предобработка + вект_у
model10 = LinearRegression() #модель текст + предобработка + вект_у

model11 = LinearRegression() #модель заголовок без предобработки + вект_у
model12 = LinearRegression() #модель текст без предобработки + вект_у


#Обучение модели
model.fit(vectorizer.transform(train['title']), train['score'])
model2.fit(vectorizer2.transform(train['summary'].values.astype('U')), train['score'])

model3.fit(vectorizer2.transform(train['title']), train['score'])
model4.fit(vectorizer.transform(train['summary'].values.astype('U')), train['score'])


model5.fit(vectorizer3.transform(train2['title']), train2['score'])
model6.fit(vectorizer4.transform(train2['summary'].values.astype('U')), train2['score'])

model7.fit(vectorizer4.transform(train2['title']), train2['score'])
model8.fit(vectorizer3.transform(train2['summary'].values.astype('U')), train2['score'])



#Обучение модели
model9.fit(vectorizerU.transform(train['title']), train['score'])
model10.fit(vectorizerU.transform(train['summary'].values.astype('U')), train['score'])

model11.fit(vectorizerU.transform(train2['title']), train2['score'])
model12.fit(vectorizerU.transform(train2['summary'].values.astype('U')), train2['score'])

#print(mean_squared_error(test['score'], model.predict(vectorizer.transform(test['title']))))
#0.23952362826629248

def CheckSigner(pred_list):
    i = 0
    cntr = 0
    for scr in test['score']:
        if (pred_list[i] * scr) > 0:
            cntr+=1
        i+=1

    return cntr

def AccFinder(clear_pred_t, proc_pred_t, clear_pred_s, proc_pred_s):

    test_len = len(test['title'])
    cs = (CheckSigner(clear_pred_t) + CheckSigner(clear_pred_s))/2 #среднее число угаданных тональностей без предобраб
    cs2 = (CheckSigner(proc_pred_t) + CheckSigner(proc_pred_s))/2 #среднее число угаданных тональностей с предобраб

    coef_clear = test_len / cs
    coef_proc = test_len / cs2

    clear_mse_err =(mean_squared_error(test2['score'], clear_pred_t ) + mean_squared_error(test2['score'], clear_pred_s ))/2
    proc_mse_err =(mean_squared_error(test['score'], proc_pred_t ) + mean_squared_error(test['score'], proc_pred_s ))/2

    clear_mae_err =(mean_absolute_error(test2['score'], clear_pred_t ) + mean_absolute_error(test2['score'], clear_pred_s ))/2
    proc_mae_err =(mean_absolute_error(test['score'], proc_pred_t ) + mean_absolute_error(test['score'], proc_pred_s ))/2

    Avg_clear_err = (clear_mse_err + clear_mae_err)/2 * coef_clear
    Avg_proc_err = (proc_mse_err + proc_mae_err)/2 * coef_proc

    acc_res = [Avg_clear_err, Avg_proc_err, clear_mse_err, clear_mae_err, proc_mse_err, proc_mae_err, 1/coef_clear, 1/coef_proc] 
    # 0 - средняя ошибка без предобр, 1 - средн ошибка с предобр. 2 - mse_clear, 3 - mae_clear, 4 - mse_proc, 5 - mae_proc, 6 - coef_clear, 7 - coef_proc

    return acc_res

#Оценка точности каскада. Принимает функцию каскада и режим оценки. Режимы - сырые данные и предобработанные
def CascadeAccFinder(Cascade, mode):
    texts1 = test2['title']
    texts2 = test2['summary']
    if mode == 1:
        texts1 = test['title']
        texts2 = test['summary']
    pred1 = []
    pred2 = []
    test_len = len(test['title'])
    for text in texts1:
        pred1.append(Cascade(text))
    for text in texts2:
        pred2.append(Cascade(text))

    cs = (CheckSigner(pred1) + CheckSigner(pred2))/2 #среднее число угаданных тональностей
    coef = test_len / cs
    mse_err =(mean_squared_error(test2['score'], pred1 ) + mean_squared_error(test2['score'], pred2 ))/2
    mae_err =(mean_absolute_error(test2['score'], pred1 ) + mean_absolute_error(test2['score'], pred2 ))/2
    Avg_err = (mse_err + mae_err)/2 * coef

    acc_res = [Avg_err, mse_err, mae_err, 1/coef]
    return acc_res

def M1Pred(text):
    p_all = []
    p_all.append( model.predict(vectorizer.transform([text])))
    p_all.append( model.predict(vectorizer.transform([Preproc_string(text)])))
    return p_all

def M1Acc():
    clear_pred_title = model.predict(vectorizer.transform(test2['title']))
    proc_pred_title = model.predict(vectorizer.transform(test['title']))
    clear_pred_sum = model.predict(vectorizer.transform(test2['summary']))
    proc_pred_sum = model.predict(vectorizer.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M2Acc():
    clear_pred_title = model2.predict(vectorizer2.transform(test2['title']))
    proc_pred_title = model2.predict(vectorizer2.transform(test['title']))
    clear_pred_sum = model2.predict(vectorizer2.transform(test2['summary']))
    proc_pred_sum = model2.predict(vectorizer2.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M3Acc():
    clear_pred_title = model3.predict(vectorizer2.transform(test2['title']))
    proc_pred_title = model3.predict(vectorizer2.transform(test['title']))
    clear_pred_sum = model3.predict(vectorizer2.transform(test2['summary']))
    proc_pred_sum = model3.predict(vectorizer2.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M4Acc():
    clear_pred_title = model4.predict(vectorizer.transform(test2['title']))
    proc_pred_title = model4.predict(vectorizer.transform(test['title']))
    clear_pred_sum = model4.predict(vectorizer.transform(test2['summary']))
    proc_pred_sum = model4.predict(vectorizer.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)
def M5Acc():
    clear_pred_title =  model5.predict(vectorizer3.transform(test2['title']))
    proc_pred_title = model5.predict(vectorizer3.transform(test['title']))
    clear_pred_sum = model5.predict(vectorizer3.transform(test2['summary']))
    proc_pred_sum = model5.predict(vectorizer3.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M6Acc():
    clear_pred_title =  model6.predict(vectorizer4.transform(test2['title']))
    proc_pred_title = model6.predict(vectorizer4.transform(test['title']))
    clear_pred_sum = model6.predict(vectorizer4.transform(test2['summary']))
    proc_pred_sum = model6.predict(vectorizer4.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M7Acc():
    clear_pred_title =  model7.predict(vectorizer4.transform(test2['title']))
    proc_pred_title = model7.predict(vectorizer4.transform(test['title']))
    clear_pred_sum = model7.predict(vectorizer4.transform(test2['summary']))
    proc_pred_sum = model7.predict(vectorizer4.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M8Acc():
    clear_pred_title =  model8.predict(vectorizer3.transform(test2['title']))
    proc_pred_title = model8.predict(vectorizer3.transform(test['title']))
    clear_pred_sum = model8.predict(vectorizer3.transform(test2['summary']))
    proc_pred_sum = model8.predict(vectorizer3.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M9Acc():
    clear_pred_title =  model9.predict(vectorizerU.transform(test2['title']))
    proc_pred_title = model9.predict(vectorizerU.transform(test['title']))
    clear_pred_sum = model9.predict(vectorizerU.transform(test2['summary']))
    proc_pred_sum = model9.predict(vectorizerU.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M10Acc():
    clear_pred_title =  model10.predict(vectorizerU.transform(test2['title']))
    proc_pred_title = model10.predict(vectorizerU.transform(test['title']))
    clear_pred_sum = model10.predict(vectorizerU.transform(test2['summary']))
    proc_pred_sum = model10.predict(vectorizerU.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M11Acc():
    clear_pred_title =  model11.predict(vectorizerU.transform(test2['title']))
    proc_pred_title = model11.predict(vectorizerU.transform(test['title']))
    clear_pred_sum = model11.predict(vectorizerU.transform(test2['summary']))
    proc_pred_sum = model11.predict(vectorizerU.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

def M12Acc():
    clear_pred_title =  model12.predict(vectorizerU.transform(test2['title']))
    proc_pred_title = model12.predict(vectorizerU.transform(test['title']))
    clear_pred_sum = model12.predict(vectorizerU.transform(test2['summary']))
    proc_pred_sum = model12.predict(vectorizerU.transform(test['summary']))

    return AccFinder(clear_pred_title, proc_pred_title, clear_pred_sum, proc_pred_sum)

print(M1Acc())
print(M2Acc())
print(M3Acc())
print(M4Acc())
print(M5Acc())
print(M6Acc())
print(M7Acc())
print(M8Acc())
print(M9Acc())
print(M10Acc())
print(M11Acc())
print(M12Acc())


MetricsList = [M1Acc(), M2Acc(),M3Acc(),M4Acc(),M5Acc(),M6Acc(),M7Acc(),M8Acc(),M9Acc(),M10Acc(),M11Acc(),M12Acc()]

def PlotterInterface(title, metNum, metNum2, filename, legend, AccList):
    i = 0
    pldt = []
    pldt2 = []
    plnum = []
    lnAcc = len(AccList)
    while i < lnAcc:
        if metNum2 != -1:
            pldt2.append(AccList[i][metNum2])
        pldt.append(AccList[i][metNum])
        plnum.append(i+1)
        i+=1
    if metNum2 != -1:
        plt.axis([0,lnAcc + 1,0,max([max(pldt), max(pldt2)]) + 0.2])
    else:
        plt.axis([0,13,0,max(pldt) + 0.2])
    plt.title(title, fontsize=24, fontname='Times New Roman')
    plt.xlabel('Номер модели', color='gray', fontsize=20)
    plt.ylabel('Значение ошибки',color='gray', fontsize=20)
    i = 0
    while i < lnAcc:
        if metNum2 != -1:
            plt.text(i + 0.81,max([AccList[i][metNum],AccList[i][metNum2]]) + 0.1, i+1)
        else:
            plt.text(i + 0.81,AccList[i][metNum] + 0.1, i+1)
        i+=1
    plt.plot(plnum,pldt, linewidth=1.5)
    plt.plot(plnum,pldt, 'ro')
    if metNum2 != -1:
        plt.plot(plnum,pldt2, linewidth=1.5)
        plt.plot(plnum,pldt2, 'go')
    plt.grid(True)
    plt.legend(legend)
    plt.savefig(filename)
    plt.show()
    # plt.close()
    return 1

PlotterInterface('Гибридная ошибка. Сопоставление', 0, 1, 'AvgErr.png', ['На сырых данных','Значение на сырых данных','На обработанных данных','Значение на обработанных данных'], MetricsList)

Clear_acc_list = []
Proc_acc_list = []

i = 0
for metr in MetricsList:
    Clear_acc_list.append((metr[0], i))
    Proc_acc_list.append((metr[1], i))
    i+=1

Clear_acc_list.sort(key=lambda x: x[0])
Proc_acc_list.sort(key=lambda x: x[0])

#Единый интерфейс моделей. Номер модели, текст, режим
def GetModelNumRes(num, text, mode):
    txt = text
    if mode == 1:
        txt = Preproc_string(text)

    if num == 0:
        return model.predict(vectorizer.transform([txt]))
    elif num == 1:
        return model2.predict(vectorizer2.transform([txt]))
    elif num == 2:
        return model3.predict(vectorizer2.transform([txt]))
    elif num == 3:
        return model4.predict(vectorizer.transform([txt]))
    elif num == 4:
        return model5.predict(vectorizer3.transform([txt]))
    elif num == 5:
        return model6.predict(vectorizer4.transform([txt]))
    elif num == 6:
        return model7.predict(vectorizer4.transform([txt]))
    elif num == 7:
        return model8.predict(vectorizer3.transform([txt]))
    elif num == 8:
        return model9.predict(vectorizerU.transform([txt]))
    elif num == 9:
        return model10.predict(vectorizerU.transform([txt]))
    elif num == 10:
        return model11.predict(vectorizerU.transform([txt]))
    elif num == 11:
        return model12.predict(vectorizerU.transform([txt]))
    else:
        print('Ошибка. Интерфейс. Не тот номер')
    return 1


#Значение по каскаду без обработок
def Get_Clear_Cascade(text):
    Acc, num = Clear_acc_list[0]
    res = GetModelNumRes(num, text, 0)
    Acc, num = Clear_acc_list[1]
    res += GetModelNumRes(num, text, 0)
    Acc, num = Clear_acc_list[2]
    res += GetModelNumRes(num, text, 0)
    return res/3

#Значение по каскаду с обработками
def Get_Proc_Cascade(text):
    Acc, num = Proc_acc_list[0]
    res = GetModelNumRes(num, text, 1)
    Acc, num = Proc_acc_list[1]
    res += GetModelNumRes(num, text, 1)
    Acc, num = Proc_acc_list[2]
    res += GetModelNumRes(num, text, 1)
    return res/3

#Получение значения по каскаду
def Get_Cascade(text):
    return (Get_Clear_Cascade(text) + Get_Proc_Cascade(text))/2

def Preproc_string(text):
    word_tok = nltk.word_tokenize(text)
    filtered_sent = ''

    for w in word_tok:
        if (w not in stop_words) and (w not in stop_list):
            p = morph.parse(w)[0]
            filtered_sent += ' ' + p.normal_form
    return filtered_sent

print('')

cascade_metrics = [CascadeAccFinder(Get_Clear_Cascade, 0), CascadeAccFinder(Get_Clear_Cascade, 1),
                  CascadeAccFinder(Get_Proc_Cascade, 0), CascadeAccFinder(Get_Proc_Cascade, 1),
                  CascadeAccFinder(Get_Cascade, 0), CascadeAccFinder(Get_Cascade, 1)]

print(cascade_metrics)
cascade_metrics2 = [cascade_metrics[0] + cascade_metrics[1],
                   cascade_metrics[2] + cascade_metrics[3],
                   cascade_metrics[4] + cascade_metrics[5]]
PlotterInterface('Гибридная ошибка. Каскады', 0, 4, 'AvgErr_Cascades.png', ['На сырых данных','Значение на сырых данных','На обработанных данных','Значение на обработанных данных'], cascade_metrics2)

Cascade_accList = []

i = 0
for metr in cascade_metrics:
    Cascade_accList.append((metr[0], i))
    i+=1

Cascade_accList.sort(key=lambda x: x[0])

def GetBestSentiment(text):

    accdata, num = Cascade_accList[0]

    if num == 0:
        return Get_Clear_Cascade(text)
    elif num == 1:
        return Get_Clear_Cascade(Preproc_string(text))
    elif num == 2:
        return Get_Proc_Cascade(text)
    elif num == 3:
        return Get_Proc_Cascade(Preproc_string(text))
    elif num == 4:
        return Get_Cascade(text)
    elif num == 5:
        return Get_Cascade(Preproc_string(text))
    return 1




df3 = pd.read_csv("C:\dev/data_fin.tsv", sep='\t')
i = 0
while i < len(df3['open']):
    print(i)
    open2 = float(df3['open'][i])
    close = float(df3['close'][i])
    df3['open'][i] = ( close - open2) / open2 * 100
    i+=1

X = df3[['score']].values
y = df3['open'].values

modelSF = LinearRegression()
trX, teX, trY, teY = train_test_split(X, y, test_size=0.3, random_state=42)

modelSF.fit(trX, trY)

print(mean_squared_error(teY, modelSF.predict(teX)))
print(mean_absolute_error(teY, modelSF.predict(teX)))

def GetFin(text):
    return modelSF.predict([GetBestSentiment(text)])

def GetFinFromSent(sent):
    return modelSF.predict([sent])



news = []
news.append("Илон Маск предсказал чудовищное падение акций Google")
news.append("Сокрушительное землетрясение уничтожило завод Tesla")
news.append("Сокрушительное землетрясение уничтожило завод Tesla. Материальный ущерб превысил $130 млн")
news.append("Котировки компании Google взлетели на 13%")
news.append("Прибыль компании Google в этом году составила $450 млн")
news.append("Финансовая нестабильность в регионе ведёт к рискам нового финансового кризиса")
news.append("Ожидается укрепление позиций российских компаний на фондовых рынках")
news.append("Илон Маск вложил деньги в новый стартап. Ожидаемая прибыль составит 13$ млн")
news.append("Яндекс увеличил прибыль на $1 млрд")
news.append("Рекордный убыток компании Рейнметалл в этом году составил $43 млн")
news.append("При покупке отечественного автомобиля каждому россиянину дадут по 1 млн")
#Предсказываем результаты
#print(model.predict(vectorizer.transform(["Рекордный убыток компании Рейнметалл в этом году составил $43 млн"])))
#print(model2.predict(vectorizer.transform(["Рекордный убыток компании Рейнметалл в этом году составил $43 млн"])))

res_an = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # какие метрики сколько угадали
res_news_an = [0,0,0,0,0,0,0,0,0,0,0] #сколько какие новости угаданы
res_news = [-1,-1,-1,1,1,-1,1,1,1,-1,1] # целевые знаки новостей
res = []

i = 0
while i < len(news):
    print('Сентимент:' + str(GetBestSentiment(news[i])))
    print('Показатель:' + str(GetFin(news[i])))
    i+=1

i = 0
while i < len(news):
    subres = []
    subres.append(model.predict(vectorizer.transform([news[i]])))
    subres.append(model.predict(vectorizer.transform([Preproc_string(news[i])])))
    subres.append(model2.predict(vectorizer2.transform([news[i]])))
    subres.append(model2.predict(vectorizer2.transform([Preproc_string(news[i])])))
    subres.append(model3.predict(vectorizer2.transform([news[i]])))
    subres.append(model3.predict(vectorizer2.transform([Preproc_string(news[i])])))
    subres.append(model4.predict(vectorizer.transform([news[i]])))
    subres.append(model4.predict(vectorizer.transform([Preproc_string(news[i])])))
    subres.append(model5.predict(vectorizer3.transform([news[i]])))
    subres.append(model5.predict(vectorizer3.transform([Preproc_string(news[i])])))
    subres.append(model6.predict(vectorizer4.transform([news[i]])))
    subres.append(model6.predict(vectorizer4.transform([Preproc_string(news[i])])))
    subres.append(model7.predict(vectorizer4.transform([news[i]])))
    subres.append(model7.predict(vectorizer4.transform([Preproc_string(news[i])])))
    subres.append(model8.predict(vectorizer3.transform([news[i]])))
    subres.append(model8.predict(vectorizer3.transform([Preproc_string(news[i])])))

    subres.append(model9.predict(vectorizerU.transform([news[i]])))
    subres.append(model9.predict(vectorizerU.transform([Preproc_string(news[i])])))
    subres.append(model10.predict(vectorizerU.transform([news[i]])))
    subres.append(model10.predict(vectorizerU.transform([Preproc_string(news[i])])))
    subres.append(model11.predict(vectorizerU.transform([news[i]])))
    subres.append(model11.predict(vectorizerU.transform([Preproc_string(news[i])])))
    subres.append(model12.predict(vectorizerU.transform([news[i]])))
    subres.append(model12.predict(vectorizerU.transform([Preproc_string(news[i])])))

    j = 0
    while j < len(subres):
        if subres[j] * res_news[i] > 0 :
            res_an[j] += 1
            res_news_an[i] +=1
        j+=1
    #print(news[i])

    res.append(subres)

    i+=1
print(res_an)
print(res_news_an)

i =0
while i < len(res_an):
    if res_an[i] > 7:
        j = 0
        print(i)
        while j < len(res_news_an):
            print(res[j][i])
            j+=1
        print(' ')
    i+=1

def get_res(text):
    subres = []
    subres.append(model.predict(vectorizer.transform([Preproc_string(text)])))
    subres.append(model3.predict(vectorizer2.transform([text])))
    subres.append(model3.predict(vectorizer2.transform([Preproc_string(text)])))
    subres.append(model6.predict(vectorizer4.transform([text])))
    subres.append(model8.predict(vectorizer3.transform([text])))
    subres.append(model12.predict(vectorizerU.transform([text])))

    res = 0
    for val in subres:
        res+= val
    res = res / 6

    return res

print(' ')
print(' ')

i = 0
while i < len(news):
    print(get_res(news[i]))
    i+=1


#Элемент предобработки. Заполняет Nan-ы
i = 0
while i < len(df2['summary']):
    if type(df2['summary'][i]) == float:
        df2['summary'][i] = df2['title'][i]
    i+=1

nltk_tokenizer = RegexpTokenizer(r'[а-яёa-z]+')

def find_features(document):
    words = []
    doc = nltk_tokenizer.tokenize(document.lower())
    for word in doc:
        if not word in stop_words:
            p = morph.parse(word)[0]
            words.append(p.normal_form)
    words = set(words)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

def text_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features[word.lower()] = True
    return features


i = 0

all_words1 = []
while i < len(df2['summary']):
    words = nltk_tokenizer.tokenize(df2['summary'][i].lower())
    words2 = nltk_tokenizer.tokenize(df2['title'][i].lower())
    for w in words:
        all_words1.append(w.lower())
    for w in words2:
        all_words1.append(w.lower())

    i+=1

print('Слова посчитаны')

# Получение частоты употребления для всех слов 
all_words1 = nltk.FreqDist(all_words1)

all_words = {}

for word in all_words1.keys():
    if not word in stop_words:
        p = morph.parse(word)[0]
        all_words[p.normal_form] = all_words1[word]

print('Слова чет там')

# взять первые 3000 наиболее значимых частых слов 
word_features = list(all_words.keys())[:2000]

print('фичи сделаны')

dataset1 = [] #title
dataset2 = [] #summary

i = 0
while i < len(df2['summary']):
    categ = 'pos'
    if df2['score'][i] < 0:
        categ = 'neg'

    text = df2['summary'][i]
    text2 = df2['title'][i]
    dataset2.append((find_features(text), categ))
    dataset1.append((find_features(text2), categ))

    i+=1

print('датасеты есть')

# Тренировочный набор данных
training_data = dataset1[:450]
 
# Тестирующий набор данных
test_data = dataset1[450:]
# Тренировочный набор данных
training_data2 = dataset2[:450]
 
# Тестирующий набор данных
test_data2 = dataset2[450:]

print('наборы сделаны')

classifier = nltk.NaiveBayesClassifier.train(training_data)
print("Процент точности классификатора:",(nltk.classify.accuracy(classifier, test_data))*100)
classifier2 = nltk.NaiveBayesClassifier.train(training_data2)
print("Процент точности классификатора:",(nltk.classify.accuracy(classifier, test_data2))*100)
classifier3 = nltk.NaiveBayesClassifier.train(training_data2 + training_data)
print("Процент точности классификатора:",(nltk.classify.accuracy(classifier, test_data + test_data2))*100)

print('модели обучены')

news = []
news.append("Илон Маск предсказал чудовищное падение акций Google")
news.append("Сокрушительное землетрясение уничтожило завод Tesla")
news.append("Сокрушительное землетрясение уничтожило завод Tesla. Материальный ущерб превысил $130 млн")
news.append("Котировки компании Google взлетели на 13%")
news.append("Прибыль компании Google в этом году составила $450 млн")
news.append("Финансовая нестабильность в регионе ведёт к рискам нового финансового кризиса")
news.append("Ожидается укрепление позиций российских компаний на фондовых рынках")
news.append("Илон Маск вложил деньги в новый стартап. Ожидаемая прибыль составит 13$ млн")
news.append("Яндекс увеличил прибыль на $1 млрд")
news.append("Рекордный убыток компании Рейнметалл в этом году составил $43 млн")
news.append("При покупке отечественного автомобиля каждому россиянину дадут по 1 млн")

i = 0
while i < len(news):
    my_data = text_features(news[i])

    print(classifier.classify(my_data))
    print(classifier2.classify(my_data))
    print(classifier3.classify(my_data))
    print(news[i])

    i+=1

def Result(text):
    res = []
    res.append(classifier2.classify(text_features(text))) #По Байесу
    gr = GetBestSentiment(text)[0]#get_res(text)
    sent = 'pos'
    if gr < 0:
        sent = 'neg'
    res.append(sent) # По лин.регр
    res.append(gr) # знач лин регр

    gr2 = gr
    gr3 = ''
    if gr < 0:
        gr2 *=-1

    if gr2 < 0.99:
        gr3 = 'Сверхзначительная'
    if gr2 < 0.85:
        gr3 = 'Очень значительная'
    if gr2 < 0.60:
        gr3 = 'Значительная'
    if gr2 < 0.40:
        gr3 = 'Ощутимая'
    if gr2 < 0.20:
        gr3 = 'Нормальная'
    if gr2 < 0.10:
        gr3 = 'Низкая'
    if gr2 < 0.05:
        gr3 = 'Малая'
    if gr2 < 0.005:
        gr3 = 'Незначительная'

    res.append(gr3)

    gr2 = GetFin(text)[0]

    res.append(gr2)

    return res

import telebot;
from telebot import types
bot = telebot.TeleBot('Token');


users = {}
user_text = {}
# 1 - гл. меню, 2 - набор новости, 3 - набор полной новости

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
  try:
    if message.text == "Привет":
        users[message.from_user.id] = 1
        bot.send_message(message.from_user.id, "Привет, чем я могу тебе помочь?")
    elif message.text == "/start":
        users[message.from_user.id] = 1
        bot.send_message(message.from_user.id, "Здравстуйте, я могу провести анализ текста экономической новости из соцсети.")
        keyboard = types.InlineKeyboardMarkup(); #наша клавиатура
        key_yes = types.InlineKeyboardButton(text='Начать', callback_data='start'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        bot.send_message(message.from_user.id, text='Начнём?', reply_markup=keyboard)
        print(message.from_user.id)
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напишите /start")
    elif users[message.from_user.id] == 3:
        #bot.send_message(message.from_user.id, "Напишите /start")

        msg = message.text.split('\n')
        msg2 = ''
        i = 0
        while i < len(msg):
            if (msg[i] != ' ') and (msg[i] != ''):
                msg2 += msg[i] + ' '
            i+=1

        regr_v = get_res(user_text[message.from_user.id])
        regr_v2 = get_res(msg2)

        with open('C:\dev/data7.tsv', newline='') as f:
            f.readline()
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)

        output = []

        for line in data:
            if len(line) > 0:
                title, summary, score = line
                output.append([title, summary, score])

        with open('C:\dev/data7.tsv', 'w', newline='') as f:
            headers = ['title', 'summary', 'score']
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(headers)
            writer.writerows(output)
            writer.writerow([user_text[message.from_user.id], msg2, str((regr_v + regr_v2)/2)])

        keyboard = types.InlineKeyboardMarkup(); #наша клавиатура
        key_yes = types.InlineKeyboardButton(text='Начать', callback_data='start'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        bot.send_message(message.from_user.id, text='Начнём?', reply_markup=keyboard)


    elif users[message.from_user.id] == 2:
        user_text[message.from_user.id] = message.text

        keyboard = types.InlineKeyboardMarkup(); #наша клавиатура
        key_yes = types.InlineKeyboardButton(text='По Байесу', callback_data='Beyes'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        key_yes = types.InlineKeyboardButton(text='По регрессии', callback_data='Reg'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        key_yes = types.InlineKeyboardButton(text='Значение регрессии', callback_data='Reg_val'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        key_yes = types.InlineKeyboardButton(text='Важность', callback_data='Weight'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        key_yes = types.InlineKeyboardButton(text='Показатель', callback_data='Pok'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру

        key_yes = types.InlineKeyboardButton(text='Новый текст', callback_data='start'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру
        key_yes = types.InlineKeyboardButton(text='Ввести полный текст', callback_data='New'); #кнопка «Да»
        keyboard.add(key_yes); #добавляем кнопку в клавиатуру

        bot.send_message(message.from_user.id, text='Можно получить такие общие данные по тексту:', reply_markup=keyboard)
        #bot.send_message(message.from_user.id, str(len(message.text)))
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")
  except Exception as e:
            print(e) 
            time.sleep(4)


@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    if call.data == "start": 
        try:
            print(call.message.chat.id)
            users[call.message.chat.id] = 2
            bot.send_message(call.message.chat.id, 'Введите текст новости для анализа:');
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "New":
        try:
         users[call.message.chat.id] = 3
         bot.send_message(call.message.chat.id, "Введите текст:");
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "Beyes":
        try:
         rs = Result(user_text[call.message.chat.id])
         if rs[0] == 'pos':
            bot.send_message(call.message.chat.id, "Прогноз благоприятный");
         else:
            bot.send_message(call.message.chat.id, "Прогноз неблагоприятный");
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "Reg":
        try:
         rs = Result(user_text[call.message.chat.id])
         if rs[1] == 'pos':
            bot.send_message(call.message.chat.id, "Прогноз благоприятный");
         else:
            bot.send_message(call.message.chat.id, "Прогноз неблагоприятный");
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "Reg_val":
        try:
         rs = Result(user_text[call.message.chat.id])
         bot.send_message(call.message.chat.id, "Значение регрессии: "+str(rs[2]));
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "Weight":
        try:
         rs = Result(user_text[call.message.chat.id])
         bot.send_message(call.message.chat.id, 'Значимость новости: ' + str(rs[3]));
        except Exception as e:
            print(e) 
            time.sleep(4)
    elif call.data == "Pok":
        try:
         rs = Result(user_text[call.message.chat.id])
         bot.send_message(call.message.chat.id, 'Ожидается падение акций на ' + str(rs[4]) + '%.');
        except Exception as e:
            print(e) 
            time.sleep(4)


while True:
    try:
        bot.polling(none_stop=True)

    except Exception as e:
        print(e) 
      
        time.sleep(4)
#bot.polling(none_stop=True, interval=0)