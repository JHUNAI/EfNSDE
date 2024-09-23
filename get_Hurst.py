# This code for R/S static.

import numpy as np
import akshare as ak


# plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族为Times New Roman

def s(inputdata):
    # 输入numpy数组
    n = inputdata.shape[0]
    t = 0
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                if inputdata[j] > inputdata[i]:
                    t = t + 1
                elif inputdata[j] < inputdata[i]:
                    t = t - 1
                else:
                    t = t
    return t


def beta(inputdata):
    n = inputdata.shape[0]
    t = []
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                t.append((inputdata[j] - inputdata[i]) / ((j - i) * 1.0))
    return np.median(t)


def Hurst(x):
    # x为numpy数组
    n = x.shape[0]
    t = np.zeros(n - 1)  # t为时间序列的差分
    for i in range(n - 1):
        t[i] = x[i + 1] - x[i]
    mt = np.zeros(n - 1)  # mt为均值序列,i为索引,i+1表示序列从1开始
    for i in range(n - 1):
        mt[i] = np.sum(t[0:i + 1]) / (i + 1)

    # Step3累积离差和极差,r为极差
    r = []
    for i in np.arange(1, n):  # i为tao
        cha = []
        for j in np.arange(1, i + 1):
            if i == 1:
                cha.append(t[j - 1] - mt[i - 1])
            if i > 1:
                if j == 1:
                    cha.append(t[j - 1] - mt[i - 1])
                if j > 1:
                    cha.append(cha[j - 2] + t[j - 1] - mt[i - 1])
        r.append(np.max(cha) - np.min(cha))
    s = []
    for i in np.arange(1, n):
        ss = []
        for j in np.arange(1, i + 1):
            ss.append((t[j - 1] - mt[i - 1]) ** 2)
        s.append(np.sqrt(np.sum(ss) / i))
    r = np.array(r)
    s = np.array(s)
    xdata = np.log(np.arange(2, n))
    ydata = np.log(r[1:] / s[1:])

    h, b = np.polyfit(xdata, ydata, 1)
    return h


# 20个中国股市代码列表
def stock_info():
    stock_codes = [
        "600000",  # 浦发银行（上海证券交易所）
        "600519",  # 贵州茅台（上海证券交易所）
        "601398",  # 工商银行（上海证券交易所）
        "601857",  # 中国石油（上海证券交易所）
        "601988",  # 中国银行（上海证券交易所）
        "600028",  # 中国石化（上海证券交易所）
        "600104",  # 上汽集团（上海证券交易所）
        "600276",  # 恒瑞医药（上海证券交易所）
        "600690",  # 海尔智家（上海证券交易所）
        "600887",  # 伊利股份（上海证券交易所）
        "000001",  # 平安银行（深圳证券交易所）
        "000333",  # 美的集团（深圳证券交易所）
        "000538",  # 云南白药（深圳证券交易所）
        "000651",  # 格力电器（深圳证券交易所）
        "000858",  # 五粮液（深圳证券交易所）
        "000895",  # 双汇发展（深圳证券交易所）
        "002415",  # 海康威视（深圳证券交易所）
        "002475",  # 立讯精密（深圳证券交易所）
        "002594",  # 比亚迪（深圳证券交易所）
        "300750",  # 宁德时代（深圳证券交易所）
    ]

    stock_codes_dic = {
        "600000": "浦发银行（上海证券交易所）",
        "600519": "贵州茅台（上海证券交易所）",
        "601398": "工商银行（上海证券交易所）",
        "601857": "中国石油（上海证券交易所）",
        "601988": "中国银行（上海证券交易所）",
        "600028": "中国石化（上海证券交易所）",
        "600104": "上汽集团（上海证券交易所）",
        "600276": "恒瑞医药（上海证券交易所）",
        "600690": "海尔智家（上海证券交易所）",
        "600887": "伊利股份（上海证券交易所）",
        "000001": "平安银行（深圳证券交易所）",
        "000333": "美的集团（深圳证券交易所）",
        "000538": "云南白药（深圳证券交易所）",
        "000651": "格力电器（深圳证券交易所）",
        "000858": "五粮液（深圳证券交易所）",
        "000895": "双汇发展（深圳证券交易所）",
        "002415": "海康威视（深圳证券交易所）",
        "002475": "立讯精密（深圳证券交易所）",
        "002594": "比亚迪（深圳证券交易所）",
        "300750": "宁德时代（深圳证券交易所）",
    }
    return stock_codes, stock_codes_dic


def pick_best_stock():
    stock_codes, stock_codes_dic = stock_info()  # 导入数据
    hursts = np.zeros_like(stock_codes, dtype=float)  # 初始化数组
    for i, stock_code in enumerate(stock_codes):
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code,
                                                period="daily", start_date="20230528", end_date='20240528', adjust="")
        stock_data = stock_zh_a_hist_df['收盘'].values
        hurst_index = Hurst(stock_data)
        hursts[i] = hurst_index
        # plt.plot(stock_data)
        # plt.title(f'{stock_codes_dic[stock_code]}')
        # plt.show()
        # print(f'Hurst Exponent: {hurst_index}')

    max_hursts = np.max(hursts)
    arg_hursts = np.argmax(hursts)

    stock_choice = stock_codes[arg_hursts]
    stock_info_choice = ak.stock_individual_info_em(symbol=stock_choice)

    stock_datas_choice = ak.stock_zh_a_hist(symbol=stock_choice,
                                            period="daily", start_date="20230528", end_date='20240528', adjust="")
    stock_data_choice = stock_datas_choice['收盘'].values
    print(f'Best Stock name: {stock_codes_dic[stock_choice]}, Hurst: {max_hursts:.3f}')
    # save the best stock
    save_path = f'./data/Best_stock_H_{max_hursts:.3f}.npy'
    np.save(save_path, stock_data_choice)
    return stock_info_choice, stock_datas_choice, stock_data_choice, max_hursts
    # 输出【股票信息， 股票数据们， 股票收盘价数据, 最大Hurst指数】


if __name__ == '__main__':
    # Compute all the prices in {stock_codes_dic}

    # stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol='300750',
    #                                         period="daily", start_date="20230528", end_date='20240528', adjust="")
    # time = stock_zh_a_hist_df['日期'].values
    # stock_data = stock_zh_a_hist_df['收盘'].values
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, stock_data)
    # plt.title(f'Stock closing price of CATL', fontsize=16)
    # plt.xlabel('Time', fontsize=14)
    # plt.ylabel('Price', fontsize=14)
    # plt.tight_layout()
    # plt.savefig('Stock_closing_price_of_CATL.eps', dpi=300)
    # plt.show()

    pick_best_stock()
