import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
%matplotlib inline

from sklearn.linear_model import LinearRegression as LR

from sklearn import preprocessing

data = pd.read_excel("SUUMOスクレイピング.xlsx")

#データを分割
data["徒歩分"] = data["アクセス"].apply(lambda x : x.split(" ")[1])
data["徒歩"] = data["徒歩分"].apply(lambda x : x.split('分')[0])
data["walk"] = data["徒歩"].apply(lambda x : x.split('歩')[1])

data["walk"] = data["walk"].astype(np.int)

data["家賃金額"] = data["家賃"].apply(lambda x : x.split("万円")[0])

data["平米数"] = data["面積"].apply(lambda x : x.split("m2")[0])

data["築年数test"] = data["築年数"].apply(lambda x : x.split("年")[0])
data["築年数数字"] = data["築年数test"].apply(lambda x : x.split("築")[1])

data['walk'] = data['walk'].astype('int')
data["平米数"] = data["平米数"].astype('float')

replace = data["築年数数字"].replace("新築", 0)
data_updated = data.replace("新築", "0")
data_updated2 = data.replace("", "0")

data_updated2["築年数数字"] = data_updated2["築年数数字"].astype('int')
data_updated2["家賃金額"] = data_updated2["家賃金額"].astype('float')

data_complete = data_updated2

df = data_complete

df.to_excel("all-data.xlsx")

train = pd.read_excel("train-suumo.xlsx")
test = pd.read_excel("test-suumo.xlsx")

mm = preprocessing.MinMaxScaler()

train_test = train[["築年数数字", "平米数", "walk"]]

train_nm = mm.fit_transform(train_test)

df_train = pd.DataFrame(train_nm)

df = df_train
df_train = df.rename(columns={0: "築年数数字", 1: "平米数", 2: "walk"})

test = test[["築年数数字", "平米数", "walk"]]

test_nm = mm.fit_transform(test)

df_test = pd.DataFrame(test_nm)

df2 = df_train
df_test = df2.rename(columns={0: "築年数数字", 1: "平米数", 2: "walk"})

model = LR()

target_data = pd.DataFrame(train["家賃金額"])
model.fit(target_data, df_train)

model.coef_

model.intercept_

model.score(target_data, df_train)

#後で確認
plt.scatter(data_complete["家賃金額"], data_complete["平米数"])

user_input_m2 = input("平米数を書いてね（数字だけでいいよ）：")
user_input_m2 = int(user_input_m2)

user_input_yachin = int(input("家賃を書いてね(数字だけでいいよ)："))
user_input_yachin = int(user_input_yachin)

new_input_data = pd.DataFrame(
    data={'家賃金額': user_input_yachin, 
          '平米数': user_input_m2},
    index=[0]
)

origin_data = data_complete[["家賃金額", "平米数"]]
merged_data= origin_data.append(new_input_data)

#散布図の作成

plt.title('家賃と平米数の相関図',
                      fontsize=20) # タイトル
plt.xlabel("平米数（m2）", fontsize=20) # x軸ラベル
plt.ylabel("家賃（万円）", fontsize=20) # y軸ラベル
plt.grid(True) # 目盛線の表示
plt.tick_params(labelsize = 12) # 目盛線のラベルサイズ

plt.scatter(merged_data["家賃金額"], merged_data["平米数"], c="b", label="世の中の平均")

#散布図の関数化
plot_data = plt.scatter(merged_data["家賃金額"], merged_data["平米数"], c="b", label="世の中の平均")

plt.scatter(merged_data["家賃金額"], merged_data["平米数"], c="b", label="世の中の平均")
plt.scatter(user_input_yachin, user_input_m2, c="r", label="あなたのデータ") 