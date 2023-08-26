# -*- coding: utf-8 -*-

# Bu dosya calistirilirken tum printler acilabilir
# Ama run_with_pkl.py dosyasini adimlara uyarak calistiriyorsaniz bu dosyaya dokunmayiniz

### Library
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import utils
from sklearn.metrics import mean_absolute_error
import config as cnf

"""Dataframe'i Ice Aktaralim"""


def data_prep(df):
    # Column isimlerini buyuk harf yapalım
    df.columns = [col.upper() for col in df.columns]
    print("---------------------------------")
    print(df.isnull().any())
    print("---------------------------------")

    """Veriyi Analiz Edelim"""

    """
    print("##################### SHAPE #####################")
    print(df.shape)
    print("##################### TYPES #####################")
    print(df.dtypes)
    print("##################### HEAD #####################")
    print(df.head())
    print("##################### TAIL #####################")
    print(df.tail())
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### DESCRIBE #####################")
    print(df.describe())
    """
    """Kategorik Ve Numerik Değişkenleri Ayiralim"""

    # kategorik ve kardinal degiskenler icin sinir degerler
    # Burada hedef kategorik gozukup kardinal olan kardinal gozukup kategorik olan degiskenleri ayirmak
    cat_cols, num_cols, cat_but_car = utils.grab_col_names(df,cat_th=5)

    # print(cat_cols)
    # print(num_cols)
    # print(cat_but_car)

    """Numerik Degiskenler Icin Outlier Kontolu Yapalim"""
    # Quantile degerleri %90 %10 olarak belirlendi
    for col_name in num_cols:
        low_limit, up_limit = utils.outlier_thresholds(df, col_name)
        if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
            print(f"\nOUTLIER {col_name}")
        else:
            print(f"\nNO PROBLEM {col_name}")
    # Degiskenlerde yukarıdaki fonksiyonla kayda deger bir outlier bulunamadi.
    # Verideki dogalligi bozmamak icin bu islem hassas bir sekilde yapildi.
    # Kullanilan fonksiyon kismi okunarak daha anlasilir olmasi umuluyor.

    # Local Outlier Kullanimi
    # LocalOutlierFactor kullanarak outlier degisken var mi kesin emin olalim
    train_df = df.select_dtypes(include=['float64', 'int64'])
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(train_df)
    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    df_scores = -df_scores
    np.sort(df_scores)[0:5]

    th = np.sort(df_scores)[3]
    train_df[df_scores < th]
    train_df[df_scores < th].shape
    train_df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

    # Outlierlarin indeksleri bulundu
    print(train_df[df_scores < th].index)
    #Outlier degerler drop ediliyor
    df.drop(index=df[df_scores < th].index, inplace=True)

    """Correlation Matrix ve Degiskenlerin Sikliklarini Kontrol Edelim"""

    # fig = plt.gcf()
    # fig.set_size_inches(10, 8)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    # plt.show(block=True)

    """Kategorik ve Numerik Degiskenleri Inceleyelim"""

    for num_col in num_cols:
        utils.num_summary(df,num_col,plot=False)
        new_cat_cols = [col for col in cat_cols if col not in ["SEX"]]
    for cat_col in new_cat_cols:
        utils.cat_summary(df,cat_col,plot=False)

    """Degiskenlerin Birbiriyle Iliskisini Inceleyelim"""

    # SMOKER and CHARGES
    # print(df.groupby("SMOKER")["CHARGES"].mean())
    # print("\n")
    # BMI and SEX
    print(df.groupby("SEX")["BMI"].mean())
    print("\n")
    # REGION and CHILDEREN
    print(df.groupby("REGION")["CHILDREN"].sum())
    print("\n")
    # BMI and CHILDREN
    print(df.groupby("CHILDREN")["BMI"].mean())
    print("\n")
    # SEX and CHARGES
    # print(df.groupby("SEX")["CHARGES"].mean())
    # print("\n")
    # CHARGES and CHILDREN
    # print(df.groupby("CHILDREN")["CHARGES"].mean())
    # print("\n")

    """Degiskenler Hakkinda Yeterince Bilgi Edindik
    Simdi Feature Extraction Yapalim
    """

    # Bir kategorik degisken olusturacagim genc orta yasli ve yasli diye
    df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'YOUNG'
    df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'MIDDELAGE'
    df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'OLD'

    df.loc[(df['BMI'] < 25), "NEW_BMI_CAT"] = 'NORMAL'
    df.loc[(df['BMI'] >= 25) & (df['BMI'] <= 30), "NEW_BMI_CAT"] = 'OVERWEIGHT'
    df.loc[(df['BMI'] > 30), "NEW_BMI_CAT"] = 'OBESITY'

    # Son degisken incelemesi
    # BMI and CHARGES
    # print(df.groupby("NEW_BMI_CAT")["CHARGES"].mean())
    # print("\n")

    """Simdi Degiskenleri Hazir Fonksiyonlarla Bir Daha Denetleyelim"""

    # Degiskenleri tiplerine gore ayirmistik bir daha ayiralim
    cat_cols, num_cols, cat_but_car = utils.grab_col_names(df,cat_th=5)

    """Verileri One Hot Encoderdan Gecirelim"""

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # print("##################### HEAD #####################")
    # print(df.head())
    # print("##################### TAIL #####################")
    # print(df.tail())

    """Verileri Olceklendirelim"""

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    # print("##################### HEAD #####################")
    # print(df.head())
    # print("##################### TAIL #####################")
    # print(df.tail())
    return df

"""Bagimli ve Bagimsiz Degiskenleri Ayiralim Train Test Ayrimini da Yapalim"""
df = pd.read_csv(cnf.medcost)
df = data_prep(df)

y = df["CHARGES"]
X = df.drop("CHARGES",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

"""Model Olarak DECISION TREE'yi alalim Hiperparametre optimizasyonu yapalim MSE ve MAE degerlerine bakalim"""

dt_regressor = DecisionTreeRegressor()

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(dt_regressor, cnf.param_grid_decision_tree, cv=5)
grid_search.fit(X_train, y_train)

# En iyi parametreleri görüntüleme
print("En iyi parametreler:", grid_search.best_params_)

# Test verisi üzerinde tahmin yapma
y_pred = grid_search.predict(X_test)

# Test verisi için MSE hesaplaması
mse = mean_squared_error(y_test, y_pred)
print("Test Verisi MSE:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("Test Verisi MAE:", mae)
# En iyi parametreler: {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 10}
# Test Verisi MSE: 0.18883719547006478

regressor = SVR()

""" Model Olarak DECISION TREE'yi alalim Hiperparametre optimizasyonu yapalim MSE ve MAE degerlerine bakalim """


# GridSearchCV ile hiperparametre optimizasyonu
# grid_search = GridSearchCV(regressor, cnf.param_grid_svr, cv=5)
# grid_search.fit(X_train, y_train)

# # En iyi parametreleri görüntüleme
# print("En iyi parametreler:", grid_search.best_params_)

# # Test verisi üzerinde tahmin yapma
# y_pred = grid_search.predict(X_test)

# # Test verisi için MSE hesaplaması
# mse = mean_squared_error(y_test, y_pred)
# print("Test Verisi MSE:", mse)

# En iyi parametreler: {'C': 10, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}
# Test Verisi MSE: 0.17501467328223763

"""Model Secildi 0.1750 MSE ye sahip Hiperparametre Optimizasyonu Yapilmis SVR Kullanilacak 
Bu Modeli Kuralim Icin Mean Absolute Error Hesaplatalim Sonuca Bakalim"""

svr_regressor = SVR(C = 10, epsilon = 0.1, gamma = 0.1, kernel = 'rbf')

# Modeli eğitme
svr_regressor.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = svr_regressor.predict(X_test)

# Test verisi için MAE hesaplaması
mae = mean_absolute_error(y_test, y_pred)
print("Test Verisi MAE:", mae)
mse = mean_squared_error(y_test, y_pred)
print("\nTest Verisi MSE:", mse)

"""Son Hali Icin SVR Yapildi (C = 10, epsilon = 0.1, gamma = 0.1, kernel = 'rbf') ve Bu Parametreler Kullanildi"""

# SVR icin pkl dosyasi olusturuldu ve ayni dizine kaydeildi
# joblib.dump(svr_regressor, "med_cost_analysis_svr.pkl")

"""Feature importance grafigi"""

# from sklearn.inspection import permutation_importance

# result = permutation_importance(svr_regressor, X_test, y_pred, n_repeats=30, random_state=42)
# importance_scores = result.importances_mean
# sorted_indices = np.argsort(importance_scores)[::-1]

# plt.barh(X_test.columns[sorted_indices], importance_scores[sorted_indices])
# plt.xlabel('Permütasyon Önem Skorları')
# plt.ylabel('Değişkenler')
# plt.title('Değişkenlerin Permütasyon Önem Skorlari')
# plt.show()

