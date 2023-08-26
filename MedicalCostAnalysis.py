# -*- coding: utf-8 -*-

### Library
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

"""Kullanacagim Outlier Inceleme Ile Ilgili Fonksiyon Burada Quantile Hesaplamasi Yapilacak"""

# Burada q1=0.05,q3=0.95 degerleri cok fazla ucta olan degerleri kirpmak icin bu sekilde kullanilmistir.
# Keyfidir degistirilip model uzerindeki etkisi denenebilir.
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Parameters
    ------
        dataframe: dataframe
                Degisken isimleri alınmak istenilen dataframe
        col_name: list
                numerik degisken verilerini iceren bir sutun listesi
        q1: float, optional
                outlier sinir degerini ifade eden deger varsayilan olarak 0.05
        q1: float, optional
                outlier sinir degerini ifade eden deger varsayilan olarak 0.95
        Returns
    ------
        low_limit: int or float
                Kategorik degişken listesi
        up_limit: int or float
                Numerik degisken listesi icerisinde outlier sinir degeri

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        cat_summary(df)


    Notes
    ------
        Ceyreklikleri hesaplar
        Alt limiti ve ust limiti return eder
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

"""Kategorik ve Numerik Degiskenleri Incelemek Icin Birer Fonksiyon Yazildi


"""

def cat_summary(dataframe, col_name, plot=False):
    """
    Parameters
    ------
        dataframe: dataframe
                Degisken isimleri alinmak istenilen dataframe
        col_name: list
                kategorik degisken verilerini iceren bir sutun listesi
        plot: Bool, optional
                Grafik seklinde dagilim gorulmek isteniyorsa True yapilmalidir.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        cat_summary(df)


    Notes
    ------
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olamamktadir.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    """
    Parameters
    ------
        dataframe: dataframe
                Degisken isimleri alinmak istenilen dataframe
        numerical_col: list
                numerik degisken verilerini iceren bir sutun listesi
        plot: Bool, optional
                Grafik seklinde dagilim gorulmek isteniyorsa True yapilmalidir.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        num_summary(df)


    Notes
    ------
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olamamaktadir.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

"""Sutunlarin Degisken Turlerini Bulan Fonksiyon"""

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alinmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sinif eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sinif eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

"""Dataframe'i Ice Aktaralim"""

df = pd.read_csv('insurance.csv')
# Column isimlerini buyuk harf yapalım
df.columns = [col.upper() for col in df.columns]
print("---------------------------------")
print(df.isnull().any())
print("---------------------------------")

"""Veriyi Analiz Edelim"""

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

"""Kategorik Ve Numerik Değişkenleri Ayiralim"""

# kategorik ve kardinal degiskenler icin sinir degerler
# Burada hedef kategorik gozukup kardinal olan kardinal gozukup kategorik olan degiskenleri ayirmak
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=5)
print(cat_cols)
print(num_cols)
print(cat_but_car)

"""Numerik Degiskenler Icin Outlier Kontolu Yapalim"""
# Quantile degerleri %90 %10 olarak belirlendi
for col_name in num_cols:
  low_limit, up_limit = outlier_thresholds(df, col_name)
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

fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
plt.show(block=True)

"""Kategorik ve Numerik Degiskenleri Inceleyelim"""

for num_col in num_cols:
  num_summary(df,num_col,plot=False)
new_cat_cols = [col for col in cat_cols if col not in ["SEX"]]
for cat_col in new_cat_cols:
  cat_summary(df,cat_col,plot=False)

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
print(df.groupby("SEX")["CHARGES"].mean())
print("\n")
# CHARGES and CHILDREN
print(df.groupby("CHILDREN")["CHARGES"].mean())
print("\n")

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
print(df.groupby("NEW_BMI_CAT")["CHARGES"].mean())
print("\n")

"""Simdi Degiskenleri Hazir Fonksiyonlarla Bir Daha Denetleyelim"""

# Degiskenleri tiplerine gore ayirmistik bir daha ayiralim
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=5)

"""Verileri One Hot Encoderdan Gecirelim"""

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("##################### HEAD #####################")
print(df.head())
print("##################### TAIL #####################")
print(df.tail())

"""Verileri Olceklendirelim"""

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("##################### HEAD #####################")
print(df.head())
print("##################### TAIL #####################")
print(df.tail())

"""Bagimli ve Bagimsiz Degiskenleri Ayiralim Train Test Ayrimini da Yapalim"""

y = df["CHARGES"]
X = df.drop("CHARGES",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

"""Model Olarak DECISION TREE'yi alalim Hiperparametre optimizasyonu yapalim MSE ve MAE degerlerine bakalim"""

regressor = DecisionTreeRegressor()

# Parametre grid'i oluşturma
param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(regressor, param_grid, cv=5)
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

# Parametre grid'i oluşturma
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.1, 0.2, 0.3]
}

""" Model Olarak DECISION TREE'yi alalim Hiperparametre optimizasyonu yapalim MSE ve MAE degerlerine bakalim """


# GridSearchCV ile hiperparametre optimizasyonu
# grid_search = GridSearchCV(regressor, param_grid, cv=5)
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

regressor = SVR(C = 10, epsilon = 0.1, gamma = 0.1, kernel = 'rbf')

# Modeli eğitme
regressor.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = regressor.predict(X_test)

# Test verisi için MAE hesaplaması
mae = mean_absolute_error(y_test, y_pred)
print("Test Verisi MAE:", mae)
mse = mean_squared_error(y_test, y_pred)
print("\nTest Verisi MSE:", mse)

"""Son Hali Icin SVR Yapildi (C = 10, epsilon = 0.1, gamma = 0.1, kernel = 'rbf') ve Bu Parametreler Kullanildi"""