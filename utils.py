# Bu dosya bu proje icin yazilmis olan fonksiyonlari bulundurur.
# Her fonksiyon icin docstring yazilmistir


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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