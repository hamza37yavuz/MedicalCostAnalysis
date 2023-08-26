import joblib
import config as cnf
import MedicalCostAnalysis

svr_model = joblib.load(cnf.pkl)

# Burasi test icin yeni bir dataframe yuklenirse kullanilacak sekilde ayarlandi

# charges sutunu olmayan dataframe okunacak
# sonra aşağıdaki fonksiyonunicine verilerek calistirilacak
# df = MedicalCostAnalysis.data_prep(dataframe)

# Sonra aşagidaki kod calistirilacak
# y_pred = grid_search.predict(df)

# y_pred bizim charges degerlerimizi verecektir
