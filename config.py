# Bu dosya bize verilen dosya yolu ve parametrelerin barındığı dosyadir
# Ana dosyaninin ve diger dosyalarin calismasi icin gereklidir

medcost = "insurance.csv"

pkl = "med_cost_analysis_svr.pkl"

param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.1, 0.2, 0.3]
}

param_grid_decision_tree = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}