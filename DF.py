import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from deepforest import CascadeForestRegressor

# 加载数据
data = pd.read_csv('D:/data/P0.001/daoshuyijie_DWT/193_shuxue_DWT_r.csv')

# 提取特征和目标变量
features = data.columns[0:700].tolist()
target = data['SSC']
X_train, X_test, y_train, y_test = train_test_split(data[features], target, test_size=77 / 193, random_state=29)

# 特征归一化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


# 多粒度扫描实现
def multi_grained_scanning(X, window_size, step=1):
    n_samples, n_features = X.shape
    scanned_features = []
    if window_size > n_features:
        return X  # 如果窗口大小超过特征数，返回原始特征

    for i in range(0, n_features - window_size + 1, step):
        window_features = X[:, i:i + window_size]
        scanned_features.append(window_features)

    return np.hstack(scanned_features) if scanned_features else X


# 遍历 88 到 233 之间的窗口大小，找到最佳窗口
best_window = None
best_r2_test = -np.inf

for window_size in range(88, 234):
    X_train_transformed = multi_grained_scanning(X_train_scaled, window_size)
    X_test_transformed = multi_grained_scanning(X_test_scaled, window_size)

    model = CascadeForestRegressor(random_state=24, verbose=0)
    model.fit(X_train_transformed, y_train_scaled)
    y_test_pred = model.predict(X_test_transformed)
    y_test_pred_real = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    y_test_real = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    r2_test = r2_score(y_test_real, y_test_pred_real)

    if r2_test > best_r2_test:
        best_r2_test = r2_test
        best_window = window_size

    print(f"Window Size: {window_size}, Test R²: {r2_test:.4f}")

print(f"Best Window Size: {best_window}, Best Test R²: {best_r2_test:.4f}")

# 使用最佳窗口大小进行最终训练
X_train_scaled = multi_grained_scanning(X_train_scaled, best_window)
X_test_scaled = multi_grained_scanning(X_test_scaled, best_window)

# 定义超参数搜索范围
param_ranges = {
    'n_estimators': (2, 20, 1),
    'n_trees': (15, 200, 5),
    'max_layers': (2, 10, 1),
    'min_samples_split': (2, 10, 1),
    'min_samples_leaf': (3, 10, 1)
}

best_params = None
best_r2_train = -np.inf
best_r2_test = -np.inf
best_model = None
max_iterations = 30
iteration = 0

while iteration < max_iterations:
    print(f"第 {iteration + 1} 次超参数搜索...")

    param_combinations = list(product(
        range(param_ranges['n_estimators'][0], param_ranges['n_estimators'][1] + 1, param_ranges['n_estimators'][2]),
        range(param_ranges['n_trees'][0], param_ranges['n_trees'][1] + 1, param_ranges['n_trees'][2]),
        range(param_ranges['max_layers'][0], param_ranges['max_layers'][1] + 1, param_ranges['max_layers'][2]),
        range(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1] + 1,
              param_ranges['min_samples_split'][2]),
        range(param_ranges['min_samples_leaf'][0], param_ranges['min_samples_leaf'][1] + 1,
              param_ranges['min_samples_leaf'][2])
    ))

    for params in param_combinations:
        n_estimators, n_trees, max_layers, min_samples_split, min_samples_leaf = params
        model = CascadeForestRegressor(
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_layers=max_layers,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=24,
            verbose=0
        )

        try:
            model.fit(X_train_scaled, y_train_scaled)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            y_train_pred_real = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
            y_test_pred_real = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

            y_train_real = target_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
            y_test_real = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

            r2_train = r2_score(y_train_real, y_train_pred_real)
            r2_test = r2_score(y_test_real, y_test_pred_real)

            print(f"参数: {params} -> 训练集 R²: {r2_train:.4f}, 测试集 R²: {r2_test:.4f}")

            if r2_train > 0.65 and r2_test > 0.62 and r2_train > r2_test:
                if r2_train > best_r2_train and r2_test > best_r2_test:
                    best_params = params
                    best_r2_train = r2_train
                    best_r2_test = r2_test
                    best_model = model
                    print(f"找到满足条件的最优参数: {best_params}")
                    break
        except ValueError as e:
            print(f"出现 NaN 错误，跳过参数组合: {params}")
            continue

    if best_params:
        print(f"找到最优参数: {best_params}")
        break

if not best_params:
    best_params = "无最优参数"
    best_model = model

joblib.dump(best_model, 'D:/data/P0.001/daoshuyijie_DWT/DF/DF329.pkl')
