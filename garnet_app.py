import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import joblib
import itertools
import warnings
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import io

warnings.filterwarnings('ignore')

# ------------------------------ 数据处理辅助函数 ------------------------------
def convert_normal(s):
    sub_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    return ''.join(sub_map.get(ch, ch) for ch in s)

def norm_formula(formula):
    if not isinstance(formula, str):
        return formula
    formula = convert_normal(formula)
    stack = []
    pairs = []
    for i, j in enumerate(formula):
        if j in '([{':
            stack.append(i)
        elif j in ')]}':
            if stack:
                left = stack.pop()
                pairs.append((left, i))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return formula
    content = [formula[i+1:j] for i, j in pairs]
    suffix = formula[pairs[-1][1]+1:]
    if len(content) >= 3:
        new_formula = f"{{{content[0]}}}[{content[1]}]({content[2]}){suffix}"
    elif len(content) == 2:
        new_formula = f"{{{content[0]}}}[{content[1]}]{suffix}"
    else:
        new_formula = f"{{{content[0]}}}{suffix}"
    return new_formula

def extract(formula):
    match = re.search(r'[\[{\(]([^\]\)}]*)[\]\)}]', str(formula))
    return match.group(1) if match else ''

def convert(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if ',' in s:
        s = s.split(',')[1].strip()
    match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s)
    if not match:
        return None
    num = float(match.group(1))
    if '%' in s:
        num = num / 100.0
    return num

# ------------------------------ 计算辅助函数 ------------------------------
def analyze_composition(comp):
    pattern = r'([A-Z][a-z]*)([0-9.₀₁₂₃₄₅₆₇₈₉]*)'
    match = re.findall(pattern, comp)
    result = {}
    for i, j in match:
        figure = 1.0 if j == '' else float(j)
        result[i] = figure
    return result

def extract_composition(formula):
    pattern = r'[\[{\(]([^\]}\)]+)[\]}\)]'
    match = re.findall(pattern, formula)
    return match[0], match[1], match[2]

def weighted_ave_std(value, weight):
    weight = np.array(weight, dtype=float)
    value = np.array(value, dtype=float)
    total = weight.sum()
    average = np.sum(weight * value) / total
    variance = np.sum(weight * (value - average) ** 2) / total
    std = np.sqrt(variance)
    return average, std

def get_property(elem, site, prop, prop_dict):
    key = (elem, site)
    if key not in prop_dict:
        raise KeyError(f"属性缺失:{elem}在{site}位")
    return prop_dict[key][prop]

def compute_site_prop(comp_dict, site, prop, prop_dict):
    value = []
    weight = []
    for elem, j in comp_dict.items():
        val = get_property(elem, site, prop, prop_dict)
        value.append(val)
        weight.append(j)
    return value, weight

# ------------------------------ 评估辅助函数 ------------------------------
def evaluate_model(X, y, outer_cv, base_model, param_dist, use_scaler=True):
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []
    best_params_list = []
    fold = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        steps = []
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', base_model))
        pipeline = Pipeline(steps)
        random_search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=20, cv=3,
            scoring='r2', random_state=42, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        best_params_list.append(random_search.best_params_)
        y_pred = best_model.predict(X_test)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-6))) * 100
        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape_list.append(mape)
    return np.array(r2_list), np.array(mae_list), np.array(rmse_list), np.array(mape_list), best_params_list

def evaluate_catboost(X, y, outer_cv, param_dist):
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []
    best_params_list = []
    fold = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        cat = CatBoostRegressor(random_seed=42, verbose=0)
        random_search = RandomizedSearchCV(
            cat, param_dist, n_iter=60, cv=3,
            scoring='r2', random_state=42, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        best_params_list.append(random_search.best_params_)
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
    return np.array(r2_list), np.array(mae_list), np.array(rmse_list), np.array(mape_list), best_params_list

# ------------------------------ 预测模块辅助函数（已修正）------------------------------
def calc_average_radius(comp_dict, site, prop_dict):
    total_coeff = sum(comp_dict.values())
    if total_coeff == 0:
        return np.nan
    weighted_sum = 0.0
    for elem, coeff in comp_dict.items():
        key = (elem, site)
        radius = prop_dict[key]['IonicRadius']
        weighted_sum += coeff * radius
    return weighted_sum / total_coeff

def tolerance_factor(r_A, r_B, R_O=1.38):
    return (r_A + R_O) / (np.sqrt(2) * (r_B + R_O))

def generate_composition(element, target_sum, step=0.5, max_element=2):
    n_step = int(target_sum / step)
    composition = []
    for size in range(1, max_element + 1):
        for subset in itertools.combinations(element, size):
            remain = n_step - size
            if remain < 0:
                continue
            for x in itertools.combinations_with_replacement(range(remain + 1), size):
                if sum(x) == remain:
                    coeff = [(xi + 1) * step for xi in x]
                    comp = {elem: coeff for elem, coeff in zip(subset, coeff)}
                    composition.append(comp)
    return composition

def apply_ce(A_comp, ce_conc, priority):          # 已增加 priority 参数
    for elem in priority:
        if elem in A_comp and A_comp[elem] >= ce_conc:
            A_new = A_comp.copy()
            A_new[elem] -= ce_conc
            if abs(A_new[elem]) < 1e-6:
                del A_new[elem]
            A_new['Ce'] = ce_conc
            return A_new
    return None

def check_charge_balance(A_comp, B_comp, C_comp, A_valence, B_valence, C_valence):
    total = 0.0
    for elem, coeff in A_comp.items():
        total += coeff * A_valence[elem]
    for elem, coeff in B_comp.items():
        total += coeff * B_valence[elem]
    for elem, coeff in C_comp.items():
        total += coeff * C_valence[elem]
    return abs(total - 24) < 1e-2

def compute_feature(A_comp, B_comp, C_comp, dc, prop_dict):   # 已增加 prop_dict 参数
    feat = {'DC': dc}
    val, wt = compute_site_prop(B_comp, 'B', 'ValenceElectronConcentration', prop_dict)
    if val:
        feat['B_ValenceElectronConcentration_Avg'], _ = weighted_ave_std(val, wt)
    else:
        return None
    val, wt = compute_site_prop(A_comp, 'A', 'Electronegativity', prop_dict)
    if val:
        feat['A_Electronegativity_Avg'], _ = weighted_ave_std(val, wt)
    else:
        return None
    val, wt = compute_site_prop(A_comp, 'A', 'FirstIonizationEnergy', prop_dict)
    if val:
        feat['A_FirstIonizationEnergy_Max'] = max(val)
    else:
        return None
    val, wt = compute_site_prop(A_comp, 'A', 'AtomicVolume', prop_dict)
    if val:
        feat['A_AtomicVolume_Avg'], _ = weighted_ave_std(val, wt)
    else:
        return None
    val, wt = compute_site_prop(C_comp, 'C', 'IonicRadius', prop_dict)
    if val:
        feat['C_IonicRadius_Avg'], _ = weighted_ave_std(val, wt)
    else:
        return None
    val, wt = compute_site_prop(B_comp, 'B', 'ThirdIonizationEnergy', prop_dict)
    if val:
        feat['B_ThirdIonizationEnergy_Max'] = max(val)
    else:
        return None
    val, wt = compute_site_prop(C_comp, 'C', 'SecondIonizationEnergy', prop_dict)
    if val:
        feat['C_SecondIonizationEnergy_Avg'], _ = weighted_ave_std(val, wt)
    else:
        return None
    val, wt = compute_site_prop(C_comp, 'C', 'Polarizability', prop_dict)
    if val:
        feat['C_Polarizability_Max'] = max(val)
    else:
        return None
    return feat

# ------------------------------ 缓存数据加载 ------------------------------
@st.cache_data
def load_element_properties(file):
    elem_df = pd.read_csv(file)
    prop_dict = {}
    for _, row in elem_df.iterrows():
        elem = row['Element'].strip()
        site = row['Site'].strip()
        prop_dict[(elem, site)] = {
            'IonicRadius': row['IonicRadius(Å)'],
            'FirstIonizationEnergy': row['First ionization energy(eV)'],
            'SecondIonizationEnergy': row['Second ionization energy(eV)'],
            'ThirdIonizationEnergy': row['ThirdIonizationEnergy(eV)'],
            'Electronegativity': row['Electronegativity'],
            'ValenceElectronConcentration': row['Valence electron concentration'],
            'Polarizability': row['Polarizability(Å³)'],
            'MeltingTemperature': row['Melting temperature(K)'],
            'AtomicVolume': row['Atomic volume(cm³/mol)'],
            'PeriodicNumber': row['Periodic number'],
            'MendeleevNumber': row['Mendeleev number']
        }
    return prop_dict

# ------------------------------ 流式布局界面 ------------------------------
st.set_page_config(page_title="石榴石荧光材料发射波长预测系统", layout="wide")
st.title("机器学习驱动的紫光激发石榴石型荧光材料发射波长预测系统")
st.markdown("上传实验数据与元素属性表，完成数据预处理、模型训练/载入及虚拟化合物高通量筛选。")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 操作面板")
    st.subheader("1. 数据文件")
    data_file = st.file_uploader("上传石榴石文献数据 (Excel)", type=["xlsx"])
    elem_file = st.file_uploader("上传元素属性表 (CSV)", type=["csv"])
    if elem_file is not None:
        prop_dict = load_element_properties(elem_file)
    else:
        prop_dict = None

    st.subheader("2. 模型选项")
    use_existing_model = st.checkbox("加载已有训练模型", value=False)
    model_file = None
    if use_existing_model:
        model_file = st.file_uploader("上传集成模型 (pkl)", type=["pkl"])
        features_file = st.file_uploader("上传特征列表 (pkl)", type=["pkl"])

    st.subheader("3. 虚拟筛选参数")
    ce_concs = st.text_input("Ce掺杂浓度（逗号分隔）", value="0.02,0.04,0.06")
    wavelength_min = st.number_input("发射波长下限 (nm)", value=480)
    wavelength_max = st.number_input("发射波长上限 (nm)", value=510)

    run_btn = st.button("▶️ 开始运行", type="primary")

# 主界面
if run_btn and data_file is not None and elem_file is not None:
    df_raw = pd.read_excel(data_file, sheet_name='石榴石文献数据')

    # ---------------------- 数据预处理 ----------------------
    with st.expander("📂 原始数据", expanded=False):
        raw_display = df_raw.copy()
        raw_display.index = range(1, len(raw_display) + 1)
        st.dataframe(raw_display, height=400, use_container_width=True)

    df = df_raw.iloc[66:].reset_index(drop=True)
    old_cols = ['组成', '发射波长（峰值）', '掺杂元素及比例', '掺杂元素占据位点', '合成方法']
    new_cols = [col for col in old_cols if col in df.columns]
    df = df[new_cols]
    rename_dict = {}
    if '发射波长（峰值）' in df.columns:
        rename_dict['发射波长（峰值）'] = '发射波长'
    if '掺杂元素及比例' in df.columns:
        rename_dict['掺杂元素及比例'] = '掺杂浓度'
    df.rename(columns=rename_dict, inplace=True)
    m = df['合成方法'].str.contains('高温', na=False) & df['合成方法'].str.contains('固相', na=False)
    df = df[m].copy()
    if '组成' in df.columns:
        li = df['组成'].apply(lambda x: 'Al' in extract(str(x)))
        df = df[~li]
    if '组成' in df.columns:
        df['new_formula'] = df['组成'].apply(norm_formula)
        df = df.drop_duplicates(subset=['new_formula', '发射波长'], keep='first')
        df['组成'] = df['new_formula']
        df = df.drop(columns=['new_formula'])
    if '掺杂浓度' in df.columns:
        df['掺杂浓度'] = df['掺杂浓度'].apply(convert)

    with st.expander("📂 清洗后数据（共{}条）".format(len(df)), expanded=True):
        clean_display = df.copy()
        clean_display.index = range(1, len(clean_display) + 1)
        st.dataframe(clean_display, height=400, use_container_width=True)

    # ---------------------- 特征工程 ----------------------
    with st.spinner("正在计算特征矩阵..."):
        if prop_dict is None:
            st.error("请上传元素属性表！")
            st.stop()
        R_O = 1.38
        comp_df = df.copy()
        comp_df['波长'] = comp_df['发射波长'].str.extract(r'(\d+)').astype(float)
        all_feature = []
        for index, row in comp_df.iterrows():
            formula = row['组成']
            wavelength = row['波长']
            dc = row['掺杂浓度']
            site_info = row['掺杂元素占据位点']
            try:
                A, B, C = extract_composition(formula)
                A_comp = analyze_composition(A)
                B_comp = analyze_composition(B)
                C_comp = analyze_composition(C)
            except:
                continue
            feat = {'化合物': formula, '波长': wavelength, 'DC': dc}
            prop_names = list(prop_dict[list(prop_dict.keys())[0]].keys())
            for site in ['A', 'B', 'C']:
                comp = {'A': A_comp, 'B': B_comp, 'C': C_comp}[site]
                for prop in prop_names:
                    value, weight = compute_site_prop(comp, site, prop, prop_dict)
                    if value:
                        min_val, max_val = min(value), max(value)
                        avg_val, std_val = weighted_ave_std(value, weight)
                    else:
                        min_val = max_val = avg_val = std_val = np.nan
                    feat[f'{site}_{prop}_Min'] = min_val
                    feat[f'{site}_{prop}_Max'] = max_val
                    feat[f'{site}_{prop}_Avg'] = avg_val
                    feat[f'{site}_{prop}_Std'] = std_val
            # 自定义特征
            A_IR_val, A_IR_weight = compute_site_prop(A_comp, 'A', 'IonicRadius', prop_dict)
            avg_IR_A, std_IR_A = weighted_ave_std(A_IR_val, A_IR_weight)
            B_IR_val, B_IR_weight = compute_site_prop(B_comp, 'B', 'IonicRadius', prop_dict)
            avg_IR_B, _ = weighted_ave_std(B_IR_val, B_IR_weight)
            BC_val, BC_weight = [], []
            for elem, j in B_comp.items():
                BC_val.append(get_property(elem, 'B', 'IonicRadius', prop_dict))
                BC_weight.append(j)
            for elem, j in C_comp.items():
                BC_val.append(get_property(elem, 'C', 'IonicRadius', prop_dict))
                BC_weight.append(j)
            avg_IR_BC, std_IR_BC = weighted_ave_std(BC_val, BC_weight)
            feat['Dif_RABC'] = avg_IR_A - avg_IR_BC if not np.isnan(avg_IR_A) and not np.isnan(avg_IR_BC) else np.nan
            feat['Std_RABC'] = std_IR_A - std_IR_BC if not np.isnan(std_IR_A) and not np.isnan(std_IR_BC) else np.nan
            t = (avg_IR_A + R_O) / (np.sqrt(2) * (avg_IR_B + R_O)) if not np.isnan(avg_IR_A) and not np.isnan(avg_IR_B) else np.nan
            feat['ToleranceFactor'] = t
            match = re.search(r'取代([A-Z][a-z]*)', site_info)
            if match:
                replace_elem = match.group(1)
                try:
                    Rm = get_property(replace_elem, 'A', 'IonicRadius', prop_dict)
                    Rce = get_property('Ce', 'A', 'IonicRadius', prop_dict)
                    feat['IRM'] = abs(Rm - Rce) / Rm
                except:
                    feat['IRM'] = np.nan
            else:
                feat['IRM'] = np.nan
            all_feature.append(feat)

        feature_df = pd.DataFrame(all_feature)
        other_col = [c for c in feature_df.columns if c not in ['化合物', '波长', 'DC']]
        order_col = ['化合物', '波长', 'DC'] + other_col
        feature_df = feature_df[order_col]
        st.success("特征计算完成！")

        with st.expander("📂 特征矩阵（共{}行特征）".format(len(feature_df)), expanded=False):
            feat_display = feature_df.copy()
            feat_display.index = range(1, len(feat_display) + 1)
            st.dataframe(feat_display, height=400, use_container_width=True)

        csv = feature_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载特征矩阵 (feature_matrix.csv)", csv, "feature_matrix.csv")

    # ---------------------- 模型训练 / 加载 ----------------------
    if use_existing_model and model_file is not None and features_file is not None:
        with st.spinner("加载已有模型..."):
            models = joblib.load(model_file)
            feature_cols = joblib.load(features_file)
            st.session_state['models'] = models
            st.session_state['feature_cols'] = feature_cols
            st.success("模型加载成功！")
    else:
        with st.spinner("正在训练模型（可能需要几分钟）..."):
            target_col = '波长'
            df_model = feature_df.copy()
            df_model[target_col] = df_model[target_col].astype(str).str.extract(r'(\d+(?:\.\d+)?)').astype(float)
            feature_cols = [
                'B_ValenceElectronConcentration_Avg',
                'A_Electronegativity_Avg',
                'A_FirstIonizationEnergy_Max',
                'A_AtomicVolume_Avg',
                'C_IonicRadius_Avg',
                'B_ThirdIonizationEnergy_Max',
                'C_SecondIonizationEnergy_Avg',
                'C_Polarizability_Max',
                'DC'
            ]
            missing = [f for f in feature_cols if f not in df_model.columns]
            if missing:
                st.error(f"缺失特征: {missing}")
                st.stop()
            df_model[feature_cols] = df_model[feature_cols].apply(pd.to_numeric, errors='coerce')
            X = df_model[feature_cols].values
            y = df_model[target_col].values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=42)
            param_dist = {
                'iterations': [50, 100, 200, 300, 400],
                'depth': [1, 4, 6, 8],
                'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'colsample_bylevel': [0.6, 0.7, 0.8, 0.9]
            }
            r2, mae, rmse, mape, best_params_list = evaluate_catboost(X, y, outer_cv, param_dist)
            st.write("**CatBoost 交叉验证性能**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2.mean():.3f} ± {r2.std():.3f}")
            col2.metric("MAE (nm)", f"{mae.mean():.2f}")
            col3.metric("RMSE (nm)", f"{rmse.mean():.2f}")
            col4.metric("MAPE (%)", f"{mape.mean():.1f}")
            param_counter = Counter([tuple(sorted(p.items())) for p in best_params_list])
            most_common_params = dict(param_counter.most_common(1)[0][0])
            st.write("**最优超参数**", most_common_params)
            n_bootstrap = 100
            rng = np.random.RandomState(42)
            models = []
            for b in range(n_bootstrap):
                indices = rng.choice(len(X), size=len(X), replace=True)
                X_boot, y_boot = X[indices], y[indices]
                model = CatBoostRegressor(**most_common_params, random_seed=b, verbose=0)
                model.fit(X_boot, y_boot)
                models.append(model)
            st.success("集成模型训练完毕（100个CatBoost模型）")
            st.session_state['models'] = models
            st.session_state['feature_cols'] = feature_cols

    # ---------------------- 虚拟化合物生成与预测 ----------------------
    if 'models' in st.session_state and 'feature_cols' in st.session_state:
        models = st.session_state['models']
        feature_cols = st.session_state['feature_cols']
    else:
        st.warning("请先训练模型或加载已有模型！")
        st.stop()

    ce_concentration = [float(x.strip()) for x in ce_concs.split(',')]
    A_element = ['Ba', 'Ca', 'Gd', 'La', 'Lu', 'Mg', 'Sr', 'Y']
    A_valence = {'Ba':2, 'Ca':2, 'Gd':3, 'La':3, 'Lu':3, 'Mg':2, 'Sr':2, 'Y':3, 'Ce':3}
    B_element = ['Y', 'Al', 'Ga', 'Sc', 'Mg', 'Zr', 'Hf']
    B_valence = {'Y':3, 'Al':3, 'Ga':3, 'Sc':3, 'Mg':2, 'Zr':4, 'Hf':4}
    C_element = ['Al', 'Ga', 'Si', 'Ge']
    C_valence = {'Al':3, 'Ga':3, 'Si':4, 'Ge':4}
    full_priority = ['La', 'Ca', 'Gd', 'Sr', 'Y', 'Lu', 'Mg', 'Ba']
    priority = [elem for elem in full_priority if elem in A_element]

    with st.spinner("生成虚拟化合物并预测..."):
        A_composition = generate_composition(A_element, 3.0, step=0.5, max_element=2)
        B_composition = generate_composition(B_element, 2.0, step=0.5, max_element=2)
        C_composition = generate_composition(C_element, 3.0, step=0.5, max_element=2)
        virtual_compound = []
        success = 0
        for A_base in A_composition:
            for B_comp in B_composition:
                for C_comp in C_composition:
                    for ce_conc in ce_concentration:
                        A_comp = apply_ce(A_base, ce_conc, priority)
                        if A_comp is None: continue
                        if not check_charge_balance(A_comp, B_comp, C_comp, A_valence, B_valence, C_valence): continue
                        r_A = calc_average_radius(A_comp, 'A', prop_dict)
                        r_B = calc_average_radius(B_comp, 'B', prop_dict)
                        if np.isnan(r_A) or np.isnan(r_B): continue
                        t = tolerance_factor(r_A, r_B)
                        if t <= 0.85 or t >= 1.05: continue
                        feat = compute_feature(A_comp, B_comp, C_comp, ce_conc, prop_dict)
                        if feat is None: continue
                        A_str = ''.join(f"{elem}{coeff:.4g}" for elem, coeff in sorted(A_comp.items()))
                        B_str = ''.join(f"{elem}{coeff:.4g}" for elem, coeff in sorted(B_comp.items()))
                        C_str = ''.join(f"{elem}{coeff:.4g}" for elem, coeff in sorted(C_comp.items()))
                        formula = f"{{{A_str}}}[{B_str}]({C_str})O12"
                        virtual_compound.append({'化合物': formula, **feat, '容忍因子': t})
                        success += 1
        if success == 0:
            st.error("未生成有效虚拟化合物，请检查参数！")
            st.stop()
        virtual_df = pd.DataFrame(virtual_compound)
        virtual_df = virtual_df[['化合物'] + feature_cols + ['容忍因子']]
        X_virtual = virtual_df[feature_cols].values
        pred = np.zeros((len(X_virtual), len(models)))
        for i, model in enumerate(models):
            pred[:, i] = model.predict(X_virtual)
        ave_pred = pred.mean(axis=1)
        std_pred = pred.std(axis=1)
        virtual_df['预测波长均值'] = ave_pred
        virtual_df['预测波长标准差'] = std_pred
        mask = (ave_pred >= wavelength_min) & (ave_pred <= wavelength_max)
        select = virtual_df[mask].copy()
        st.success(f"虚拟化合物生成完成！共 {success} 个有效化合物，筛选出 {len(select)} 个目标波长 ({wavelength_min}-{wavelength_max} nm) 的候选。")
        with st.expander("📂 筛选结果（共{}个候选化合物）".format(len(select)), expanded=True):
            select_display = select.copy()
            select_display.index = range(1, len(select_display) + 1)
            st.dataframe(select_display, height=400, use_container_width=True)
        csv_select = select.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载筛选结果 (virtual_compounds_select.csv)", csv_select, "virtual_compounds_select.csv")
