# optimizer.py
"""
Модуль для выполнения оптимизации подбора заказов (MIP).
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from collections import defaultdict
from typing import Dict, Any, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mosek")

DEFAULT_OPT_PARAMS = {
    "max_blanks_per_order": None,
    "min_volume_threshold": None,
    "penalty_blanks":       1e0,
    "penalty_sets":         1e1,
    "slack_penalty":        1e5,
    "penalty_rkm":          1e3,
    "solver":               cp.MOSEK,
    "solver_verbose":       True,
    "mosek_params":         {}
}

def run_optimization(
    processed_df_input: pd.DataFrame,
    order_df_for_dict_input: pd.DataFrame,
    opt_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, float, Dict, Dict]:
    """
    Выполняет MIP-оптимизацию распределения заготовок по заказам.
    """
    print("Начало этапа MIP-оптимизации...")
    params = {**DEFAULT_OPT_PARAMS, **opt_params}

    # --- Подготовка данных внутри функции, как в ноутбуке/твоем последнем скрипте ---
    merged_df = processed_df_input.copy() # Используем копии
    order_for_dict = order_df_for_dict_input.copy()

    # Колонка 'key' теперь должна приходить из processed_df_input (data_preprocessor)
    if 'key' not in merged_df.columns:
        # Создаем, если вдруг не пришла (защита)
        print("Предупреждение (optimizer): колонка 'key' отсутствует в processed_df_input, создаю заново.")
        merged_df['key'] = list(zip(
            merged_df['Типоразмер трубы'], merged_df['Вн. марка/Марка'],
            merged_df['Номер заказа'], merged_df['Позиция заказа']
        ))
    
    # Аналогично для order_for_dict, если это еще не было сделано в preprocessor для order_df_prepared
    if 'key' not in order_for_dict.columns:
         order_for_dict['key'] = list(zip(
            order_for_dict['Типоразмер трубы'], order_for_dict['Вн. марка/Марка'],
            order_for_dict['Номер заказа'], order_for_dict['Позиция заказа']
        ))
    groups = merged_df.groupby('key')
    order_dict = order_for_dict.groupby('key')['Неотгруженное кол-во'].sum().to_dict()
    
    # pipe_keys должен включать все ключи из order_dict, чтобы для всех был создан s (недобор)
    pipe_keys_from_groups = list(groups.groups.keys())
    all_possible_pipe_keys = list(set(pipe_keys_from_groups + list(order_dict.keys())))
    pipe_keys = sorted(all_possible_pipe_keys)

    # Сбор характеристик (только для ключей, присутствующих в groups)
    rkm = {k: np.ascontiguousarray(g['РКМ'].values.astype(np.float64))
           for k, g in groups}
    cost_price = {k: np.ascontiguousarray(g['Себестоимость (уникальная)'].values.astype(np.float64))
                  for k, g in groups}
    wrh_volume = {k: np.ascontiguousarray(g['ОстКонПериода'].values.astype(np.float64))
                  for k, g in groups}
    sizes = {k: np.ascontiguousarray(g['Длина заготовки'].values.astype(np.float64))
             for k, g in groups}
    plavka = {k: g['№ плавки1'].tolist() for k, g in groups}
    sizes_diam = {k: np.ascontiguousarray(g['Диаметр заготовки'].values.astype(np.float64))
                  for k, g in groups}

    n_blanks = {key: len(wrh_volume[key]) for key in pipe_keys_from_groups} # n_blanks только для ключей с заготовками

    sets_per_key = {}
    for key_spk in pipe_keys_from_groups:
        # Убедимся, что все данные для sets_per_key существуют для данного ключа
        if key_spk in sizes and key_spk in sizes_diam and \
           n_blanks.get(key_spk, 0) > 0 and \
           len(sizes[key_spk]) == n_blanks[key_spk] and \
           len(sizes_diam[key_spk]) == n_blanks[key_spk]:
            sets_per_key[key_spk] = defaultdict(list)
            for j_spk in range(n_blanks[key_spk]):
                char_set_spk = (sizes[key_spk][j_spk], sizes_diam[key_spk][j_spk])
                sets_per_key[key_spk][char_set_spk].append(j_spk)
        # else: # Можно добавить вывод, если данные для sets_per_key неполные
            # print(f"Предупреждение (sets_per_key): Неполные данные для ключа {key_spk} при формировании sets_per_key.")

    # blank_ids используется в оригинальном коде ноутбука, но это просто plavka
    # blank_ids_dict = plavka 
    blank_to_keys = defaultdict(list)
    for key_btk in pipe_keys_from_groups: # Итерация только по ключам с заготовками
        if key_btk in plavka: # plavka уже отфильтрована по pipe_keys_from_groups
            for bid_item in plavka[key_btk]:
                blank_to_keys[bid_item].append(key_btk)
    # --- Конец подготовки данных ---

    # --- Проверки данных из ноутбука (полезно оставить) ---
    print("Проверка данных для оптимизатора...")
    for key_check in pipe_keys_from_groups: # Проверяем только ключи с заготовками
        # Проверка длины списков (уже частично сделана выше при создании словарей)
        assert len(rkm[key_check]) == n_blanks[key_check], f"rkm[{key_check}] неверной длины"
        assert len(cost_price[key_check]) == n_blanks[key_check], f"cost_price[{key_check}] неверной длины"
        assert len(wrh_volume[key_check]) == n_blanks[key_check], f"wrh_volume[{key_check}] неверной длины"
        assert len(sizes[key_check]) == n_blanks[key_check], f"sizes[{key_check}] неверной длины"
        assert len(plavka[key_check]) == n_blanks[key_check], f"plavka[{key_check}] неверной длины"
        assert len(sizes_diam[key_check]) == n_blanks[key_check], f"sizes_diam[{key_check}] неверной длины"


        if key_check in sets_per_key:
            for char_set_check, indices_check in sets_per_key[key_check].items():
                assert all(i >= 0 and i < n_blanks[key_check] for i in indices_check), \
                    f"Неверные индексы в sets_per_key для {key_check}, {char_set_check}: {indices_check}"
    print("Проверка данных завершена.")
    # --- Конец проверок ---

    # -------------------------
    # Определение переменных оптимизации (MIP)
    # -------------------------
    print("Создание переменных MIP...")
    x = {
        key: cp.Variable(shape=n_blanks[key], nonneg=True, name=f"x_{key}_mip")
        for key in pipe_keys_from_groups
    }
    s = {key: cp.Variable(nonneg=True, name=f"s_{key}_mip") for key in pipe_keys}
    z = {
        key: cp.Variable(shape=n_blanks[key], boolean=True, name=f"z_{key}_mip")
        for key in pipe_keys_from_groups
    }
    w = {
        (key, char_set): cp.Variable(boolean=True, name=f"w_{key}_{char_set}_mip")
        for key in pipe_keys_from_groups if key in sets_per_key
        for char_set in sets_per_key.get(key, {})
    }
    v = {}
    for bid_v, keys_using_bid_v in blank_to_keys.items():
        for key_item_v in keys_using_bid_v:
            # Переменная v создается только если для key_item_v есть заготовки
            if key_item_v in pipe_keys_from_groups:
                 v[bid_v, key_item_v] = cp.Variable(boolean=True, name=f"v_{bid_v}_{key_item_v}_mip")
    
    # Переменная y (закупки) - если не используется, ее можно удалить
    # y = {key: cp.Variable(nonneg=True, name=f"y_{key}_mip") for key in pipe_keys}

    # -------------------------
    # Формулировка ограничений (MIP)
    # -------------------------
    print("Формирование ограничений MIP...")
    constraints = []

    # 1. Каждая плавка (`bid` = `№ плавки1`) используется только для одного ключа заказа.
    for bid, keys_for_bid in blank_to_keys.items():
        v_sum_terms = [
            v_var for key_item in keys_for_bid
            if (v_var := v.get((bid, key_item))) is not None
        ]
        if v_sum_terms:
            constraints.append(cp.sum(v_sum_terms) <= 1)

    # 2. Связи x, z, v
    for key_tuple in pipe_keys_from_groups:
        current_x = x[key_tuple]
        current_z = z[key_tuple]
        current_wrh_volume_np = np.array(wrh_volume[key_tuple])

        # Ограничение 5 из ноутбука: x_j <= W_j * z_j
        constraints.append(current_x <= cp.multiply(current_wrh_volume_np, current_z))

        # Связь z_j с v_bid,key:
        for j in range(n_blanks[key_tuple]):
            bid_j = plavka[key_tuple][j]
            v_var_for_z = v.get((bid_j, key_tuple))
            if v_var_for_z is not None:
                constraints.append(current_z[j] <= v_var_for_z)
            else:
                constraints.append(current_z[j] == 0)

    # 3. Выполнение заказа
    rkm_mean = {
        key_rm: np.mean(rkm_val_list)
        if (rkm_val_list := rkm.get(key_rm)) is not None and len(rkm_val_list) > 0 and np.sum(np.isfinite(rkm_val_list)) > 0
        else 1.0
        for key_rm in pipe_keys
    }

    for key_tuple in pipe_keys:
        effective_x_sum = 0
        if key_tuple in pipe_keys_from_groups: # Если для этого заказа есть заготовки/переменные x
            current_rkm_values = np.array(rkm.get(key_tuple, []))
            if key_tuple in n_blanks and len(current_rkm_values) == n_blanks[key_tuple]:
                default_rkm_for_key = rkm_mean.get(key_tuple,1.0) # Уже безопасное значение
                safe_rkm_values = np.where((current_rkm_values <= 0) | ~np.isfinite(current_rkm_values),
                                           default_rkm_for_key, current_rkm_values)
                safe_rkm_values = np.where(safe_rkm_values == 0, 1e-9, safe_rkm_values)
                if x[key_tuple].size == len(safe_rkm_values):
                     effective_x_sum = cp.sum(cp.multiply(x[key_tuple], 1.0 / safe_rkm_values))
                # else: обработка несоответствия длин (если нужна, но n_blanks должна совпадать)
            elif key_tuple in n_blanks and n_blanks[key_tuple] > 0 : # РКМ некорректен, но x есть
                safe_rkm_mean_val = rkm_mean.get(key_tuple,1.0)
                effective_x_sum = cp.sum(x[key_tuple]) / safe_rkm_mean_val
        
        # Закомментированная переменная y из ноутбука
        # effective_y = y[key_tuple] / rkm_mean[key_tuple] 
        constraints.append(effective_x_sum + s[key_tuple] == order_dict[key_tuple])

    # 4. Ограничение: нельзя использовать больше, чем имеется на складе (x_j <= W_j).
    # В ноутбуке это было, но оно неявно покрывается x <= W*z и z - бинарная.
    # for key_c4 in pipe_keys_from_groups:
    #    constraints.append(x[key_c4] <= wrh_volume[key_c4])

    # 6. Связь z и w
    for key_c6 in pipe_keys_from_groups:
        if key_c6 in sets_per_key and key_c6 in z:
            current_z_c6 = z[key_c6]
            for char_set_c6, indices_c6 in sets_per_key[key_c6].items():
                w_var_c6 = w.get((key_c6, char_set_c6))
                if w_var_c6 is not None and indices_c6:
                    constraints.append(current_z_c6[indices_c6] <= w_var_c6)
    
    # Опциональные ограничения из ноутбука не переносим, если не было явного указания
    # (max_blanks_per_order, min_volume_threshold)

    # -------------------------
    # Целевая функция (MIP)
    # -------------------------
    obj_terms = []
    if s and all(isinstance(val, cp.Variable) for val in s.values()):
        obj_terms.append(params["slack_penalty"] * cp.sum(list(s.values())))

    z_sum_expressions = [cp.sum(z_vars) for z_vars in z.values() if z_vars is not None and z_vars.size > 0]
    if z_sum_expressions:
        obj_terms.append(params["penalty_blanks"] * cp.sum(z_sum_expressions))

    sum_w_terms = [w_val for w_val in w.values() if w_val is not None] # w.values() содержит cp.Variable
    if sum_w_terms:
        obj_terms.append(params["penalty_sets"] * cp.sum(sum_w_terms))

    rkm_penalty_expressions = []
    for key_rkm_loop in pipe_keys_from_groups:
        if key_rkm_loop in rkm and rkm.get(key_rkm_loop) is not None and \
           key_rkm_loop in x and x[key_rkm_loop].size > 0 and \
           key_rkm_loop in n_blanks and len(rkm[key_rkm_loop]) == n_blanks[key_rkm_loop]:
            current_rkm_for_obj = np.array(rkm[key_rkm_loop])
            rkm_coeffs_for_obj = np.where((current_rkm_for_obj > 0) & np.isfinite(current_rkm_for_obj),
                                          current_rkm_for_obj, 0)
            rkm_penalty_expressions.append(cp.sum(cp.multiply(rkm_coeffs_for_obj, x[key_rkm_loop])))
    if rkm_penalty_expressions:
        obj_terms.append(params["penalty_rkm"] * cp.sum(rkm_penalty_expressions))

    if not obj_terms:
        objective = cp.Minimize(0)
        print("Предупреждение MIP: Целевая функция пуста.")
    else:
        objective = cp.Minimize(cp.sum(obj_terms))

    # -------------------------
    # Формирование и решение задачи (MIP)
    # -------------------------
    problem = cp.Problem(objective, constraints)
    print(f"Запуск решения MIP с решателем: {params['solver']}...")
    
    optimal_value = float('inf') # Значение по умолчанию
    try:
        solve_kwargs = {'solver': params['solver'], 'verbose': params['solver_verbose']}
        if params['solver'] == cp.MOSEK and params.get('mosek_params'):
            solve_kwargs['mosek_params'] = params['mosek_params']
        
        optimal_value = problem.solve(**solve_kwargs)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Внимание: MIP-решение не найдено или не оптимально. Статус: {problem.status}")
            optimal_value = float('inf') if optimal_value is None else optimal_value
            
    except cp.error.SolverError as e:
        print(f"Ошибка MIP-решателя: {e}")
        raise # Перевыбрасываем, чтобы main обработал
    except Exception as e_gen: # Ловим другие возможные ошибки при solve
        print(f"Непредвиденная ошибка при problem.solve(): {e_gen}")
        raise

    print(f"MIP-оптимизация завершена. Оптимальное значение: {optimal_value}")

    # ... (остальной код формирования результатов results_list_mip и df_results_raw_mip 
    #      аналогичен тому, что был в run_mip_optimization моего предыдущего ответа,
    #      используя переменные x, s, plavka, sizes, wrh_volume, sizes_diam) ...
    # Начало формирования результатов
    results_list_mip = []
    if problem.status in ["optimal", "optimal_inaccurate"]:
        for key_tuple_res in pipe_keys: # Итерируемся по всем ключам заказов
            pipe_type, steel_grade, num_zakaza, pos_zakaza = key_tuple_res
            
            x_val_arr_res = np.array([])
            # Получаем значения x, только если для этого ключа были заготовки и переменные x
            if key_tuple_res in pipe_keys_from_groups and key_tuple_res in x and x[key_tuple_res].value is not None:
                x_val_arr_res = x[key_tuple_res].value
            
            s_val_scalar_res = 0.0
            if key_tuple_res in s and s[key_tuple_res].value is not None:
                s_val_scalar_res = s[key_tuple_res].value

            if np.abs(s_val_scalar_res) < 1e-6:
                s_val_clean_res_scalar = 0.0
            else:
                s_val_clean_res_scalar = np.round(s_val_scalar_res, 3)
                
            x_val_clean_res = np.where(np.abs(x_val_arr_res) < 1e-6, 0, np.round(x_val_arr_res, 3))
            # s_val_clean_res из твоего кода был:
            # s_val_clean_res = np.where(np.abs(s_val_scalar_res) < 1e-6, 0, np.round(s_val_scalar_res, 3))
            # Теперь мы используем s_val_clean_res_scalar

            # Характеристики заготовок берем только если они есть для данного ключа
            current_lengths_res = sizes.get(key_tuple_res, [])
            current_volumes_res = wrh_volume.get(key_tuple_res, [])
            current_diams_res = sizes_diam.get(key_tuple_res, [])
            current_plavki_res = plavka.get(key_tuple_res, []) # plavka - это словарь plavka_ids_dict

            # Если для данного ключа были заготовки и переменные x
            # pipe_keys_from_groups должен быть актуальным списком ключей, для которых есть n_blanks > 0
            if key_tuple_res in pipe_keys_from_groups and x_val_clean_res.size > 0:
                for j_idx_res in range(len(x_val_clean_res)):
                    # Добавляем строку только если есть использование со склада или недобор (для первой заготовки этой группы ключа)
                    # и если это первая заготовка для ключа или если x_val_clean_res[j_idx_res] > 0
                    # Условие для добавления строки:
                    # 1. Если есть использование со склада (x_val > порога)
                    # 2. ИЛИ Если это первая обрабатываемая заготовка для данного ключа (j_idx_res == 0),
                    #    И есть недобор (s_val > порога),
                    #    И при этом нет никакого использования со склада для ВСЕХ заготовок этого ключа.
                    #    (чтобы не дублировать строку с недобором для каждой заготовки, если склад не используется)
                    
                    add_row = False
                    if abs(x_val_clean_res[j_idx_res]) > 1e-6:
                        add_row = True
                    elif j_idx_res == 0 and abs(s_val_clean_res_scalar) > 1e-6:
                        # Проверяем, есть ли вообще использование со склада для этого ключа
                        if not np.any(np.abs(x_val_clean_res) > 1e-6):
                            add_row = True
                    
                    if add_row:
                        results_list_mip.append({
                            'Типоразмер трубы': pipe_type,
                            'Вн. марка/Марка': steel_grade,
                            'Номер заказа': num_zakaza,
                            'Позиция заказа': pos_zakaza,
                            'Длина заготовки': current_lengths_res[j_idx_res] if j_idx_res < len(current_lengths_res) else None,
                            'ОстКонПериода_исходный': current_volumes_res[j_idx_res] if j_idx_res < len(current_volumes_res) else None,
                            'Диаметр заготовки': current_diams_res[j_idx_res] if j_idx_res < len(current_diams_res) else None,
                            '№ плавки1': current_plavki_res[j_idx_res] if j_idx_res < len(current_plavki_res) else None,
                            'Количество со склада': x_val_clean_res[j_idx_res],
                            # Недобор указываем только для первой строки группы (или если только недобор)
                            'Недобор': s_val_clean_res_scalar if j_idx_res == 0 else 0.0 
                        })
            # Если для ключа не было заготовок (он не в pipe_keys_from_groups), но есть заказ и, возможно, недобор
            elif key_tuple_res not in pipe_keys_from_groups and abs(s_val_clean_res_scalar) > 1e-6 :
                 results_list_mip.append({
                    'Типоразмер трубы': pipe_type, 'Вн. марка/Марка': steel_grade,
                    'Номер заказа': num_zakaza, 'Позиция заказа': pos_zakaza,
                    'Длина заготовки': None, 'ОстКонПериода_исходный': None, 
                    'Диаметр заготовки': None, '№ плавки1': None,
                    'Количество со склада': 0.0, 
                    'Недобор': s_val_clean_res_scalar # Используем скаляр
                })
                
    df_results_raw_mip = pd.DataFrame(results_list_mip)
    if not df_results_raw_mip.empty:
         df_results_raw_mip = df_results_raw_mip[
             (df_results_raw_mip['Количество со склада'].abs() > 1e-7) | 
             (df_results_raw_mip['Недобор'].abs() > 1e-7)
         ].copy()

    return df_results_raw_mip, optimal_value, x, s