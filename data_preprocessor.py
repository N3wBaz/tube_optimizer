# data_preprocessor.py
"""
Модуль для предобработки и объединения данных САП, математической модели и заказов.
Готовит DataFrame для оптимизатора.
"""
import pandas as pd
import numpy as np
from typing import Tuple

PRICE_PURCHASE = 1e+6  # Себестоимость для закупаемых позиций по умолчанию
EPSILON = 1e-1         # Малое число для создания уникальных значений себестоимости


def _validate_input_dataframes(
    sap_df: pd.DataFrame, math_model_df: pd.DataFrame, order_df: pd.DataFrame
) -> None:
    """Проверяет, что входные DataFrame не пусты."""
    if sap_df.empty:
        raise ValueError("Входной DataFrame САП (sap_df) пуст.")
    if math_model_df.empty:
        raise ValueError("Входной DataFrame мат. модели (math_model_df) пуст.")
    if order_df.empty:
        raise ValueError("Входной DataFrame заказов (order_df) пуст.")


def preprocess_data(
    sap_raw: pd.DataFrame, math_model_raw: pd.DataFrame, order_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Предобрабатывает и объединяет данные САП, мат. модели и заказов.
    """
    _validate_input_dataframes(sap_raw, math_model_raw, order_raw)

    sap = sap_raw.copy()
    math_model = math_model_raw.copy()
    order = order_raw.copy()

    # 1. Предобработка САП (изменения из ноутбука)
    # В ноутбуке: sap.rename(columns={'Максимальная длина заготовки' : 'Длина заготовки'}, inplace=True)
    # Это должно быть согласовано с SAP_COLS в data_loader.py. Если там уже 'Длина заготовки', то это не нужно.
    # Предполагаем, что data_loader уже загружает 'Длина заготовки'.

    # Заполнение NaN в 'Вн. марка/Марка' из 'Марка стали'
    if 'Вн. марка/Марка' in sap.columns and 'Марка стали' in sap.columns:
        sap['Вн. марка/Марка'].fillna(sap['Марка стали'], inplace=True)
        # print(f"NaN в 'Вн. марка/Марка' после fillna: {sap['Вн. марка/Марка'].isna().sum()}") # Для отладки

    sap['Длина заготовки'] = sap['Длина заготовки'].astype('int64') # Из ноутбука

    sap_grouped = sap.groupby(['№ плавки'], as_index=False).agg(
        ОстКонПериода_sum=('ОстКонПериода', 'sum'),
        Плавок_count=('№ плавки', 'count')
    ).rename(columns={
        'ОстКонПериода_sum': 'Всего в плавке, т',
        'Плавок_count': 'Штук в плавке'
    })
    sap_ext = sap.merge(sap_grouped, on='№ плавки', how='left')
    
    # Переименование 'Диаметр' в 'Диаметр заготовки' (было раньше, но в ноутбуке порядок колонок в sap_ext.columns другой)
    # Убедимся, что это происходит ДО merge с math_model
    if 'Диаметр' in sap_ext.columns:
        sap_ext.rename(columns={'Диаметр': 'Диаметр заготовки'}, inplace=True)
    
    sap_ext['Марка стали'] = sap_ext['Марка стали'].astype(str)
    sap_ext['Вн. марка/Марка'] = sap_ext['Вн. марка/Марка'].astype(str) # Уже должно быть заполнено
    sap_ext['№ плавки1'] = (
        sap_ext['№ плавки'].astype(str) + '_' +
        (sap_ext.groupby('№ плавки').cumcount() + 1).astype(str)
    )

    # 2. Предобработка Заказов (изменения из ноутбука)
    # Переименование колонок в order
    order.rename(columns={
        'Типоразмер': 'Типоразмер трубы',
        'Вн. марка/Марка': 'Вн. марка/Марка' # Такое имя было в ноутбуке в order_cols
        # 'Марка стали' - остается
        # 'Неотгруженное кол-во' - остается
        # 'Длина труб от' - остается
        # 'Длина труб до' - остается
        # 'Номер заказа' - остается
        # 'Позиция заказа' - остается
    }, inplace=True)
    # Если в data_loader.py в ORDER_COLS другие имена, нужно согласовать там или здесь.
    # Предполагаем, что после rename колонки называются так, как ожидает остальной код.

    order['Длина труб до'] = order['Длина труб до'] + 200 # Из ноутбука
    order['Вн. марка/Марка'] = order['Вн. марка/Марка'].astype(str)
    order['Марка стали'] = order['Марка стали'].astype(str)
    order_df_prepared = order.copy()

    # 3. Предобработка Математической Модели (как в ноутбуке)
    math_model = math_model[
        math_model['Типоразмер трубы'].isin(order['Типоразмер трубы'])
    ].copy()

    def check_length(row_mm, orders_df): # Функция из ноутбука
        matching_orders = orders_df[orders_df['Типоразмер трубы'] == row_mm['Типоразмер трубы']]
        if matching_orders.empty:
            return False
        return any(
            (row_mm['Длина'] >= o_row['Длина труб от']) &
            (row_mm['Длина'] <= o_row['Длина труб до'])
            for _, o_row in matching_orders.iterrows()
        )
    
    if not math_model.empty:
        math_model = math_model[
            math_model.apply(check_length, axis=1, orders_df=order)
        ]

    if not math_model.empty:
        math_model = math_model.loc[
            math_model.groupby(
                ['Длина заготовки', 'Диаметр заготовки', 'Типоразмер трубы'], dropna=False
            )['РКМ'].idxmin()
        ].copy() # Добавил .copy()
    else:
        print("Предупреждение: DataFrame math_model пуст перед или после фильтрации по длинам.")

    # 4. Объединение данных
    if not sap_ext.empty and not math_model.empty and \
       'Длина заготовки' in sap_ext.columns and 'Диаметр заготовки' in sap_ext.columns and \
       'Длина заготовки' in math_model.columns and 'Диаметр заготовки' in math_model.columns:
        filtered_sap = sap_ext.merge(
            math_model, on=['Длина заготовки', 'Диаметр заготовки'], how='inner'
        )
        if not filtered_sap.empty:
            filtered_sap['Вн. марка/Марка'] = filtered_sap['Вн. марка/Марка'].astype(str)
        else:
            print("Предупреждение: filtered_sap пуст после merge sap_ext и math_model.")
    else:
        print("Предупреждение: sap_ext или math_model пусты или не содержат ключей для merge. filtered_sap будет пуст.")

        expected_cols = list(sap_ext.columns) + [col for col in math_model_raw.columns if col not in sap_ext.columns]
        filtered_sap = pd.DataFrame(columns=expected_cols)


    # В ноутбуке был how='inner', что может отсеять заказы без подходящих заготовок.
    # Возвращаю how='right', чтобы все заказы учитывались хотя бы для недобора 's'.
    # Если нужен 'inner', то pipe_keys в оптимизаторе должен строиться из merged_df.groupby('key').
    merged_df = filtered_sap.merge(
        order_df_prepared,
        on=['Типоразмер трубы', 'Вн. марка/Марка'],
        how='right',
        suffixes=('_sap', '_order')
    )
    print(f"Размер merged_df после merge с order: {merged_df.shape}")


    # 5. Постобработка merged_df
    unknown_counter = 0
    mask_na_plavki = merged_df['№ плавки'].isna()
    for idx in merged_df[mask_na_plavki].index:
        unknown_counter += 1
        merged_df.loc[idx, '№ плавки'] = f"unknown_{unknown_counter}"

    fill_values_dp = {
        'Длина заготовки': 0, 'ОстКонПериода': 0, 'ФактичСтоим': 0,
        'Всего в плавке, т': 0, 'Диаметр заготовки': 0, 'Штук в плавке': 0,
        'РКМ': 5, 'Длина': 0
    }
    merged_df.fillna(value=fill_values_dp, inplace=True)
    

    merged_df['№ плавки1'].fillna(merged_df['№ плавки'], inplace=True)



    merged_df['Склад, объем в трубе'] = 0.0
    non_zero_rkm_mask = merged_df['РКМ'] != 0
    merged_df.loc[non_zero_rkm_mask, 'Склад, объем в трубе'] = (
        merged_df.loc[non_zero_rkm_mask, 'ОстКонПериода'] /
        merged_df.loc[non_zero_rkm_mask, 'РКМ']
    )
    merged_df['Склад, объем в трубе'] = merged_df['Склад, объем в трубе'].replace(
        [np.inf, -np.inf], 0
    ).fillna(0).round(3)

    merged_df['Себестоимость'] = 0.0

    merged_df['ФактичСтоим'].fillna(0, inplace=True)
    non_zero_ost_mask = merged_df['ОстКонПериода'] != 0
    merged_df.loc[non_zero_ost_mask, 'Себестоимость'] = (
        merged_df.loc[non_zero_ost_mask, 'ФактичСтоим'] /
        merged_df.loc[non_zero_ost_mask, 'ОстКонПериода']
    )
    merged_df['Себестоимость'] = merged_df['Себестоимость'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(PRICE_PURCHASE).round(0)

    merged_df['Себестоимость в трубе'] = (
        merged_df['Себестоимость'] * merged_df['РКМ']
    ).fillna(0).round(2)

    # Очистка и переименование столбцов
    if 'ФактичСтоим' in merged_df.columns: # Уже должно быть обработано, но на всякий случай
        merged_df.drop(columns=['ФактичСтоим'], inplace=True, errors='ignore')
    # Обработка 'Марка стали_sap' и 'Марка стали_order'
    if 'Марка стали_sap' in merged_df.columns and 'Марка стали_order' in merged_df.columns:
        merged_df.drop(columns=['Марка стали_sap'], inplace=True)
        merged_df.rename(columns={'Марка стали_order': 'Марка стали'}, inplace=True)
    elif 'Марка стали_order' in merged_df.columns:
        merged_df.rename(columns={'Марка стали_order': 'Марка стали'}, inplace=True)
    elif 'Марка стали_sap' in merged_df.columns:
         merged_df.rename(columns={'Марка стали_sap': 'Марка стали'}, inplace=True)
            
    if 'Марка стали' in merged_df.columns:
        merged_df['Марка стали'] = merged_df['Марка стали'].astype(str)

    # Создание 'Себестоимость (уникальная)'
    if merged_df['Себестоимость'].isnull().any():
         merged_df['Себестоимость'].fillna(PRICE_PURCHASE, inplace=True)

    merged_df['Себестоимость (уникальная)'] = merged_df['Себестоимость']
    # Проверяем, не пуст ли датафрейм перед groupby
    if not merged_df.empty:
        cumcount = merged_df.groupby("Себестоимость", dropna=False).cumcount()
        mask_duplicates = cumcount > 0
        merged_df.loc[mask_duplicates, 'Себестоимость (уникальная)'] = (
            merged_df.loc[mask_duplicates, "Себестоимость"] +
            cumcount[mask_duplicates] * EPSILON
        )
    
    required_cols_for_key = ['Типоразмер трубы', 'Вн. марка/Марка', 'Номер заказа', 'Позиция заказа']
    if all(col in merged_df.columns for col in required_cols_for_key):
        merged_df['key'] = list(zip(
            merged_df['Типоразмер трубы'],
            merged_df['Вн. марка/Марка'],
            merged_df['Номер заказа'],
            merged_df['Позиция заказа']
        ))
    else:
        missing = [col for col in required_cols_for_key if col not in merged_df.columns]
        print(f"Предупреждение: Не удалось создать столбец 'key' в merged_df, отсутствуют колонки: {missing}")
        merged_df['key'] = None 
    required_cols_for_order_id = ['Номер заказа', 'Позиция заказа']
    if all(col in merged_df.columns for col in required_cols_for_order_id):

        merged_df['order_id'] = merged_df['Номер заказа'].astype(str) + "_" + \
                                merged_df['Позиция заказа'].astype(str)
    else:
        missing_oi = [col for col in required_cols_for_order_id if col not in merged_df.columns]
        print(f"Предупреждение: Не удалось создать столбец 'order_id' в merged_df, отсутствуют колонки: {missing_oi}")
        merged_df['order_id'] = None 

    print(order_df_prepared.dtypes)
    print("Предобработка данных завершена.")
    print(order_df_prepared.columns)
    print(merged_df.columns)
    return merged_df, order_df_prepared 