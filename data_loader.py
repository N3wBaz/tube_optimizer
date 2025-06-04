# data_loader.py
import pandas as pd
from typing import Tuple, List, Optional # Добавил Optional

# Константы для СПИСКОВ КОЛОНОК остаются здесь
SAP_COLS = [
    'ОстКонПериода', 'ФактичСтоим', 'Длина заготовки', 'Марка стали',
    'Диаметр', '№ плавки', 'Вн. марка/Марка', 'Партия'
]
MODEL_COLS = [
    'Длина заготовки', 'Диаметр заготовки', 'Типоразмер трубы', 'РКМ', 'Длина'
]
ORDER_COLS = [ # Исходные имена колонок из Excel
    'Типоразмер', 'Вн. марка/Марка', 'Неотгруженное кол-во', 'Номер заказа',
    'Позиция заказа', 'Марка стали', 'Длина труб от', 'Длина труб до'
]

def load_data_from_sheet(
    file_path: str, sheet_name: str, use_cols: Optional[List[str]]
) -> pd.DataFrame:
    # ... (без изменений) ...
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=use_cols)
        if data.empty and use_cols: # Проверка use_cols, чтобы не было ошибки при use_cols=None
            print(f"Предупреждение: Загружен пустой DataFrame с листа '{sheet_name}'.")
        if use_cols: # Проверяем только если use_cols не None
            missing_cols = [col for col in use_cols if col not in data.columns]
            if missing_cols:
                # Позволим загрузить, но с предупреждением, если некоторых колонок нет
                print(f"Предупреждение: На листе '{sheet_name}' отсутствуют некоторые из ожидаемых колонок: {missing_cols}. "
                      f"Присутствуют: {list(data.columns)}. Загрузка будет продолжена с имеющимися колонками.")
                # Если критично, чтобы были все - можно здесь бросать ValueError
                # raise ValueError(...) 
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        raise
    except ValueError as e: # Ловим ошибки от read_excel, например, "Worksheet named '...' not found"
        print(f"Ошибка при чтении листа '{sheet_name}' из '{file_path}': {e}")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка при чтении листа '{sheet_name}' из '{file_path}': {e}")
        raise


def load_all_data(
    file_path: str,
    sap_sheet_name: str,
    model_sheet_name: str,
    order_sheet_name: str
    # Списки колонок теперь используются из констант этого модуля
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    print(f"Загрузка данных из файла: {file_path}")

    sap_df = load_data_from_sheet(file_path, sap_sheet_name, SAP_COLS)
    print(f"... Данные САП ('{sap_sheet_name}') загружены. Строк: {len(sap_df)}")

    math_model_df = load_data_from_sheet(file_path, model_sheet_name, MODEL_COLS)
    print(f"... Данные модели ('{model_sheet_name}') загружены. Строк: {len(math_model_df)}")

    order_df = load_data_from_sheet(file_path, order_sheet_name, ORDER_COLS)
    print(f"... Данные заказов ('{order_sheet_name}') загружены. Строк: {len(order_df)}")

    return sap_df, math_model_df, order_df