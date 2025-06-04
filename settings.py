"""
Модуль для загрузки и предоставления доступа к параметрам конфигурации.
Конфигурация загружается и кешируется при первом вызове load_config() или get_config_value().
"""
import yaml
import os
from typing import Dict, Any, List, Optional # Добавил List и Optional

CONFIG_FILE_PATH_ENV = "APP_CONFIG_PATH"
DEFAULT_CONFIG_FILE = "config.yml"

# Кеш для загруженной конфигурации
_config_cache: Optional[Dict[str, Any]] = None


DEFAULT_PENALTIES = {
    "slack": 1e6,
    "blanks": 1.0,
    "sets": 10.0,
    "rkm": 1e3,
}

DEFAULT_SOLVER_OPTIONS = {
    "verbose": True,
    "mosek_params": {}
}

def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Загружает конфигурацию из YAML-файла и кеширует ее.
    Порядок поиска пути: переменная окружения, аргумент config_path, DEFAULT_CONFIG_FILE.

    Args:
        config_path (Optional[str]): Явный путь к файлу конфигурации.
        force_reload (bool): Если True, перезагружает конфигурацию, даже если она уже в кеше.

    Returns:
        Dict[str, Any]: Загруженный словарь конфигурации.

    Raises:
        FileNotFoundError: Если файл конфигурации не найден.
    """
    global _config_cache
    if _config_cache is not None and not force_reload:
        return _config_cache

    actual_config_path = os.getenv(CONFIG_FILE_PATH_ENV)
    if not actual_config_path: # Если переменная окружения не задана
        if config_path: # Используем переданный путь
            actual_config_path = config_path
        else: # Используем путь по умолчанию
            actual_config_path = DEFAULT_CONFIG_FILE
    
    if not os.path.exists(actual_config_path):
        # Попробуем найти относительно текущего файла config_manager.py, если путь по умолчанию
        if actual_config_path == DEFAULT_CONFIG_FILE:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path_near_module = os.path.join(base_dir, DEFAULT_CONFIG_FILE)
            if os.path.exists(path_near_module):
                actual_config_path = path_near_module
            else:
                 raise FileNotFoundError(
                    f"Файл конфигурации не найден: '{actual_config_path}' (или '{path_near_module}'). "
                    f"Убедитесь, что он существует, или установите переменную окружения {CONFIG_FILE_PATH_ENV}, "
                    f"или передайте явный путь в load_config()."
                )
        else: # Если был передан явный путь и он не найден
            raise FileNotFoundError(
                f"Файл конфигурации не найден по указанному пути: '{actual_config_path}'."
            )


    print(f"Загрузка конфигурации из: {actual_config_path}")
    with open(actual_config_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    if not isinstance(loaded_config, dict):
        raise ValueError(f"Содержимое файла конфигурации '{actual_config_path}' не является словарем (dict).")

    _config_cache = loaded_config
    return _config_cache

def get_config_value(key_path: str, default: Any = ...) -> Any:
    """
    Получает значение из загруженной конфигурации по пути ключа (например, "penalties.slack").
    Если конфигурация не загружена, сначала пытается загрузить ее с путем по умолчанию.
    Если default не Ellipsis (...) и ключ не найден, возвращает default.
    Если default Ellipsis (...) и ключ не найден, вызывает KeyError.
    """
    config = _config_cache
    if config is None:
        print("Предупреждение: Конфигурация не была явно загружена. Попытка загрузки по умолчанию.")
        try:
            config = load_config() # Попытка загрузить с путем по умолчанию
        except FileNotFoundError:
            if default is not ...: # Если default был предоставлен и файл не найден
                print(f"Файл конфигурации не найден, возвращается значение по умолчанию для '{key_path}'.")
                return default
            raise # Перевыбрасываем FileNotFoundError, если default не задан

    keys = key_path.split('.')
    value = config
    try:
        for key_part in keys:
            if not isinstance(value, dict): # Проверка, что мы все еще в словаре
                raise KeyError(f"Промежуточный ключ в '{key_path}' не ведет к словарю.")
            value = value[key_part]
        return value
    except KeyError:
        if default is not ...: # Используем ... как маркер того, что default не был передан
            return default
        raise KeyError(f"Ключ '{key_path}' не найден в конфигурации, и значение по умолчанию не предоставлено.")
    except TypeError: # Может возникнуть, если value не словарь на одном из шагов
         if default is not ...:
            return default
         raise TypeError(f"Неверный тип данных при доступе к ключу '{key_path}'. Ожидался словарь.")


# --- Функции-геттеры для конкретных секций или параметров ---
# Они будут использовать get_config_value, которая при необходимости вызовет load_config

def get_input_excel_file() -> str:
    return get_config_value("input_excel_file")

def get_output_excel_file() -> str:
    return get_config_value("output_excel_file")

def get_sheet_names() -> Dict[str, str]:
    # Можно добавить значения по умолчанию для каждого листа, если нужно
    return get_config_value("sheet_names", default={
        "sap": "Выгрузка САП", "model": "Модель", "order": "Заказы"
    })

def get_penalties() -> Dict[str, float]:
    file_penalties = get_config_value("penalties", default={})
    # Объединяем со значениями по умолчанию, значения из файла имеют приоритет
    return {**DEFAULT_PENALTIES, **file_penalties}

def get_solver_options() -> Dict[str, Any]:
    file_solver_opts = get_config_value("solver_options", default={})
    # Для вложенных словарей, таких как mosek_params, нужно более глубокое объединение
    default_mosek = DEFAULT_SOLVER_OPTIONS.get('mosek_params', {})
    file_mosek = file_solver_opts.get('mosek_params', {})
    
    final_opts = {**DEFAULT_SOLVER_OPTIONS, **file_solver_opts}
    final_opts['mosek_params'] = {**default_mosek, **file_mosek}
    return final_opts

# # Добавим геттеры для колонок, если решим их выносить
# def get_sap_cols() -> List[str]:
#     return get_config_value("excel_sheets.sap.use_cols", default=[
#         'ОстКонПериода', 'ФактичСтоим', 'Длина заготовки', 'Марка стали',
#         'Диаметр', '№ плавки', 'Вн. марка/Марка'
#     ])

# def get_model_cols() -> List[str]:
#      return get_config_value("excel_sheets.model.use_cols", default=[
#         'Длина заготовки', 'Диаметр заготовки', 'Типоразмер трубы', 'РКМ', 'Длина'
#     ])

# def get_order_cols_initial() -> List[str]: # Исходные имена колонок из Excel
#     return get_config_value("excel_sheets.order.use_cols", default=[
#         'Типоразмер', 'Вн марка/марка', 'Неотгруженное кол-во', 'Номер заказа',
#         'Позиция заказа', 'Марка стали', 'Длина труб от', 'Длина труб до'
#     ])

# def get_order_rename_map() -> Dict[str, str]:
#     return get_config_value("excel_sheets.order.rename_cols", default={
#         'Типоразмер': 'Типоразмер трубы',
#         'Вн марка/марка': 'Вн. марка/Марка'
#     })