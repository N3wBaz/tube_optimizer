# main.py
import sys
import pandas as pd
import cvxpy as cp

import settings as cfg # Используем новое имя модуля
from data_loader import load_all_data
from data_preprocessor import preprocess_data
from optimizer import run_optimization, DEFAULT_OPT_PARAMS
from result_saver import finalize_and_save_results
from result_filter import result_filter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main_orchestrator():
    YELLOW = '\033[1;33m'
    GREEN = '\033[1;32m'
    RED = '\033[1;31m'
    RESET = '\033[0m'


    try:
        # Явно загружаем конфигурацию. Можно передать путь, если он не стандартный.
        # cfg.load_config("path/to/your/custom_config.yml")
        cfg.load_config() # Попытается найти config.yml
    except (FileNotFoundError, ValueError) as e:
        print(f"Критическая ошибка при загрузке конфигурации: {e}")
        sys.exit(1)

    # Получаем только согласованные параметры из конфига
    input_file = cfg.get_input_excel_file()
    output_file = cfg.get_output_excel_file()
    sheet_names_config = cfg.get_sheet_names()
    penalties_config = cfg.get_penalties()
    
    print(YELLOW + f"--- Начало процесса оптимизации для файла: {input_file} ---" + RESET)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 50)

    # Логика выбора решателя остается здесь
    mosek_available = cp.MOSEK in cp.installed_solvers()
    selected_solver = cp.MOSEK # По умолчанию
    if mosek_available:
        print("Решатель MOSEK доступен.")
    elif cp.ECOS_BB in cp.installed_solvers():
        print("MOSEK недоступен. Используется ECOS_BB.")
        selected_solver = cp.ECOS_BB
    elif cp.GLPK_MI in cp.installed_solvers():
        print("MOSEK и ECOS_BB недоступны. Используется GLPK_MI.")
        selected_solver = cp.GLPK_MI
    else:
        print(RED + "Критическая ошибка: Подходящий MIP решатель (MOSEK, ECOS_BB, GLPK_MI) не найден." + RESET)
        sys.exit(1)

    try:
        # 1. Загрузка данных
        print(YELLOW + "\n--- Этап 1: Загрузка данных ---" + RESET)
        # Списки колонок (SAP_COLS и т.д.) пока остаются жестко заданными в data_loader.py
        sap_df, model_df, order_df_raw = load_all_data(
            file_path=input_file,
            sap_sheet_name=sheet_names_config.get('sap', 'Выгрузка САП'), # Используем .get для безопасности
            model_sheet_name=sheet_names_config.get('model', 'Модель'),
            order_sheet_name=sheet_names_config.get('order', 'Заказы')
        )
        if sap_df.empty or model_df.empty or order_df_raw.empty:
            print(RED + "Ошибка: Один из исходных DataFrame пуст. Прерывание." + RESET)
            return

        # 2. Предобработка данных
        print(YELLOW + "\n--- Этап 2: Предобработка данных ---" + RESET)
        # Карта переименования колонок заказа пока остается в data_preprocessor.py
        processed_data_for_opt, order_data_for_dict = preprocess_data(
            sap_df, model_df, order_df_raw
        )
        if processed_data_for_opt.empty:
            print(RED + "Ошибка: DataFrame после предобработки пуст. Прерывание." + RESET)
            return
        print(f"Размер processed_data_for_opt (после предобработки): {processed_data_for_opt.shape}")
        if sum(processed_data_for_opt['ОстКонПериода'].unique()) == 0:
            print(GREEN + "Нет вариантов для подбора заказов." + RED + "Остановка." + RESET)
            return

        
    
        # 3. Оптимизация
        print(YELLOW + "\n--- Этап 3: MIP-Оптимизация ---" + RESET)
    
        # Собираем параметры для оптимизатора
        # DEFAULT_OPT_PARAMS из optimizer.py будет содержать все параметры,
        # а penalties_config их перекроет.
        current_opt_params = {
            **DEFAULT_OPT_PARAMS, # Загружаем все дефолты из optimizer.py
            **penalties_config,   # Перекрываем штрафы из конфига
            "solver": selected_solver,
        }
        # Если solver_options вынесены в config.yml и есть геттер cfg.get_solver_options():
        # solver_opts_from_cfg = cfg.get_solver_options()
        # current_opt_params["solver_verbose"] = solver_opts_from_cfg.get("verbose", DEFAULT_OPT_PARAMS["solver_verbose"])
        # current_opt_params["mosek_params"] = solver_opts_from_cfg.get("mosek_params", DEFAULT_OPT_PARAMS["mosek_params"])


        raw_optimization_results, opt_value, _, _ = run_optimization(
            processed_data_for_opt.copy(),
            order_data_for_dict.copy(),
            current_opt_params
        )
        # print(f"Оптимальное значение целевой функции (MIP): {opt_value}" )

        if opt_value == float('inf') or \
           (raw_optimization_results is not None and raw_optimization_results.empty) or \
           raw_optimization_results is None:
            print("Оптимизация не дала допустимого решения или результаты пусты.")
        else:
            print(YELLOW + "\n--- Этап 4: Сохранение результатов ---" + RESET)
            finalize_and_save_results(
                raw_optimization_results,
                processed_data_for_opt,
                output_file
            )
        print(YELLOW + "\n--- Процесс оптимизации подбора заказов успешно завершен ---" + RESET)
        
        print(YELLOW + "\n--- Этап 5: Фильтрация и сохранение результатов ---" + RESET)
        result_filter(output_file)
        print(YELLOW + "\n--- Процесс фильтрации итогового подбора заказов успешно завершен ---" + RESET)

    except FileNotFoundError as e_fnf: # Более специфичный отлов
        print(f"Критическая ошибка: Файл не найден. {e_fnf}")
    except ValueError as ve:
        print(f"Критическая ошибка значения или данных: {ve}")
    except cp.error.SolverError as se:
        print(f"Критическая ошибка решателя CVXPY: {se}")
    except ImportError as ie:
        print(f"Критическая ошибка импорта: {ie}.")
    except Exception as e:
        print(f"Произошла непредвиденная критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(GREEN + "\n--- Завершение работы скрипта ---" + RESET)

if __name__ == "__main__":
    main_orchestrator()