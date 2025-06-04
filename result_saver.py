"""
Модуль для финальной обработки результатов оптимизации и сохранения их в Excel.
"""
import pandas as pd
import numpy as np

def get_excel_col_letter(n_col_idx_0_based):
    """Преобразует номер колонки (0-based) в буквенное обозначение Excel (A, B, ..., Z, AA, ...)."""
    n = n_col_idx_0_based + 1 
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

def finalize_and_save_results(
    raw_opt_results_df: pd.DataFrame,
    processed_input_df_with_keys: pd.DataFrame,
    output_file_path: str
) -> None:
    if raw_opt_results_df is None or raw_opt_results_df.empty:
        print("Нет данных для сохранения (результаты оптимизации None или пусты).")
        try:
            with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
                pd.DataFrame({'Сообщение': ['Результаты оптимизации отсутствуют или пусты.']}).to_excel(
                    writer, sheet_name='Результаты оптимизации', index=False
                )
            print(f"Создан пустой файл результатов: {output_file_path}")
        except Exception as e_empty:
            print(f"Ошибка при создании пустого файла Excel: {e_empty}")
        return

    print(f"Подготовка и сохранение результатов в файл: {output_file_path}")

    # --- Блок мержа и очистки данных ---
    merge_on_cols = [
        'Типоразмер трубы', 'Вн. марка/Марка', 'Номер заказа', 'Позиция заказа',
        'Длина заготовки', 'Диаметр заготовки', '№ плавки1'
    ]
    actual_merge_on_cols = [
        col for col in merge_on_cols
        if col in raw_opt_results_df.columns and col in processed_input_df_with_keys.columns
    ]
    
    cols_to_keep_from_processed = list(set(actual_merge_on_cols + [
        'Марка стали', 'РКМ', '№ плавки', 'Всего в плавке, т', 'Штук в плавке', 
        'Длина', 'Склад, объем в трубе', 'Себестоимость', 'Себестоимость в трубе',
        'key', 'order_id', 'Неотгруженное кол-во', 'Партия'
    ]))
    cols_to_keep_from_processed = [
        col for col in cols_to_keep_from_processed if col in processed_input_df_with_keys.columns
    ]
    
    df_to_merge_with = processed_input_df_with_keys[cols_to_keep_from_processed].copy()
    
    final_result_df = pd.merge(
        raw_opt_results_df, df_to_merge_with,
        on=actual_merge_on_cols, how='left', suffixes=('_opt', '_base')
    )

    if 'ОстКонПериода_исходный' in final_result_df.columns:
        final_result_df.rename(columns={'ОстКонПериода_исходный': 'ОстКонПериода'}, inplace=True)
        if 'ОстКонПериода_base' in final_result_df.columns:
            final_result_df.drop(columns=['ОстКонПериода_base'], inplace=True, errors='ignore')
    elif 'ОстКонПериода_base' in final_result_df.columns:
        final_result_df.rename(columns={'ОстКонПериода_base': 'ОстКонПериода'}, inplace=True)

    if 'Длина' in final_result_df.columns:
        final_result_df.rename(columns={'Длина': 'Длина трубы в модели'}, inplace=True)
    cols_to_drop_after_merge = ['key_base', 'order_id_base', 'Себестоимость (уникальная)']
    if 'key_opt' in final_result_df.columns: cols_to_drop_after_merge.append('key_opt')
    base_cols_to_drop = [col for col in final_result_df.columns if col.endswith('_base')]
    opt_cols_to_drop_if_base_exists = []
    for col in final_result_df.columns:
        if col.endswith('_opt'):
            base_equivalent = col[:-4] + '_base'
            original_name = col[:-4]
            if base_equivalent in final_result_df.columns:
                opt_cols_to_drop_if_base_exists.append(col)
            elif original_name not in final_result_df.columns:
                final_result_df.rename(columns={col: original_name}, inplace=True)

    cols_to_drop_after_merge.extend(base_cols_to_drop)
    cols_to_drop_after_merge.extend(opt_cols_to_drop_if_base_exists)
    
    for col_drop in list(set(cols_to_drop_after_merge)):
        if col_drop in final_result_df.columns:
            final_result_df.drop(columns=[col_drop], inplace=True, errors='ignore')
            
    if not final_result_df.empty:
        final_result_df = final_result_df[
            (final_result_df['Количество со склада'].abs() > 1e-7) |
            (final_result_df['Недобор'].abs() > 1e-7)
        ].reset_index(drop=True)

    if not final_result_df.empty:
        if 'order_id' in final_result_df.columns:
             final_result_df['order_id'] = final_result_df['order_id'].astype(str)
        if 'key' in final_result_df.columns:
             final_result_df['key'] = final_result_df['key'].astype(str)

    subset_cols_for_dedup = [
       'Типоразмер трубы', 'Вн. марка/Марка', 'Длина заготовки',
       'ОстКонПериода', 'Диаметр заготовки', '№ плавки1', 'order_id', 
       'Количество со склада', 'Недобор', 'Марка стали', 'РКМ',
       '№ плавки', 'Неотгруженное кол-во', 'Номер заказа', 'Позиция заказа',
       'Склад, объем в трубе', 'Себестоимость', 'Себестоимость в трубе', 'key' 
    ]
    actual_subset_cols_dedup = [col for col in subset_cols_for_dedup if col in final_result_df.columns]
    
    if not final_result_df.empty and actual_subset_cols_dedup:
        temp_stringified_cols_for_dedup = {} 
        current_subset_for_dedup = list(actual_subset_cols_dedup) 

        for col_name in actual_subset_cols_dedup:
            if col_name in ['key', 'order_id']: continue
            if col_name in final_result_df.columns and not final_result_df[col_name].dropna().empty:
                try:
                    first_val = final_result_df[col_name].dropna().iloc[0]
                    if isinstance(first_val, (list, np.ndarray)):
                        temp_col_name = f"{col_name}_str_dedup"
                        final_result_df[temp_col_name] = final_result_df[col_name].astype(str)
                        temp_stringified_cols_for_dedup[col_name] = temp_col_name
                        current_subset_for_dedup = [temp_col_name if c == col_name else c for c in current_subset_for_dedup]
                except IndexError: pass 
        try:
            final_result_df.drop_duplicates(subset=current_subset_for_dedup, keep='first', inplace=True)
        except TypeError as e_dedup:
            print(f"Ошибка при drop_duplicates: {e_dedup}. Колонки для проверки: {current_subset_for_dedup}")

        for _, temp_col in temp_stringified_cols_for_dedup.items():
            if temp_col in final_result_df.columns:
                final_result_df.drop(columns=[temp_col], inplace=True)
    
    if 'order_id' in final_result_df.columns and 'Вн. марка/Марка' in final_result_df.columns:
        if not final_result_df.empty:
             final_result_df.sort_values(by=['order_id', 'Вн. марка/Марка'], inplace=True, kind='mergesort')

    # --- Сохранение в Excel ---
    try:
        with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
            if not final_result_df.empty:
                final_result_df.to_excel(writer, sheet_name='Результаты оптимизации', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Результаты оптимизации']

                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'center',
                    'align': 'center',
                    'bg_color': '#218c9a',
                    'font_color': 'white'
                })

                for col_num, value in enumerate(final_result_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                worksheet.autofilter(0, 0, final_result_df.shape[0], final_result_df.shape[1] - 1)

                for i, col in enumerate(final_result_df.columns):
                    column_len = final_result_df[col].astype(str).map(len).max()
                    header_len = len(col)
                    max_len = max(column_len, header_len) + 2
                    worksheet.set_column(i, i, max_len)

                if 'order_id' in final_result_df.columns:
                    final_result_df["order_id"] = final_result_df["order_id"].astype(str)
                    order_id_col_idx = final_result_df.columns.get_loc('order_id')
                    order_id_col_letter = get_excel_col_letter(order_id_col_idx)

                    pipe_sizes = final_result_df["order_id"].unique()
                    colors = ["#FFFFFF", "#F2F2F2"]
                    size_color_map = {size: colors[i % len(colors)] for i, size in enumerate(pipe_sizes)}

                    start_row = 1
                    end_row = final_result_df.shape[0]

                    for pipe_size, color in size_color_map.items():
                        format_ = workbook.add_format({'bg_color': color})
                        worksheet.conditional_format(
                            f'$A$2:${get_excel_col_letter(len(final_result_df.columns) - 1)}${end_row + 1}',
                            {'type': 'formula', 'criteria': f'=${order_id_col_letter}2="{pipe_size}"', 'format': format_}
                        )
                else:
                    print("Колонка 'order_id' отсутствует, форматирование по ней пропущено.")
            else:
                pd.DataFrame({'Сообщение': ['Результаты оптимизации пусты.']}).to_excel(writer, sheet_name='Результаты оптимизации', index=False)
        print(f"Результаты успешно сохранены в {output_file_path}")
    except Exception as e_save:
        print(f"Ошибка при сохранении файла Excel: {e_save}")
        import traceback
        traceback.print_exc()