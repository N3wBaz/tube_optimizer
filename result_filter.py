import pandas as pd


    # Функция для выбора лучшей подгруппы
def select_best_subgroup(df):
    # Сортируем подгруппы по приоритету: 
    # 1. количество плавок (возрастание)
    # 2. недобор (возрастание)
    # 3. РКМ (возрастание)
    df_sorted = df.sort_values(
        by=['количество_плавок', 'недобор', 'РКМ'], 
        ascending=[True, True, True]
    )
    
    # Ищем первую подгруппу с положительным количеством на складе
    for i in range(len(df_sorted)):
        if df_sorted.iloc[i]['Количество со склада'] > 0:
            return df_sorted.iloc[i]['original_indices']
    
    # Если не нашли - возвращаем первую подгруппу (даже с 0)
    return df_sorted.iloc[0]['original_indices']

def result_filter(path: str) -> None:
    """Функция для фильтрации результатов оптимизатора

    Args:
        path (str): Путь к файлу с результатами оптимизации
    Output:
        None
    """
    if path is None:
    #     print("Путь к файлу с результатами оптимизации не указан")
    #     raise ValueError("Путь к файлу с результатами оптимизации не указан")

        path = "Подбор_заготовок_алгоритм.xlsx"

    dtype_spec = {
        'Типоразмер трубы': str,
        'Вн. марка/Марка': str,
        'Номер заказа': str,
        'Позиция заказа': str,
        'Количество со склада': float,
        '№ плавки': str,
        'Неотгруженное кол-во': float,
        'РКМ': float
    }

    df = pd.read_excel(
        path, 
        header=0, 
        usecols=list(dtype_spec.keys()),  # Берем столбцы из словаря dtype_spec
        dtype=dtype_spec                  # Передаем спецификацию типов данных
    )
    podbor =df.copy()

    # Создаем столбец с идентификатором группы A.i
    podbor['group_id'] = podbor.groupby(['Номер заказа', 'Позиция заказа']).ngroup() + 1
    podbor['group_id'] = 'A.' + podbor['group_id'].astype(str)

    # Функция для создания подгрупп через марку стали
    def create_subgroup_id(df):
        codes, _ = pd.factorize(df['Вн. марка/Марка'])
        return pd.Series(codes + 1, index=df.index)

    # Применяем к нужному столбцу, исключая группировочные
    podbor['subgroup_id'] = (
        podbor.groupby('group_id', group_keys=False)['Вн. марка/Марка']
        .apply(lambda x: create_subgroup_id(podbor.loc[x.index]))
    )

    # Формируем полный id группы (A.i.j)
    podbor['full_group_id'] = podbor['group_id'] + '.' + podbor['subgroup_id'].astype(str)

    # Удаляем промежуточные столбцы
    podbor = podbor.drop(columns=['group_id', 'subgroup_id'])

    # Группируем по full_group_id и сохраняем индексы оригинальных строк
    grouped = podbor.groupby('full_group_id').agg({
        'Типоразмер трубы': 'first',
        'Вн. марка/Марка': 'first',
        'Номер заказа': 'first',
        'Позиция заказа': 'first',
        'Количество со склада': 'sum',  # Суммируем для анализа
        '№ плавки': lambda x: ', '.join(x.dropna().astype(str).unique()),
        'Неотгруженное кол-во': 'first',
        'РКМ': 'first'
    }).reset_index()

    # Добавляем столбец с индексами оригинальных строк для каждой группы
    grouped['original_indices'] = (
        podbor.groupby('full_group_id', group_keys=False)
        .apply(lambda x: x.index.tolist(), include_groups=False)
        .values
    )

    # исходный порядок столбцов
    grouped = grouped[list(podbor.columns) + ['original_indices']]

    # Вычисляем количество плавок в каждой подгруппе
    grouped['количество_плавок'] = grouped['№ плавки'].apply(
        lambda x: len(str(x).split(', ')) if pd.notna(x) else 0
    )

    # Рассчитываем недобор
    grouped['недобор'] = grouped.apply(
        lambda row: max(0, row['Неотгруженное кол-во'] - row['Количество со склада']), 
        axis=1
    )

    # Извлекаем основную группу
    grouped['основная_группа'] = grouped['full_group_id'].apply(lambda x: '.'.join(x.split('.')[:2]))



    # Получаем индексы строк, которые войдут в финальный результат
    best_indices = []
    for _, group in grouped.groupby('основная_группа', group_keys=False):
        best_indices.extend(select_best_subgroup(group))

    # # Фильтруем оригинальный DataFrame по выбранным индексам
    # best_subgroups = podbor.loc[best_indices].reset_index(drop=True)

    # # Удаляем временные столбцы
    # best_subgroups = best_subgroups.drop(columns=['group_id', 'subgroup_id'], errors='ignore')

    # Загружаем исходные данные СО ВСЕМИ СТОЛБЦАМИ
    full_df = pd.read_excel(path, header=0)

    # Фильтруем строки по сохраненным индексам
    result_df = full_df.loc[best_indices].reset_index(drop=True)


    path = path[:-5] + " отфильтрованный.xlsx"
    result_df.to_excel(path, index=False)

