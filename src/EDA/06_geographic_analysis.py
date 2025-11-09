import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('merged_data.csv')

print("=== Географический анализ ===")

city_stats = df.groupby('Город').agg({
    'salary_target': ['count', 'mean', 'median'],
    'Название': 'count'
}).round(0)
city_stats.columns = ['Вакансий_с_ЗП', 'Средняя_ЗП', 'Медианная_ЗП', 'Всего_вакансий']
city_stats = city_stats.reset_index()

city_coords = {
    'Москва': [55.7558, 37.6173],
    'Санкт-Петербург': [59.9343, 30.3351],
    'Екатеринбург': [56.8431, 60.6454],
    'Новосибирск': [55.0084, 82.9357],
    'Краснодар': [45.0355, 38.9753],
    'Казань': [55.8304, 49.0661],
    'Нижний Новгород': [56.2965, 43.9361],
    'Ростов-на-Дону': [47.2357, 39.7015],
    'Самара': [53.2001, 50.15],
    'Челябинск': [55.1644, 61.4368],
    'Уфа': [54.7431, 55.9678],
    'Воронеж': [51.6720, 39.1843],
    'Пермь': [58.0105, 56.2502],
    'Красноярск': [56.0184, 92.8672],
    'Омск': [54.9885, 73.3242],
    'Тюмень': [57.1522, 65.5272],
    'Волгоград': [48.7194, 44.5018],
    'Ижевск': [56.8528, 53.2115],
    'Хабаровск': [48.4802, 135.0719],
    'Тула': [54.1931, 37.6173]
}

city_stats['lat'] = city_stats['Город'].map(lambda x: city_coords.get(x, [None, None])[0])
city_stats['lon'] = city_stats['Город'].map(lambda x: city_coords.get(x, [None, None])[1])
city_stats = city_stats[city_stats['lat'].notna()].copy()

print(f"Городов с координатами: {len(city_stats)}")

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=city_stats['lon'],
    lat=city_stats['lat'],
    text=city_stats['Город'] + '<br>Вакансий: ' + city_stats['Всего_вакансий'].astype(str) + 
         '<br>Средняя ЗП: ' + city_stats['Средняя_ЗП'].astype(str),
    mode='markers',
    marker=dict(
        size=city_stats['Всего_вакансий'] / 100,
        color=city_stats['Средняя_ЗП'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Средняя ЗП"),
        sizemode='diameter',
        sizeref=2,
        sizemin=4
    ),
    name='Города'
))

fig.update_geos(
    projection_type="natural earth",
    showland=True,
    landcolor="rgb(243, 243, 243)",
    showocean=True,
    oceancolor="rgb(230, 245, 255)",
    showlakes=True,
    lakecolor="rgb(230, 245, 255)",
    showcountries=True,
    countrycolor="rgb(200, 200, 200)",
    lonaxis_range=[19, 180],
    lataxis_range=[41, 82]
)

fig.update_layout(
    title='Распределение вакансий по городам России',
    geo=dict(
        scope='asia',
        center=dict(lon=100, lat=60),
        projection_scale=3
    ),
    height=800
)

fig.write_html('plots/11_geographic_map.html')
print("Карта сохранена в plots/11_geographic_map.html")

fig2 = go.Figure()

fig2.add_trace(go.Scattergeo(
    lon=city_stats['lon'],
    lat=city_stats['lat'],
    text=city_stats['Город'] + '<br>Средняя ЗП: ' + city_stats['Средняя_ЗП'].astype(str),
    mode='markers',
    marker=dict(
        size=city_stats['Средняя_ЗП'] / 1000,
        color=city_stats['Средняя_ЗП'],
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title="Средняя ЗП"),
        sizemode='diameter',
        sizeref=2,
        sizemin=4
    ),
    name='Зарплаты'
))

fig2.update_geos(
    projection_type="natural earth",
    showland=True,
    landcolor="rgb(243, 243, 243)",
    lonaxis_range=[19, 180],
    lataxis_range=[41, 82]
)

fig2.update_layout(
    title='Средняя зарплата по городам России',
    geo=dict(
        scope='asia',
        center=dict(lon=100, lat=60),
        projection_scale=3
    ),
    height=800
)

fig2.write_html('plots/12_salary_map.html')
print("Карта зарплат сохранена в plots/12_salary_map.html")

fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Топ-15 городов по вакансиям', 'Топ-15 городов по зарплате', 
                    'Распределение вакансий', 'Распределение зарплат'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "histogram"}]]
)

top_cities = city_stats.nlargest(15, 'Всего_вакансий')
fig3.add_trace(
    go.Bar(x=top_cities['Город'], y=top_cities['Всего_вакансий'], name='Вакансии'),
    row=1, col=1
)

top_salary = city_stats[city_stats['Средняя_ЗП'].notna()].nlargest(15, 'Средняя_ЗП')
fig3.add_trace(
    go.Bar(x=top_salary['Город'], y=top_salary['Средняя_ЗП'], name='Зарплата'),
    row=1, col=2
)

fig3.add_trace(
    go.Histogram(x=city_stats['Всего_вакансий'], name='Вакансии'),
    row=2, col=1
)

fig3.add_trace(
    go.Histogram(x=city_stats['Средняя_ЗП'], name='Зарплата'),
    row=2, col=2
)

fig3.update_xaxes(title_text="Город", row=1, col=1)
fig3.update_xaxes(title_text="Город", row=1, col=2)
fig3.update_xaxes(title_text="Количество вакансий", row=2, col=1)
fig3.update_xaxes(title_text="Зарплата", row=2, col=2)

fig3.update_yaxes(title_text="Количество", row=1, col=1)
fig3.update_yaxes(title_text="Зарплата", row=1, col=2)
fig3.update_yaxes(title_text="Частота", row=2, col=1)
fig3.update_yaxes(title_text="Частота", row=2, col=2)

fig3.update_layout(
    title_text="Географический анализ вакансий",
    height=1000,
    showlegend=False
)

fig3.write_html('plots/13_geographic_stats.html')
print("Статистика сохранена в plots/13_geographic_stats.html")

print("\nГеографический анализ завершен")

