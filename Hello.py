
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

LOGGER = get_logger(__name__)

def Kaart(df):
    st.title('Totaal aantal medailles per land')
 
    fig = px.choropleth(df,
                    locations='Team/NOC',  # Zorg ervoor dat deze kolom bestaat in je DataFrame
                    locationmode='country names',
                    color='Total',  # Zorg ervoor dat deze kolom bestaat in je DataFrame
                    color_continuous_scale='rdbu'  # Kies een geschikt kleurenschema
                   )
    return fig

def MedailleVerdelingPerLandEnContinent(df):
    teams = df['Team/NOC']
    gold_medals = df['Gold Medal']
    silver_medals = df['Silver Medal']
    bronze_medals = df['Bronze Medal']
    continents = df['Continent']
 
    data = pd.DataFrame({
        'Team': teams,
        'Gold': gold_medals,
        'Silver': silver_medals,
        'Bronze': bronze_medals,
        'Continent': continents
    })
 
    fig, ax = plt.subplots()
 
    def plot_stacked_bar_chart(selected_continent):
        filtered_data = data[data['Continent'] == selected_continent]
        ax.bar(filtered_data['Team'], filtered_data['Gold'], label='Gold')
        ax.bar(filtered_data['Team'], filtered_data['Silver'], bottom=filtered_data['Gold'], label='Silver')
        ax.bar(filtered_data['Team'], filtered_data['Bronze'], bottom=filtered_data['Gold'] + filtered_data['Silver'], label='Bronze')
 
        ax.set_ylabel('Aantal Medailles')
        ax.set_title(f'Medailleverdeling per Land in {selected_continent}')
        ax.set_xticks(filtered_data['Team'])
        ax.set_xticklabels(filtered_data['Team'], rotation=90)
        ax.legend()
 
    selected_continent = st.selectbox('Selecteer een Continent', data['Continent'].unique(), key='continent_selectbox')
    plot_stacked_bar_chart(selected_continent)
 
    return fig

def boxmetslider(merged_data2):

    # Sorteer de unieke jaren in oplopende volgorde
    sorted_years = sorted(merged_data2['Year'].unique())
 
    # Maak een interactieve slider om een jaar te selecteren
    selected_year = st.slider('Selecteer een jaar', min_value=min(sorted_years), max_value=max(sorted_years), value=min(sorted_years), key='year_slider')
 
    # Filter de data voor het geselecteerde jaar
    filtered_data = merged_data2[merged_data2['Year'] == selected_year]
 
    # Maak de boxplot
    fig = px.box(filtered_data, x='Sport', y='Age', labels={'Sport': 'Sport', 'Age': 'Leeftijd'})
 
    # Pas de plot aan
    fig.update_layout(xaxis_title='Sport', yaxis_title='Leeftijd')
 
    # Toon de plot in Streamlit
    return fig


 


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("Olympic games")
    df = pd.read_csv('Tokyo 2021 dataset v3.csv')
    df2 = pd.read_csv('athlete_events2.csv')
    df3 = pd.read_csv('noc_regions.csv')
    mean_age = df2['Age'].mean()  # Bereken het gemiddelde
    df2['Age'].fillna(mean_age, inplace=True)  # Vul ontbrekende waarden in met het gemiddelde
    mean_height = df2['Height'].mean()  # Bereken het gemiddelde
    df2['Height'].fillna(mean_height, inplace=True)  # Vul ontbrekende waarden in met het gemiddelde
    mean_weight = df2['Weight'].mean()  # Bereken het gemiddelde
    df2['Weight'].fillna(mean_weight, inplace=True)  # Vul ontbrekende waarden in met het gemiddelde
    df3 = df3.drop('notes', axis=1)
    merged_data = pd.merge(df, df2, left_on='NOCCode', right_on='NOC')
    merged_data2 = pd.merge(merged_data, df3, on='NOC')
    
    figkaart=Kaart(df)
    st.plotly_chart(figkaart)

    fig_box = boxmetslider(merged_data2)

    st.plotly_chart(fig_box)
    

    #st.pyplot(fig=MedailleVerdelingPerLandEnContinent(df), clear_figure=None, use_container_width=True)
    fig = MedailleVerdelingPerLandEnContinent(df)
    st.pyplot(fig)
    st.write("""
In deze stacked bar plot is te zien hoeveel medailles elk land per continent heeft gewonnen, ook onder te verdelen in gouden, zilveren en bronze medailles. Zo krijg je toch al heel snel een mooi overzicht van de data. """)



if __name__ == "__main__":
    run()
