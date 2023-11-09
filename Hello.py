
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
        page_title="Olympische Spelen",
        page_icon="👋",
    )
    worldpop = pd.read_csv("world_population.csv")
    df = pd.read_csv('Tokyo 2021 dataset v3.csv')
    df2 = pd.read_csv('athlete_events3.csv')
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
    Life = pd.read_csv("life expectancy.csv")
    worldpop = pd.read_csv("world_population.csv")



    st.title("Olympische Spelen")
    st.image('olympischespelenrio_pixabay.jpg', use_column_width=True)
    
    figkaart=Kaart(df)
    st.plotly_chart(figkaart)
    st.write("""
Op de kaart is in een snel overzicht et zien welke landen de meeste olympische medailles hebben gewonnen zonder onderscheiding in de medailles: Goud, zilver en brons. Omdat de kaart de heatmap functie heeft kan je zo gemakkelijk zien wat de hoeveelheid per land is.""")

    fig_box = boxmetslider(merged_data2)
    st.plotly_chart(fig_box)

    st.write("""
In deze boxplot is te zien wat de spreiding is van de leeftijd per sport. Je kan goed outliers zien en snel een goed overzicht krijgen van de leeftijd per sport.""")

    st.write("""
In deze stacked bar plot is te zien hoeveel medailles elk land per continent heeft gewonnen, ook onder te verdelen in gouden, zilveren en bronze medailles. Zo krijg je toch al heel snel een mooi overzicht van de data. """)

    #st.pyplot(fig=MedailleVerdelingPerLandEnContinent(df), clear_figure=None, use_container_width=True)
    fig = MedailleVerdelingPerLandEnContinent(df)
    st.pyplot(fig)
    st.write("""
In deze stacked bar plot is te zien hoeveel medailles elk land per continent heeft gewonnen, ook onder te verdelen in gouden, zilveren en bronze medailles. Zo krijg je toch al heel snel een mooi overzicht van de data. """)

####################################################################################################
    df['Team/NOC'] = df['Team/NOC'].replace('United States of America', 'United States')
    df.loc[1, 'Team/NOC'] = "China"
    a= pd.merge(df, Life, left_on=["Team/NOC"],right_on = ['Country Name'])    
    abc= pd.merge(a, worldpop, left_on="Country Name",right_on = 'Country/Territory')
    abc = abc.rename(columns={'Continent_x': 'Continent'})

    cr = abc[['Total','Prevelance of Undernourishment','CO2','Health Expenditure %','Area (km²)']].corr(method = 'pearson')
    fig = go.Figure(go.Heatmap(x=cr.columns, y = cr.columns, z = cr.values.tolist(), colorscale = 'rdylgn', zmin = -1, zmax = 1))

    st.plotly_chart(fig)
    st.write("""In deze plot kan je de Correlatie zien tussen de varabelen in de dataset""")
    abc = abc[abc['Year'] ==2019]

    abc['Life Expectancy World Bank'] = abc['Life Expectancy World Bank'].fillna(abc['Life Expectancy World Bank'].mean())
    abc['Prevelance of Undernourishment'] = abc['Prevelance of Undernourishment'].fillna(abc['Prevelance of Undernourishment'].mean())
    abc['CO2'] = abc['CO2'].fillna(abc['CO2'].mean())
    abc['Health Expenditure %'] = abc['Health Expenditure %'].fillna(abc['Health Expenditure %'].mean())
    abc['Sanitation'] = abc['Sanitation'].fillna(abc['Sanitation'].mean())
    



    fig = px.scatter(abc, x='2020 Population', y='Total', title='Spreidingsdiagram totaal aantal medailles tegen wereld populatie'
                , trendline='ols', hover_data= ['Country Name'],color = 'Continent')
    st.plotly_chart(fig)

    fig = px.scatter(abc, x='Prevelance of Undernourishment', y='Total', title='Spreidingsdiagram totaal aantal medailles tegen ondervoeding'
                 ,  hover_data= ['Team/NOC'],color = 'Continent')
    st.plotly_chart(fig)

    fig = px.scatter(abc, x='CO2', y='Total', title='Spreidingsdiagram totaal aantal medailles tegen CO2'
                 , trendline='ols',  hover_data= ['Team/NOC'],color = 'Continent')
    st.plotly_chart(fig)

    fig = px.scatter(abc, x='2020 Population', y='Total', title='Spreidingsdiagram totaal aantal medailles tegen aantal inwoners'
                 , trendline='ols',  hover_data= ['Team/NOC'],color = 'Continent')
    st.plotly_chart(fig)


    X = abc[['Prevelance of Undernourishment','CO2','Health Expenditure %','Area (km²)']]
    y = abc['Total']
    # Add a constant (intercept)
    X = sm.add_constant(X)

    # Create the Linear Regression Model
    model = sm.OLS(y, X).fit()

    # Get Regression Summary
    summary = model.summary()

    st.write('Formula: -12.1819 + -0.1218 * Undernourishment+ 7.884e-06*CO2+2.7159 * Health Expenditure + 1.509e-06 * Area (km²)')
    st.write("R-squared: 0.736")

    ypred = model.predict(X)
    X['Aantal medailles'] = ypred
    XYZ = X.merge(abc, on = ['Prevelance of Undernourishment','CO2','Health Expenditure %','Area (km²)'])
    fig = px.bar(XYZ.sort_values("Aantal medailles", ascending=False).head(10), x='Team/NOC', y='Aantal medailles'
                 , title='Voorspelling voor 2024', color ='Team/NOC' )
    fig.update_xaxes(title_text='Land')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
