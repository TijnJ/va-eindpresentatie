
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
    
    olympics2016 = pd.read_csv('athlete_events3.csv')
    regions = pd.read_csv('noc_regions.csv')
    Life = pd.read_csv("life expectancy.csv")
    cow = pd.read_csv("countries of the world.csv")
    athlete = pd.read_csv("athlete_events3.csv")
    merged_df = pd.merge(athlete, regions, on="NOC", how="inner")
    merged_df = merged_df[merged_df['Year'] == 2016]
    medal_dummies = pd.get_dummies(merged_df['Medal'])
    medal_dummies['No_Medal'] = 1  # Initialize 'No_Medal' column to 1
    medal_dummies['No_Medal'] = medal_dummies['No_Medal'].where(medal_dummies['Bronze'] + medal_dummies['Silver'] + medal_dummies['Gold'] < 1, 0)
    bbb = pd.concat([merged_df, medal_dummies], axis=1)
    merged_df = bbb.drop('Medal', axis=1)
    countries = merged_df.groupby(['Team','NOC',"Games",'Year','Season'])[['Bronze', 'Gold', 'Silver', 'No_Medal']].sum()
    countries['total'] = 0
    countries['total'] = countries['total'].astype(int)
    countries['total'] = countries['Bronze'].astype(int)+countries['Gold'].astype(int)+countries['Silver'].astype(int)
    countries = countries.reset_index()
    olympics2016 = countries[countries['Year']==2016].merge(worldpop, left_on=['Team'], right_on=['Country/Territory'], how='inner')
    countries2016= olympics2016.merge(Life, left_on=['Year', 'Team'], right_on=['Year', 'Country Name'], how='inner')
    meanv = countries2016['CO2'].mean()
    countries2016['CO2'] = countries2016['CO2'].fillna(meanv)
    europe2016 = countries2016[countries2016['Continent']=='Europe']

    st.title("Olympische Spelen")
    st.image('olympischespelenrio_pixabay.jpg', caption='Your Image Caption', use_column_width=True)
    
    figkaart=Kaart(df)
    st.plotly_chart(figkaart)

    fig_box = boxmetslider(merged_data2)

    st.plotly_chart(fig_box)
    

    #st.pyplot(fig=MedailleVerdelingPerLandEnContinent(df), clear_figure=None, use_container_width=True)
    fig = MedailleVerdelingPerLandEnContinent(df)
    st.pyplot(fig)
    st.write("""
In deze stacked bar plot is te zien hoeveel medailles elk land per continent heeft gewonnen, ook onder te verdelen in gouden, zilveren en bronze medailles. Zo krijg je toch al heel snel een mooi overzicht van de data. """)

    cr = countries2016[['total','Area (km²)','2015 Population','CO2']].corr(method = 'pearson')
    fig = go.Figure(go.Heatmap(x=cr.columns, y = cr.columns, z = cr.values.tolist(), colorscale = 'rdylgn', zmin = -1, zmax = 1))
    st.title("corelatie van de variabelen met totaal aantal medailes")
    st.plotly_chart(fig)
    st.write("""In deze plot kan je de corelatie zien tussen de varabelen in de dataset""")
    

    fig = px.scatter(countries2016, x='Area (km²)', y='total', title='Spreidingsdiagram totaal aantal medailes tegen opperflakte'
                 , trendline='ols',  hover_data= ['Team'], color = 'Continent')
    st.plotly_chart(fig)

    fig = px.scatter(countries2016, x='CO2', y='total', title='Spreidingsdiagram totaal aantal medailes tegen CO2'
                 , trendline='ols',  hover_data= ['Team'], color = 'Continent')
    st.plotly_chart(fig)

    fig = px.scatter(countries2016, x='2015 Population', y='total', title='Spreidingsdiagram totaal aantal medailes tegen wereld populatie'
                 , trendline='ols',  hover_data= ['Team'], color = 'Continent')
    st.plotly_chart(fig)
    
    # Extract features and target variable
    X = europe2016[['Area (km²)','2015 Population','CO2']]
    y = europe2016['total']

    # Add a constant (intercept)
    X = sm.add_constant(X)

    # Create the Linear Regression Model
    model = sm.OLS(y, X).fit()

    # Get Regression Summary

    coefficients = model.params
    intercept = coefficients['const']
    coeffs = coefficients.drop('const')
    formula = f'Y = {intercept:.2f} + ' + ' + '.join([f'{coeff:.2f} * {feature}' for feature, coeff in zip(coeffs.index, coeffs)])
    st.write('Formula: 4.8131 + 6.775e-06 * Area (km²)+ -8.696e-08 * 2015 Population+2.377e-05 * CO2')
    st.write("R-squared: 0.509")

    # Extract features and target variable
    X = countries2016[['Area (km²)','2015 Population','CO2']]
    y = countries2016['total']

    # Add a constant (intercept)
    X = sm.add_constant(X)

    # Create the Linear Regression Model
    model = sm.OLS(y, X).fit()

    # Get Regression Summary
    summary = model.summary()
    print(summary)
    ypred = model.predict(X)
    X['predicted wins'] = ypred
    XYZ = X.merge(countries2016, on = ['Area (km²)', '2015 Population','CO2'])
    fig = px.bar(XYZ.sort_values("predicted wins", ascending=False).head(10), x='Team', y='predicted wins', title='top 10 landen met de meeste madailles in 2016')
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
