from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn 
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = FastAPI(title='Proyecto Individual MLOPS ',description='Joaquin Millan Lanhozo')

movies_language = pd.read_csv("movies_language.csv")
movies = pd.read_csv("Duracion_peliculas.csv")
merge_franquicias = pd.read_csv("merge_franquicias.csv")
countries_counts = pd.read_csv("countries_counts.csv")
productoras_exitosas_df = pd.read_csv("productoras_exitosas_df.csv")
franquicias = pd.read_csv("franquicias.csv")
directores = pd.read_csv("directores.csv")
modelo = pd.read_csv("modelo.csv")


@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo
    
     Por ejemplo: "en"
    '''
    idioma = idioma.lower()
    idioma = str(idioma)

    fila_idioma_seleccionado = movies_language[movies_language['country_code'].values == idioma]

    if idioma in movies_language.country_code.values:
        count = fila_idioma_seleccionado['original_language'].values[0]
        int_value = count.item()

        return {'idioma':idioma, 'cantidad':count}

    else: 
        return { "message":"Codigo de idioma mal ingresado"}
    

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año de estreno de la pelicula

    Por ejemplo: Jumanji    
    '''
    pelicula = pelicula.lower().title()
    pelicula = str(pelicula)

    if pelicula in movies["title"].values:
        
        pelicula_info = movies[movies["title"]== pelicula].title.values[0]
        duracion = movies[movies["title"]== pelicula].runtime.values[0]
        anio = movies[movies["title"]== pelicula].release_year.values[0]
        return {'pelicula':pelicula_info, 'duracion':duracion, 'anio':anio}

    else:
        return {"message" : "Lo lamentamos! La pelicula buscada no tiene esta informacion al respecto" }
    
    
@app.get('/franquicia/{franquicia}')
def franquicias(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio
    
    Por ejemplo: James Bond Collection
    '''
    franquicia = str(franquicia)
    if franquicia in merge_franquicias["Franchise"].values:
        
        name_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].Franchise.values[0]
        cantidad_de_peliculas_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].Movie_Count.values[0]
        ganancia_total_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].revenue.values[0]
        ganancia_promedio_franquicia = merge_franquicias[merge_franquicias["Franchise"]== franquicia].ganancia_promedio_por_franquicia.values[0]
        
        
        return {'franquicia':franquicia, 'cantidad':cantidad_de_peliculas_franquicia, 'ganancia_total':ganancia_total_franquicia, 'ganancia_promedio':ganancia_promedio_franquicia}
        
    else:
        return {"message":"Lo lamentamos! La franquicia buscada no tiene esta informacion al respecto" }


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo
    
    Por ejemplo: Argentina
    '''
    
    pais = str(pais)
    if pais in countries_counts["production_countries_names"].values:
        
        name_pais = countries_counts[countries_counts["production_countries_names"]== pais].production_countries_names.values[0]
        conteo_paises = countries_counts[countries_counts["production_countries_names"]== pais].Movie_Count.values[0]

        return {'pais':name_pais, 'cantidad':conteo_paises}

    else:

        return {"message":"Lo lamentamos! El pais buscado no produjo peliculas"}

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio
    
    Por ejemplo: Warner Bros.
    '''
    productora = str(productora)

    if productora in productoras_exitosas_df.Production_Company.values:
        
        name_productora = productoras_exitosas_df[productoras_exitosas_df["Production_Company"]== productora].Production_Company.values[0]
        cantidad_de_peliculas_productora = productoras_exitosas_df[productoras_exitosas_df["Production_Company"]== productora].Movie_Count.values[0]
        ganancia_total_productora = productoras_exitosas_df[productoras_exitosas_df["Production_Company"]== productora].revenue.values[0]
        
        
        return {'Productora':name_productora, 'cantidad':cantidad_de_peliculas_productora, 'revenue_total':ganancia_total_productora}
        
    else:
        return {"message":"Lo lamentamos! La franquicia buscada no tiene esta informacion al respecto" }


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director):
    """
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo 
    medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, 
    retorno individual, costo y ganancia de la misma, en formato lista.

    Por ejemplo: John Lasseter
    """ 
    
    peliculas_director = directores[directores['director_name'] == nombre_director]
    
    
    if peliculas_director.empty:
        return "Director no encontrado"  

    retorno_total_director = peliculas_director['return'].sum()
    

    peliculas = []
    for index, row in peliculas_director.iterrows():
        pelicula_info = {
            'nombre': row['title'],
            'anio': row['release_date'],
            'retorno_pelicula': row['return'],
            'budget_pelicula': row['costo'],
            'revenue_pelicula': row['revenue']
        }
        peliculas.append(pelicula_info)
    
    
    respuesta = {
        'director': nombre_director,
        'retorno_total_director': retorno_total_director,
        'peliculas': peliculas
    }
    
    return respuesta

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''

    # Buscar la película por título en la columna 'title'
    movie = modelo[modelo['title'] == titulo]

    if len(movie) == 0:
        return {"message":"La película no se encuentra en la base de datos."}

    # Obtener el género y la popularidad de la película
    movie_genero = movie['genres_names'].values[0]
    movie_popularity = movie['popularity'].values[0]

    # matriz de características para el modelo de vecinos más cercanos
    features = modelo[['popularity']]
    genres = modelo['genres_names'].str.get_dummies(sep=' ')
    features = pd.concat([features, genres], axis=1)

    # Manejar valores faltantes (NaN) reemplazándolos por ceros
    features = features.fillna(0)

    # modelo de vecinos más cercanos
    nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn_model.fit(features)

    # Encontrar las películas más similares (excluyendo la película de consulta indicada por usuario)
    _, indices = nn_model.kneighbors([[movie_popularity] + [0] * len(genres.columns)], n_neighbors=6)
    similar_movies_indices = indices[0][1:]  # Excluyendo la primera película que es la misma consulta
    recomendadas = modelo.iloc[similar_movies_indices]['title']

    # Si la película de consulta está en la lista de recomendaciones, la eliminamos
    if titulo in recomendadas.tolist():
        recomendadas = recomendadas[recomendadas != pelicula]

    return {'lista recomendada': 
            recomendadas}


df_movies2 = modelo[[ 'overview', 'title', 'popularity']]
df_movies2["overview"] = df_movies2["overview"].astype(str)
tfidf= TfidfVectorizer(stop_words='english')
df_movies2['overview'] = df_movies2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df_movies2['overview'])

cosine_sim= linear_kernel(tfidf_matrix, tfidf_matrix)
@app.get('/recomendacion_td/{titulo}')
def recomendacion_td(titulo:str, cosine_sim= cosine_sim):

    indices=pd.Series(df_movies2.index, index=df_movies2['title']).drop_duplicates()

    idx= indices[titulo]
    
    sim_scores= list(enumerate(cosine_sim[idx]))
    
    sim_scores= sorted(sim_scores, key=lambda x : x[1], reverse=True)
    
    sim_scores=sim_scores[1:6]
    
    movies_indices= [i[0] for i in sim_scores]
    
    return df_movies2['title'].iloc[movies_indices]