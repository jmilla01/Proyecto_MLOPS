<p align=center><img src=https://neurona-ba.com/wp-content/uploads/2021/07/HenryLogo.jpg><p>

# <h1 align=center> **PROYECTO INDIVIDUAL MACHINE LEARNING OPERATIONS (MLOPS)** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>







<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

# JOAQUIN MILLAN LANHOZO - AGOSTO 2023 - DTPT02

## SOY HENRY

El trabajo aquí expuesto fue realizado durante la cursada en la institución Soy Henry, formo parte de la cohorte DATAPT02 y este es el primer proyecto integrador enfocado en Machine Learning Operations
<hr>  


## Tabla de contenidos
- [Archivos en el Repositorio](#archivos-en-el-repositorio)
- [Contexto](#contexto)
- [Proyecto](#proyecto)
- [Deployment](#deployment)
- [Fuentes de datos](#fuente-de-datos)
- [Video deploy](#video-deploy)
- [Stack tecnologico](#stack-tecnologico)

<hr> 

## ARCHIVOS EN EL REPOSITORIO

<hr> 

+ Carpeta "env" --> Tiene el entorno virtual generado para este proyecto.
+ "duracion_peliculas.csv" --> Archivo especifico generado para la consulta de la función "peliculas_duracion"
+ "peliculas_paises.csv" --> Csv generado para la función de "peliculas_pais"
+ "productoras_exitosas.csv" --> Sub dataset confeccionado para la consulta de la función de "productoras_exitosas"
+ countries_counts.csv --> Este csv cuenta con información del conteo de las películas realizadas por los países. Función "peliculas_pais"
+ "directores.csv" --> Dataframe exportado a csv realizado para la función de "get_director"
+ "franquicias.csv" --> Csv elaborado con la información necesaria para la función "franquicia"
+ "main.py" --> Archivo en donde corren las funciones y por donde realice el deploy a FastApi y disponibilizandolo luego por Render.
+ "merge_franquicias.csv" --> Csv elaborado con la información necesaria para la función "franquicia"
+ "productoras_exitosas_df.csv" --> Sub dataset confeccionado para la consulta de la función de "productoras_exitosas"
+ "requirements.txt" --> Archivo con las librerías utilizadas y sus versiones.

[Tabla de Contenidos](#tabla-de-contenidos)




## Contexto

Para este proyecto, simulamos ser Data Scientist para una start-up en la industria del Streaming. La start-up aun no cuenta con una plataforma desarrollada para poder obtener información respecto de las películas con las que cuenta la empresa. 

A lo largo del proyecto, se realizaran tareas de ETL (Extraccion, Transformación y Carga de datos) se extraerá información de los datasets propuestos, se realizaran algunas transformaciones a algunas columnas para poder tener dataframes más robustos y completos para poder realizar el EDA (Análisis Exploratorio de Datos) que es un análisis de los datos para poder obtener información útil.

Luego, realizaremos un modelo de machine learning para poder dar recomendaciones de películas y finalmente, el desarrollo de una aplicación para poder disponibilizar esos datos para que los usuarios puedan consumirlos.

[Tabla de Contenidos](#tabla-de-contenidos)

## Proyecto

+ Como primer paso, realizamos la lectura de los datasets propuestos, son dos datasets: uno de "movies", con información de las películas y otro de "credits" con información del elenco que realizaron las películas. Los datasets originalmente cuentan con 45466 películas

+ Transformaciones:  La data que contenían los datasets no son perfectos, es por eso que se realizan transformaciones para poder utilizar esa data. Algunas columnas fueron transformadas, ya que contenían datos con formatos no adecuados y también datos agrupados, que había que desanidar para poder disponibilizar el dato que contenían esas columnas. También se eliminaron columnas inútiles, que no serán consideradas para el propósito de este proyecto.

+ ** Tratamientos de valores faltantes**: Hay algunos valores nulos a los cuales se realizó un tratamiento de caso que se haya considerado necesario


## Creacion de Funciones
  Se crearon dataframes específicos y más concretos para eficientizar y acotar el uso de las siguientes funciones. Estas funciones servirán para poder consultar información de las películas.


+ **peliculas_idioma( *`Idioma`: str* )**:
    Función en la cual se ingresa un idioma y retorna la cantidad de películas producidas en ese idioma.

Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en `idioma`*
         

+ **peliculas_duracion( *`Pelicula`: str* )**:
    Función a la cual se le ingresa una pelicula y devuelve la duración y el año de estreno.

Ejemplo de retorno: *`X` . Duración: `x`. Año: `xx`*

+ **franquicia( *`Franquicia`: str* )**:
    Esta función recibe un input de la franquicia, y retorna la cantidad de peliculas, ganancia total de la franquicia y el promedio de ganancias
    
Ejemplo de retorno: *La franquicia `X` posee `X` peliculas, una ganancia total de `x` y una ganancia promedio de `xx`*

+ **peliculas_pais( *`Pais`: str* )**:
    La función al ingresarle un país , devuelva la cantidad de peliculas que fueron producidas en el mismo.
    
Ejemplo de retorno: *Se produjeron `X` películas en el país `X`*

+ **productoras_exitosas( *`Productora`: str* )**:
    Esta función recibe el nombre de una productora, y entrega la ganancia total y la cantidad de peliculas que realizó. 
    
Ejemplo de retorno: *La productora `X` ha tenido un revenue de `x`*

+ **get_director( *`nombre_director`* )**:
    Al ingresar en esta función el nombre de un director que se encuentre en el dataset devuelve el éxito del mismo medido a través del retorno. Además, devuelve el nombre de cada película dirigida con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

+ **Sistema de recomendación**: 
    A esta función, se ingresa el nombre de una película y recomienda 5 peliculas similares.


### EDA - Analisis Exploratorio de Datos

Se realizo un análisis exhaustivo de los datos, en el cual podemos ver datos sobre los idiomas hablados en las películas, información respecto de los países productores, los lenguajes originales de las películas. Las compañías productoras, colecciones de películas, las fechas de estreno, los presupuestos, ganancias, puntajes y valoraciones, géneros, actores y directores. 

El dataset acompaña y refleja muy bien, a lo que es la historia del cine. Hitos históricos, sociales y tecnológicas contribuyeron al desarrollo de la industria. 

El 28 de diciembre de 1895, los hermanos Lumière proyectaron una serie de cortometrajes en París, lo que se considera una de las primeras proyecciones públicas de películas en movimiento. Esta fecha a menudo se toma como el comienzo oficial del cine.

Luego, vino el Technicolor en las décadas de 1910 y 1920 lo que llevó al uso más generalizado y a la mejora en la calidad del color en el cine.

Durante esos años posteriores, hasta la década del 1930, las películas eran en blanco y negro. A partir de la década del 30, acompañado de ciertos hitos las películas comenzaron a desarrollarse en mayor cantidad. Estos hitos fueron: El sonido y el habla (anteriormente sin sonido), nacimiento de los musicales, "estreno" de las estrellas de cine y glamour, expansión de géneros. Además del sonido, hubo avances técnicos en áreas como la cinematografía, la iluminación y la edición. Estos avances permitieron una mayor calidad visual y narrativa en las películas. Lo que resultó en una mayor popularidad, diversidad y sofisticación de las películas.

Posterior le siguieron las películas animadas en los años 40. Snow White and the Seven Dwarfs (1937), esta película, producida por Walt Disney, es considerada la primera película animada en color y uno de los primeros largometrajes animados. Fue un gran hito en la historia de la animación.

Luego en 1960, también se nota otra marcada tendencia del incremento de películas producidas. Esto se debe a ciertos factores:

- Surgió lo que se conoció como el "Nuevo Hollywood", donde directores jóvenes y ambiciosos comenzaron a desafiar las convenciones y a explorar temas más oscuros y realistas.

- A nivel global, surgieron movimientos cinematográficos influyentes, como la Nouvelle Vague en Francia y el Neorrealismo italiano. Estos movimientos buscaban formas frescas y auténticas de narrar historias en la pantalla.

-  A medida que las películas se volvían más accesibles y la globalización avanzaba, el cine de diferentes países y culturas ganó popularidad en todo el mundo.

También hay otro crecimiento exponencial marcado en los años 80, ya que las innovaciones tecnológicas, como el uso de efectos especiales y el formato widescreen, permitieron nuevas posibilidades visuales en el cine. A finales de la década de 1970 y principios de la década de 1980, surgieron películas de gran presupuesto con un enfoque en el entretenimiento a gran escala. "Star Wars" (1977) y "Jaws" (1975) son ejemplos destacados de películas que dieron lugar al cine de blockbusters.

 A medida que las películas se volvían más accesibles y la globalización avanzaba, el cine de diferentes países y culturas ganó popularidad en todo el mundo.


 A partir de la década de 1990, sobre todo de los 2000 en adelante,  la tecnología digital comenzó a revolucionar la producción y distribución de películas. El auge de Internet también cambió la forma en que las películas se promocionaban y se distribuían.Acompañado a ello vino el apogeo de un monton de peliculas junto a avances tecnologicos y desarrollo exponencial de la industria cinematografica.

 [Tabla de Contenidos](#tabla-de-contenidos)
=======
 A partir de la década de 1990, sobre todo de los 2000 en adelante, la tecnología digital comenzó a revolucionar la producción y distribución de películas. El auge de Internet también cambió la forma en que las películas se promocionaban y se distribuían. Acompañado a ello vino el apogeo de un montón de películas junto a avances tecnológicos y desarrollo exponencial de la industria cinematográfica.


## Deployment

Para finalizar, se realiza la disponibilizacion en un servicio web para que pueda ser consumida por los usuarios 
+ [API](https://ejemplo-joaquinmillan-deploy.onrender.com/docs)

[Tabla de Contenidos](#tabla-de-contenidos)

## Fuente de datos

+ [Dataset](https://drive.google.com/drive/folders/1mfUVyP3jS-UMdKHERknkQ4gaCRCO2e1v): Carpeta con los 2 archivos con datos que requieren ser procesados (movies_dataset.csv y credits.csv), tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0): Diccionario con algunas descripciones de las columnas disponibles en el dataset.


[Tabla de Contenidos](#tabla-de-contenidos)

## Video Deploy
+ [Video](https://drive.google.com/file/d/1uGitaE-bxqTLBstm2RpZiYsGZ_pokkSk/view?usp=drive_link)
=======
# Video Deploy
+ [Video](https://drive.google.com/file/d/1uGitaE-bxqTLBstm2RpZiYsGZ_pokkSk/view?usp=drive_link) Video con el deploy y la prueba de las funciones.


[Tabla de Contenidos](#tabla-de-contenidos)

## Stack tecnologico

+  Notebook desarrollada en python a traves de Visual Studio Code.
+  Libreria numpy -- [Numpy](https://numpy.org/) Numerical Pyhton útil para realizar cálculos lógicos y matemáticos sobre cuadros y matrices.
+  Libreria pandas -- [Pandas](https://pandas.pydata.org/) Pandas es una librería de Python especializada en el manejo y análisis de estructuras de datos
+  Libreria matplotlib -- [Matplotlib](https://matplotlib.org/) Matplotlib libreria para la visualización de datos.
+  Libreria seaborn -- [Seaborn](https://seaborn.pydata.org/) Seaborn libreria para la visualización de datos.
+  Libreria datetime -- [Datetime](https://docs.python.org/es/3/library/datetime.html) Datetime libreria utilizada para la transformación a formato fecha
+  Libreria missingno -- [Missingno](https://pypi.org/project/missingno/) Miisingno visualización efectiva de datos nulos
+  Libreria ast -- [ast](https://docs.python.org/3/library/ast.html) El módulo ast ayuda a las aplicaciones de Python a procesar árboles de la gramática de sintaxis abstracta de Python.
+  Libreria sklearn -- [sklearn](https://scikit-learn.org/stable/) Sklearn libreria utilizada para el modelo de Machine Learning
+  Libreria plotly [Plotly](https://plotly.com/python/) Plotly libreria para la visualizacion de datos.
+  FastApi [FastApi](https://fastapi.tiangolo.com/) FastAPI es un web framework para la creacion de APIs con Python 3.7
+  Render [Render](https://render.com/) Utilzacion de Render para el deploy de la API

[Tabla de Contenidos](#tabla-de-contenidos)

