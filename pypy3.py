from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
mass,description,periastrontime,semimajoraxis,discoveryyear,
list,eccentricity,period,discoverymethod,lastupdate,periastron,name

"""
planets = pd.read_csv('planets.csv')
"""
The Pipeline is built using a list of (key, value) pairs,
where the key is a string
containing the name you want to give
this step and value is an estimator object:
"""
kmeans_pipeline = Pipeline([
 ('scale', StandardScaler()),#StandardScaler() will normalize the features i.e.
#each column of X, INDIVIDUALLY, so that each column/feature/variable will have μ = 0 and σ = 1

 ('kmeans', KMeans(8, random_state=0))#8 clusters de valor 
 ])

kmeans_data = planets[['semimajoraxis', 'period']].dropna() #de estos valores elimina los valores cero y basura
print("esta es la respuesta")
print(kmeans_pipeline)
print(kmeans_pipeline.fit(kmeans_data))
"""
The next thing you will probably want to do is to estimate some parameters
in the model. This is implemented in the fit() method.
The fit() method takes the training data as arguments,
which can be one array in the case of unsupervised learning, or two
arrays in the case of supervised learning.
Pipeline(steps=[('scale', StandardScaler()),
('kmeans', KMeans(random_state=0))])
"""

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

sns.scatterplot(
x=kmeans_data.semimajoraxis,# el valor de kmeans para semimajoraxis
y=kmeans_data.period,# el valor de kmeans pero de periodo
hue=kmeans_pipeline.predict(kmeans_data),# 
ax=ax, palette='Accent'#
)


i=0
ax.set_yscale('log')#
solar_system = planets[planets.list == 'Solar System']#
for planet in solar_system.name:
    data = solar_system.query(f'name == "{planet}"')
   
    ax.annotate(
         planet,
         (data.semimajoraxis, data.period),
         (7 + data.semimajoraxis, data.period),
         arrowprops=dict(arrowstyle='->')
 )
ax.set_title('log(orbital period) vs. semi-major axis')
#plt.show()
"""
fig = plt.figure(figsize=(7, 7))
sns.heatmap(
planets.drop(columns='discoveryyear').corr(),
 center=0, vmin=-1, vmax=1, square=True, annot=True,
cbar_kws={'shrink': 0.8}
 )
#plt.show()
"""
sns.scatterplot(
    x= planets.semimajoraxis, y= planets.period,
    hue= planets.list, alpha=0.5)
plt.title('period vs semimajoraxis')
plt.legend( title="")
plt.show()
