#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df1 = pd.read_csv("C:/Users/abhay/OneDrive/Desktop/IIRSPROJECT/world population.csv")
print(df1.head())


# In[3]:


print(df1.head(100))


# # shape of  worlds_population_dataset1

# In[4]:


print("No of rows in worlds_population_dataset1=",df1.shape[0])
print("No of columns in worlds_population_dataset1=",df1.shape[1])


# In[5]:


df2 = pd.read_csv("C:/Users/abhay/OneDrive/Desktop/IIRSPROJECT/world population by country.csv")
print(df2.head(234))


# # shape of  worlds_population_dataset2

# In[6]:


print("No of rows in worlds_population_dataset2=",df2.shape[0])
print("No of columns in worlds_population_dataset2=",df2.shape[1])


# # population trends from 2000 to 2023
# 

# In[7]:


# Select rows where 'COUNTRY' is equal to 'India'
x2 = df1[df1['COUNTRY'] == 'India']

# Extract the value in the 'COUNTRY' column from the first row
x3 = x2['COUNTRY'].values[0]

# Print the value of x1
print(x3)

#  '2000', '2001', ..., '2023' are column names in DataFrame
years = [str(year) for year in range(2000, 2024)]

# Extract values for the specified years
y2 = x2[years]

# Print the values for the years 2000 through 2023
print(y2)


# In[8]:


india_population = df1[df1['COUNTRY'] == 'India']

# Extracting years and population values
years = list(map(str, range(2000, 2024)))
population_values = india_population.iloc[:, 1:].values.flatten()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(years, population_values, marker='o', linestyle='-')
plt.title('India Population (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.show()


# # SORTING OUT TOP 10 COUNTRIES FROM 234 COUNTRIES POPULATION WISE

# In[9]:


import pandas as pd


# Sort the DataFrame based on the population in the year 2023 in descending order
df1_sorted = df1.sort_values(by='2023', ascending=False)

# Select the top 10 countries
top_10_countries = df1_sorted.head(10)

# Display the result
print(top_10_countries)


# # PLOTTING TOP 10 COUNTRIIES FROM YEAR 2000 TO 2023
# 

# In[10]:


import matplotlib.pyplot as plt

# Select the top 10 countries based on the latest year (2023) population
top_countries = df1.nlargest(10, '2023')

# Transpose the DataFrame for easier plotting
top_countries_transposed = top_countries.transpose()

# Extract the country names for labeling
countries = top_countries_transposed.iloc[0]

# Remove the 'COUNTRY' row before plotting
top_countries_transposed = top_countries_transposed[1:]

# Plotting
plt.figure(figsize=(12, 8))
for country in top_countries_transposed.columns:
    plt.plot(top_countries_transposed.index, top_countries_transposed[country], label=country)

# Assign country names to each line
for country, color in zip(top_countries_transposed.columns, plt.rcParams['axes.prop_cycle'].by_key()['color']):
    plt.text(top_countries_transposed.index[-1], top_countries_transposed[country].iloc[-1], country, color=color)

plt.title('Population Growth of Top 10 Countries (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Population in Million')
plt.legend()
plt.show()


# 95     -      India  
# 42     -      China    
# 223 - United States   
# 96    -   Indonesia      
# 157   -    Pakistan      
# 150     -   Nigeria    
# 27     -     Brazil      
# 16   -   Bangladesh      
# 170   -      Russia    
# 132    -     Mexico    

# From this graph, if we compare India and china population in year 2000 ,china population is much more than India population.
# In year 2019 china population is increasing slowly and in year 2022 china recorded its first drop in population due to one child policy in his country.
# India population is increasing rapidly from 2000 to 2022 and in year 2023 as of recent data India became most populus country in the world.

# From this graph,we can clearly see that Russia population is constantly decreasing due to death rate in russia is more than birth rate.
# people from Russia is also leaving due to Russia and ukraine war.

# In[11]:


df2.head(100)


# In[12]:


a1=df2['country']
print(a1)
a2=df2["world_share"]
print(a2)


# # Plotting continents based on their shares of the world population.

# In[13]:


import matplotlib.pyplot as plt

# Group by region and calculate the sum of world_share for each region
region_wise_share = df2.groupby('region')['world_share'].sum()

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(region_wise_share, labels=region_wise_share.index, autopct='%1.1f%%', startangle=140)
plt.title('World Share Distribution by Region')
plt.show()



# # factors affecting world poulation growth

# In[14]:


df2.head(234)


# # countries based on net migration of people

# In[15]:


# Replace missing values in the 'net_migrants' column with 0
df2['net_migrants'].fillna(0, inplace=True)
#top 10 countries based on net migrants
top_countries = df2.nlargest(10, 'net_migrants')

# plotting 
plt.figure(figsize=(12, 8))
plt.bar(top_countries['country'], top_countries['net_migrants'], color='b')
plt.xlabel('Country')
plt.ylabel('Net Migrants')
plt.title('Top 10 Countries Based on Net Migrants')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
# Select top 10 countries based on fertility rate
top_10_countries = df2.nlargest(10, 'fertility_rate')

# Define a color map for the bars
colors = plt.cm.Paired(np.arange(len(top_10_countries)))

# Plot using a different type of plot (e.g., horizontal bar plot)
plt.figure(figsize=(12, 8))
bars = plt.barh(top_10_countries['country'], top_10_countries['fertility_rate'], color=colors, edgecolor='black', alpha=0.7)
plt.title('Top 10 Countries based on Fertility Rate')
plt.xlabel('Fertility Rate')
plt.ylabel('Country')

# Adding labels for each bar
for bar, txt in zip(bars, top_10_countries['fertility_rate']):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{txt:.2f}', ha='left', va='center')

plt.show()



# # top 10 country based on median age

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

# Sort the data by median age
df2_sorted = df2.sort_values(by='median_age', ascending=False)

# Select the top 10 and lowest 10 countries based on median age
top_10_countries = df2_sorted.head(10)
lowest_10_countries = df2_sorted.tail(10)

plt.figure(figsize=(12, 8))
plt.plot(top_10_countries['country'], top_10_countries['median_age'], marker='o', linestyle='-', color='blue', label='Top 10')

# Create a line chart for the lowest 10 countries
plt.plot(lowest_10_countries['country'], lowest_10_countries['median_age'], marker='o', linestyle='-', color='orange', label='Lowest 10')

# Adding labels and title
plt.xlabel('Country')
plt.ylabel('Median Age')
plt.title('Top 10 and Lowest 10 Countries Based on Median Age')
plt.legend()

# Adding grid lines
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.show()


# A country's median age indicates whether its population is younger or older. A higher median age indicates an older population, while a lower median age indicates a younger population.
# From a median age, the government and policymakers use it for social services, education, and healthcare facilities.

# In[18]:


df2.head(234)


# In[19]:


# Sorting the dataframe by density in descending order and selecting the top 10
top_10_density_countries = df2.sort_values(by='density', ascending=False).head(10)

print(top_10_density_countries[['country', 'density']])


# In[20]:


# Sorting the dataframe by density in descending order
top_10_density_countries = df2.sort_values(by='density', ascending=False).head(10)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(top_10_density_countries['country'], top_10_density_countries['density'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Density')
plt.title('Top 10 Countries based on Density')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import pandas as pd



# Sort the DataFrame by land area in descending order and get the top 10
top_10_land_area = df2.sort_values(by='land_area', ascending=False).head(10)

# Sort the DataFrame by population in descending order and get the top 10
top_10_population = df2.sort_values(by='population', ascending=False).head(10)

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.barh(top_10_land_area['country'], top_10_land_area['land_area'], color='green')
plt.title('Top 10 Countries by Land Area')
plt.xlabel('Country')
plt.ylabel('Land Area (sq km)')


# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:




