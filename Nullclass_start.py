#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[337]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import hashlib
import nltk
import webbrowser
import os
import datetime
import pytz


# In[338]:


nltk.download('vader_lexicon')


# # Data Loading

# In[339]:


apps_df=pd.read_csv('Play Store Data.csv')
reviews_df=pd.read_csv('User Reviews.csv')


# # Data Cleaning

# In[340]:


#Step 2 : Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)


# In[341]:


#Convert the Installs columns to numeric by removing commas and +
apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)


# In[342]:


#Convert Price column to numeric after removing $
apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)


# In[343]:


apps_df


# In[344]:


reviews_df


# # Data Transformation

# In[345]:


merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')


# In[346]:


merged_df


# In[347]:


merged_df.isnull().sum()


# In[348]:


# So none of the columns have any null values


# In[349]:


merged_df['Rating'].value_counts()


# In[350]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)


# In[351]:


#Lograrithmic
apps_df['Log_Installs']=np.log(apps_df['Installs'])


# In[352]:


apps_df['Reviews']=apps_df['Reviews'].astype(int)


# In[353]:


apps_df['Log_Reviews']=np.log(apps_df['Reviews'])


# In[354]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)


# In[355]:


#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']


# In[356]:


SIA = SentimentIntensityAnalyzer()


# In[357]:


#Polarity Scores in SIA
#Positive, Negative, Neutral and Compound: -1 - Very negative ; +1 - Very positive


# In[358]:


review = "This app is amazing! I love the new features."
sentiment_score= SIA.polarity_scores(review)
print(sentiment_score)


# In[ ]:





# In[359]:


reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: SIA.polarity_scores(str(x))['compound'])


# In[360]:


apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')


# In[361]:


apps_df['Year']=apps_df['Last Updated'].dt.year


# In[362]:


apps_df


# In[363]:


apps_df_extract=apps_df[:5]


# In[364]:


apps_df_extract


# In[365]:


# Export to Excel
excel_file = "apps_df.xlsx"  # Replace with desired Excel file name
apps_df_extract.to_excel(excel_file, index=False)

print(f"CSV successfully converted to {excel_file}")


# # Data Visualisation using Plotly

# In[366]:


html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)


# In[367]:


plot_containers=""


# In[368]:


# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# In[369]:


plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# Figure 1

# In[370]:


category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig1,"Category Graph 1.html","The top categories on the Play Store are dominated by tools, entertainment, and productivity apps")
            


# Figure 2

# In[371]:


#Figure 2
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig2.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig2,"Type Graph 2.html","Most apps on the Playstore are free, indicating a strategy to attract users first and monetize through ads or in app purchases")


# Figure 3

# In[372]:


#Figure 3
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig3.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig3,"Rating Graph 3.html","Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users")


# Figure 4

# In[373]:


#Figure 4
sentiment_counts=reviews_df['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig4.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig4,"Sentiment Graph 4.html","Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")


# Figure 5

# In[374]:


#Figure 5
installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x':'Installs','y':'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig5.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig5,"Installs Graph 5.html","The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage")


# Figure 6

# In[375]:


# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "Updates Graph 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")


# Figure 7

# In[376]:


#Figure 7
revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='Revenue by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig7.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig7,"Revenue Graph 7.html","Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential")


# Figure 8

# In[377]:


#Figure 8
genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre','y':'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig8.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig8,"Genre Graph 8.html","Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games")


# Figure 9

# In[378]:


#Figure 9
fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig9.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig9,"Update Graph 9.html","The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings.")


# Figure 10

# In[379]:


#Figure 10
fig10=px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig10.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig10,"Paid Free Graph 10.html","Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for")


# In[380]:


plot_containers_split=plot_containers.split('</div>')


# In[381]:


if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers


# # Webpage Styling

# In[382]:


dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name=viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20px;
        }}
        .plot-container {{
             border: 2px solid #555;
             margin: 10px;
             padding: 10px;
             width: 100%; /* Allow full width */
             max-width: 1200px; /* Prevent it from stretching too much */
             height: auto; /* Auto adjust height */
             overflow: hidden;
             position: relative;
             cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container .plot {{
            width: 100%;
            height: auto; 
        }}
        .plot-container: hover .insights {{
            display: block;
        }}
        </style>
        <script>
            function openPlot(filename) {{
                window.open(filename, '_blank');
                }}
        </script>
    </head>
    <body>
        <div class= "header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
            <h1>Google Play Store Reviews Analytics</h1>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
        </div>
        <div class="container">
            {plots}
        </div>
    </body>
    </html>
    """


# # Task 1

# In[383]:


# Filter apps with more than 1,000 reviews
apps_df_t1 = apps_df[apps_df['Reviews'] > 1000]


# In[384]:


apps_df_t1


# In[385]:


apps_df_t1['Rating_Group'].value_counts()


# In[386]:


# Step 2: Identify the top 5 categories by app count
top_categories = apps_df_t1['Category'].value_counts().nlargest(5).index


# In[387]:


top_categories


# In[388]:


# Step 3: Filter the data for only top 5 categories
apps_df_t1 = apps_df_t1[apps_df_t1['Category'].isin(top_categories)]


# In[389]:


apps_df_t1.head()


# In[390]:


# Step 4: Merge with reviews_df to get sentiment scores
merged_df_t1 = pd.merge(apps_df_t1, reviews_df, on="App", how="inner")


# In[391]:


# Step 5: Define sentiment groups based on compound score
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# In[392]:


merged_df_t1["Sentiment"] = merged_df_t1["Sentiment_Score"].apply(classify_sentiment)


# In[393]:


merged_df_t1.head()


# In[394]:


merged_df_t1.tail()


# In[395]:


merged_df_t1["Rating_Group"] = merged_df_t1["Rating"].apply(rating_group)


# In[396]:


merged_df_t1.head()


# In[397]:


# Step 7: Aggregate data for visualization
sentiment_counts = merged_df_t1.groupby(["Category", "Rating_Group", "Sentiment"]).size().reset_index(name="Count")


# In[398]:


sentiment_counts


# In[399]:


#After applying all the filters for task 1 rating group are left with only two categories


# In[400]:



fig_t1 = px.bar(
    sentiment_counts, 
    x="Category", 
    y="Count", 
    color="Sentiment", 
    barmode="stack",
    facet_col="Rating_Group",
    title="Sentiment Distribution of User Reviews by Category and Rating Group",
    labels={"Category": "App Category", "Count": "Number of Reviews"},
    color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
    width=900,
    height=500
)

fig_t1.update_layout(
    autosize=True,  # Allow automatic resizing
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=50, b=10),
    height=600  # Increased height to avoid cramping
)
# Save the visualization as an HTML file
save_plot_as_html(fig_t1, "Sentiment_Distribution.html", "Sentiment distribution varies significantly across rating groups and categories.")
fig_t1.show()


# # Task 2

# In[401]:


# Convert current UTC time to IST
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.datetime.now(ist).time()


# In[402]:


current_time


# In[403]:


# Define the allowed time window (6 PM - 8 PM IST)
start_time = datetime.time(18, 0)  # 18:00 IST
end_time = datetime.time(20, 0)    # 20:00 IST


# In[404]:


# Aggregate installs by category and exclude unwanted categories
apps_df_t2 = apps_df.groupby('Category')['Installs'].sum().reset_index()


# In[405]:


apps_df_t2


# In[406]:


apps_df_t2 = apps_df_t2[~apps_df_t2['Category'].str.startswith(('A', 'C', 'G', 'S'))]


# In[407]:


apps_df_t2


# In[408]:


# Get the top 5 categories
top_categories_2 = apps_df_t2.nlargest(5, 'Installs')


# In[409]:


top_categories_2


# In[410]:


# Create a new column to highlight categories with installs > 1M
top_categories_2['Highlight'] = top_categories_2['Installs'].apply(lambda x: 'High Installs' if x > 1_000_000 else 'Low Installs')


# In[411]:


top_categories_2


# In[412]:


# Define a list of unique countries
country_list = ['United States', 'India', 'Germany', 'United Kingdom', 'France']
country_mapping = {'United States': 'USA', 'India': 'IND', 'Germany': 'DEU', 'United Kingdom': 'GBR', 'France': 'FRA'}


# In[413]:


# Assign unique countries to each category
top_categories_2['Country'] = [country_list[i] for i in range(len(top_categories_2))]
top_categories_2['iso_alpha'] = top_categories_2['Country'].map(country_mapping)


# In[414]:


top_categories_2


# In[415]:


# Generate the choropleth map only if the time condition is met
if start_time <= current_time <= end_time:
    fig_t2 = px.choropleth(
        top_categories_2, 
        locations='iso_alpha', 
        color='Highlight',
        hover_name='Category',
        hover_data=['Installs'],
        title='Global Installs by App Category',
        color_discrete_map={'Highlighted': 'red', 'Normal': 'blue'}
    )
    
    fig_t2.update_layout(
        geo=dict(bgcolor='black'),
        paper_bgcolor='black',
        font_color='white',
        title_font_size=16
    )
    
    # Save the plot and add it to the dashboard
    save_plot_as_html(fig_t2, "Choropleth_Map_t2.html", "This choropleth map shows the distribution of installs for the top 5 app categories worldwide, highlighting categories with installs exceeding 1 million in red.")
    fig_t2.show()
else:
    print("Current IST time is outside the allowed 6 PM - 9 PM range. Graph will not be displayed.")


# # Task 3 

# In[416]:


# Filter data based on conditions
apps_df_t3= apps_df[
    (apps_df["Content Rating"] == "Teen") &  # Only Teen-rated apps
    (apps_df["Installs"] > 10000) &  # Installs greater than 10k
    (apps_df["App"].str.startswith("E", na=False))  # Apps starting with 'E'
]


# In[417]:


apps_df_t3


# In[418]:


apps_df_t3.head()


# In[419]:


# Extract Year-Month for time series aggregation
apps_df_t3['Year-Month'] = apps_df_t3['Last Updated'].dt.to_period('M')


# In[420]:


apps_df_t3


# In[421]:


# Aggregate total installs per month per category
apps_df_t3 = apps_df_t3.groupby(['Year-Month', 'Category'])['Installs'].sum().reset_index()


# In[422]:


apps_df_t3


# In[423]:


# Convert back to datetime for plotting
apps_df_t3['Year-Month'] = apps_df_t3['Year-Month'].astype(str)
apps_df_t3['Year-Month'] = pd.to_datetime(apps_df_t3['Year-Month'])


# In[424]:


apps_df_t3


# In[425]:


# Ensure sorting for correct MoM growth calculation
apps_df_t3 = apps_df_t3.sort_values(by=['Category', 'Year-Month'])


# In[426]:


apps_df_t3


# In[427]:


# Calculate month-over-month (MoM) growth
apps_df_t3['MoM Growth'] = apps_df_t3.groupby('Category')['Installs'].pct_change() * 100


# In[428]:


apps_df_t3


# In[429]:


# Highlight periods where MoM growth exceeds 20%
apps_df_t3['Significant Growth'] = apps_df_t3['MoM Growth'] > 20


# In[430]:


apps_df_t3


# In[431]:


# Get current IST time
ist_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
allowed_time_range = (18, 21)  # 6 PM to 9 PM IST


# In[432]:


ist_now


# In[433]:


# Restrict graph display between 6 PM - 9 PM IST
if allowed_time_range[0] <= ist_now.hour < allowed_time_range[1]:
    # Plot time series with month-to-month granularity
    fig_t3 = px.line(
        apps_df_t3,
        x='Year-Month',
        y='Installs',
        color='Category',
        title='Time Series Trend of Installs (Teen Apps Starting with "E")',
        markers=True
    )

    # Highlight areas where MoM growth exceeds 20%
    for category in apps_df_t3['Category'].unique():
        category_data = apps_df_t3[apps_df_t3['Category'] == category]
        significant_growth = category_data[category_data['Significant Growth']]

        if not significant_growth.empty:
            fig_t3.add_trace(
                go.Scatter(
                    x=significant_growth['Year-Month'],
                    y=significant_growth['Installs'],
                    fill='tozeroy',
                    mode='none',
                    fillcolor='rgba(255,0,0,0.3)',
                    name=f"Growth >20% ({category})"
                )
            )

    # Update layout styling for month-to-month display
    fig_t3.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size': 16},
        xaxis=dict(
            title='Month-Year',
            title_font={'size': 12},
            tickmode='linear',
            dtick='M1',  # Ensures 1-month intervals
            tickformat="%b %Y"
        ),
        yaxis=dict(title='Total Installs', title_font={'size': 12}),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Save plot
    save_plot_as_html(fig_t3, "TimeSeries_Growth.html", 
                      "Significant growth periods (MoM > 20%) are shaded under the curve.")
else:
    print("Current IST time is outside the allowed 6 PM - 9 PM range. Graph will not be displayed.")


# # Dashboard Integration

# In[434]:


final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)


# In[435]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[436]:


with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)


# In[437]:


webbrowser.open('file://'+os.path.realpath(dashboard_path))


# # ALL TASKS COMPLETED
