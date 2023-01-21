import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go

import io
import requests

from PIL import Image
from streamlit_extras.badges import badge

# Import sklearn methods/tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, max_error

# Import all sklearn algorithms used
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def main():
    col1, col2, col3 = st.columns([0.05, 0.265, 0.035])
    
    with col1:
        url = 'https://github.com/tsu2000/insurance/raw/main/images/shield.png'
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        st.image(img, output_format = 'png')

    with col2:
        st.title('&nbsp; Insurance ML Application')

    with col3:
        badge(type = 'github', name = 'tsu2000/insurance', url = 'https://github.com/tsu2000/insurance')

    st.markdown('### 📋 &nbsp; Insurance Machine Learning Web App')
    st.markdown('This web application aims to explore the accuracy of various regression models for the selected insurance dataset with different features. The original source of the data can be found [**here**](<https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender>).')

    # Initialise dataframe
    url = 'https://raw.githubusercontent.com/tsu2000/insurance/main/insurance.csv'
    df = pd.read_csv(url)

    # Checking for null/infinite values:
    # df[df.isin([np.NaN, -np.Inf, np.Inf]).any(axis=1)] # Checks if any data has missing/infinite values

    st.write('')

    options = st.selectbox('Select a feature/machine learning model:', ['Exploratory Data Analysis',
                                                                        'Linear Regression',
                                                                        'Ridge Regression',
                                                                        'Lasso Regression',
                                                                        'Decision Tree Regressor',
                                                                        'Random Forest Regressor'])

    st.markdown('---')

    if options == 'Exploratory Data Analysis':
        eda(data = df)
    elif options == 'Linear Regression':
        linreg_model(data = df)
    elif options == 'Ridge Regression':
        ridge_model(data = df)
    elif options == 'Lasso Regression':
        lasso_model(data = df)
    elif options == 'Decision Tree Regressor':
        dt_model(data = df)
    elif options == 'Random Forest Regressor':
        rf_model(data = df)


def scaled_processing(data):
    # Preprocessing and scaling data
    ohe = OneHotEncoder(drop = 'first')
    mms = MinMaxScaler()

    # Make column transformer
    ct = make_column_transformer(
        (ohe, ['sex', 'smoker', 'region']),
        (mms, ['age', 'bmi', 'children']),
        remainder = 'passthrough'
    )

    prep_df = ct.fit_transform(data)
    df = pd.DataFrame(columns = ct.get_feature_names_out(), data = prep_df)

    return df


def cat_processing(data):
    # Processing only categoical variables
    ohe = OneHotEncoder(drop = 'first')

    # Make column transformer
    ct = make_column_transformer(
        (ohe, ['sex', 'smoker', 'region']),
        remainder = 'passthrough'
    )

    prep_df = ct.fit_transform(data)
    df = pd.DataFrame(columns = ct.get_feature_names_out(), data = prep_df)

    return df


@st.cache(suppress_st_warning = True, allow_output_mutation = True)
def regression_report(yTest, yPred, model, Data, Target, input_cv):
    df = pd.DataFrame(
        index = ['R-Squared Score', 
                 'Explained Variance Score', 
                 'Mean Absolute Error (MAE)', 
                 'Mean Squared Error (MSE)', 
                 'Root Mean Squared Error (RMSE)', 
                 'Maximum Error',
                 f'Cross Validation (k = {input_cv})'],
        data = [r2_score(yTest, yPred), 
                explained_variance_score(yTest, yPred),
                mean_absolute_error(yTest, yPred), 
                mean_squared_error(yTest, yPred), 
                mean_squared_error(yTest, yPred, squared = False), 
                max_error(yTest, yPred),
                np.mean(cross_val_score(model, Data, Target, cv = input_cv))],
        columns = ['Values']
    ).round(4)

    fig = go.Figure(data = [go.Table(columnwidth = [4, 1.75],
                                header = dict(values = ['<b>Regression Metric<b>', '<b>Score/Value<b>'],
                                                fill_color = 'navy',
                                                line_color = 'darkslategray',
                                                font = dict(color = 'white', size = 16)),
                                    cells = dict(values = [df.index, df.values], 
                                                fill_color = [['lightblue']*6 + ['palegreen']],
                                                line_color = 'darkslategray',
                                                align = ['right', 'left'],
                                                font = dict(color = [['navy']*6 + ['darkgreen'], 
                                                                     ['navy']*6 + ['darkgreen']], 
                                                            size = [14, 14]),
                                                height = 25))])
    fig.update_layout(height = 250, width = 700, margin = dict(l = 5, r = 5, t = 5, b = 5))
    return st.plotly_chart(fig, use_container_width = True)


def eda(data):
    st.markdown('## 🔎 &nbsp; Exploratory Data Analysis (EDA)')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Selection')
        df_type = st.selectbox('Select type of DataFrame to be displayed:', ['Initial DataFrame', 'Modified DataFrame'])

        st.markdown('### Initial DataFrame Info:')
        st.markdown('- Pandas DataFrame as extracted from `.csv` file on GitHub.')

        st.markdown('### Modified DataFrame Info:')
        st.markdown("- One-Hot Encoding applied to categorical variables of 'sex', 'smoker', 'region' with one column from each category dropped to avoid the 'dummy variable trap'.")
        st.markdown("- MinMax Scaler applied to numerical variables of 'age', 'bmi', 'children'.")
        st.markdown("- No changes made to the dependent variable 'charges' at all.")

    if df_type == 'Initial DataFrame':
        st.markdown('### Initial DataFrame:')
        st.dataframe(data, use_container_width = True)

    elif df_type == 'Modified DataFrame':
        st.markdown('### Modified DataFrame:')
        mod_data = scaled_processing(data)
        st.dataframe(mod_data, use_container_width = True)

    st.write(f'Shape of data:', data.shape)

    st.markdown('### Summary Statistics:')
    st.dataframe(data.describe(), use_container_width = True)

    data2 = data.copy()

    # Factorise categorical variables
    data2['sex'] = pd.factorize(data.sex)[0]
    data2['smoker'] = pd.factorize(data.smoker)[0]
    data2['region'] = pd.factorize(data.region)[0]    

    df = data2.corr().reset_index().rename(columns = {'index': 'Variable 1'})
    df = df.melt('Variable 1', var_name = 'Variable 2', value_name = 'Correlation')

    st.markdown('### EDA Heatmap:')

    base_chart = alt.Chart(df).encode(
        x = 'Variable 1',
        y = 'Variable 2'
    ).properties(
        title = 'Correlation Matrix between Different Features',
        width = 700,
        height = 700
    )

    heatmap = base_chart.mark_rect().encode(
        color = alt.Color('Correlation',
                          scale = alt.Scale(scheme = 'viridis', reverse = True)
        )
    )

    text = base_chart.mark_text(fontSize = 24).encode(
        text = alt.Text('Correlation', format = ',.2f'),
        color = alt.condition(
            alt.datum['Correlation'] > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )

    final = (heatmap + text).configure_title(
        fontSize = 25,
        offset = 10,
        anchor = 'middle'
    ).configure_axis(
        labelFontSize = 18
    )

    st.altair_chart(final, use_container_width = True, theme = 'streamlit')

    st.markdown('### Mean charges by categorical variables:')

    category = st.radio('Choose a categorical variable to view the average charges for each subgroup:', ['sex', 'smoker', 'region'], horizontal = True)
    st.markdown('---')

    # Dataframe for bar chart
    cat_means = data.groupby(category)['charges'].mean()
    d2 = {f'{category}': cat_means.index, 'Average Charges': cat_means.values}
    df2 = pd.DataFrame(data = d2)

    alt_bar_chart = alt.Chart(df2).mark_bar().encode(
        x = alt.X('Average Charges:Q'),
        y = alt.Y(f'{category}:N', sort = '-x'),
        color = alt.Color(f'{category}', sort = '-x')
    ).properties(
        title = f'Average insurance charges based on {category}',
        width = 700,
        height = 300
    ).configure_title(
        fontSize = 18,
        offset = 30,
        anchor = 'middle'
    )

    st.altair_chart(alt_bar_chart, use_container_width = True, theme = 'streamlit')


def linreg_model(data):
    st.markdown('## 🛣️ &nbsp; Linear Regression')

    st.write('')

    # Set cleaned data and target:
    X = scaled_processing(data).drop('remainder__charges', axis = 1)
    y = scaled_processing(data)['remainder__charges']

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = selected_size)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    y_pred = lin_reg.predict(X_test)
    regression_report(y_test, y_pred, lin_reg, X, y, selected_cv)

    # Attempt predictions
    st.markdown('### 🤔 &nbsp; Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Select an age:', min_value = int(data['age'].min()), 
                                          max_value = int(data['age'].max()), 
                                          value = int(data['age'].mean()))
        sex = st.radio('Choose sex:', ['male', 'female'])

    with col2:
        bmi = st.slider('Select a BMI:', min_value = float(data['bmi'].min()), 
                                         max_value = float(data['bmi'].max()), 
                                         value = float(data['bmi'].mean()))
        smo = st.radio('Smoker?', ['yes', 'no'])

    with col3:
        kid = st.slider('Select no. of children:', min_value = int(data['children'].min()), 
                                                   max_value = int(data['children'].max()), 
                                                   value = int(data['children'].mean()))
        reg = st.radio('Choose Region:', ['northeast', 'northwest', 'southeast', 'southwest'], horizontal = True)

    DF = {'sex_male': 1 if sex == 'male' else 0, 
          'smoker_yes': 1 if smo == 'yes' else 0, 
          'region_northwest': 1 if reg == 'northwest' else 0, 
          'region_southeast': 1 if reg == 'southeast' else 0, 
          'region_southwest': 1 if reg == 'southwest' else 0,
          'age': age,
          'bmi': bmi,
          'children': kid}

    st.markdown('#### User Selection DataFrame')
    x = pd.DataFrame(data = DF, index = [0])

    st.dataframe(x)

    # Create new processed model without scaling:
    X2 = cat_processing(data).drop('remainder__charges', axis = 1)
    y2 = cat_processing(data)['remainder__charges']
    X2train, X2test, y2train, y2test = train_test_split(X2, y2, test_size = selected_size)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X2train, y2train)

    st.markdown(f'Predicted Charges: &emsp; **:red[{round(lin_reg2.predict(x)[0], 2)}]**')

    st.markdown('---')


def ridge_model(data):
    st.markdown('## ⛰️ &nbsp; Ridge Regression')

    st.write('')

    # Set cleaned data and target:
    X = scaled_processing(data).drop('remainder__charges', axis = 1)
    y = scaled_processing(data)['remainder__charges']

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_alpha = st.slider('Select alpha:', min_value = 0.1, max_value = 1.0, value = 0.25)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = selected_size)

    # Initialise machine learning model
    ridge = Ridge(alpha = selected_alpha)
    ridge.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    y_pred = ridge.predict(X_test)
    regression_report(y_test, y_pred, ridge, X, y, selected_cv)

    # Attempt predictions
    st.markdown('### 🤔 &nbsp; Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Select an age:', min_value = int(data['age'].min()), 
                                          max_value = int(data['age'].max()), 
                                          value = int(data['age'].mean()))
        sex = st.radio('Choose sex:', ['male', 'female'])

    with col2:
        bmi = st.slider('Select a BMI:', min_value = float(data['bmi'].min()), 
                                         max_value = float(data['bmi'].max()), 
                                         value = float(data['bmi'].mean()))
        smo = st.radio('Smoker?', ['yes', 'no'])

    with col3:
        kid = st.slider('Select no. of children:', min_value = int(data['children'].min()), 
                                                   max_value = int(data['children'].max()), 
                                                   value = int(data['children'].mean()))
        reg = st.radio('Choose Region:', ['northeast', 'northwest', 'southeast', 'southwest'], horizontal = True)

    DF = {'sex_male': 1 if sex == 'male' else 0, 
          'smoker_yes': 1 if smo == 'yes' else 0, 
          'region_northwest': 1 if reg == 'northwest' else 0, 
          'region_southeast': 1 if reg == 'southeast' else 0, 
          'region_southwest': 1 if reg == 'southwest' else 0,
          'age': age,
          'bmi': bmi,
          'children': kid}

    st.markdown('#### User Selection DataFrame')
    x = pd.DataFrame(data = DF, index = [0])

    st.dataframe(x)

    # Create new processed model without scaling:
    X2 = cat_processing(data).drop('remainder__charges', axis = 1)
    y2 = cat_processing(data)['remainder__charges']
    X2train, X2test, y2train, y2test = train_test_split(X2, y2, test_size = selected_size)
    ridge2 = Ridge(alpha = selected_alpha)
    ridge2.fit(X2train, y2train)

    st.markdown(f'Predicted Charges: &emsp; **:red[{round(ridge2.predict(x)[0], 2)}]**')

    st.markdown('---')


def lasso_model(data):
    st.markdown('## 🪢 &nbsp; Lasso Regression')

    st.write('')

    # Set cleaned data and target:
    X = scaled_processing(data).drop('remainder__charges', axis = 1)
    y = scaled_processing(data)['remainder__charges']

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_alpha = st.slider('Select alpha:', min_value = 0.001, max_value = 0.1, value = 0.025)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = selected_size)

    # Initialise machine learning model
    lasso = Lasso(alpha = selected_alpha)
    lasso.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    y_pred = lasso.predict(X_test)
    regression_report(y_test, y_pred, lasso, X, y, selected_cv)

    # Attempt predictions
    st.markdown('### 🤔 &nbsp; Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Select an age:', min_value = int(data['age'].min()), 
                                          max_value = int(data['age'].max()), 
                                          value = int(data['age'].mean()))
        sex = st.radio('Choose sex:', ['male', 'female'])

    with col2:
        bmi = st.slider('Select a BMI:', min_value = float(data['bmi'].min()), 
                                         max_value = float(data['bmi'].max()), 
                                         value = float(data['bmi'].mean()))
        smo = st.radio('Smoker?', ['yes', 'no'])

    with col3:
        kid = st.slider('Select no. of children:', min_value = int(data['children'].min()), 
                                                   max_value = int(data['children'].max()), 
                                                   value = int(data['children'].mean()))
        reg = st.radio('Choose Region:', ['northeast', 'northwest', 'southeast', 'southwest'], horizontal = True)

    DF = {'sex_male': 1 if sex == 'male' else 0, 
          'smoker_yes': 1 if smo == 'yes' else 0, 
          'region_northwest': 1 if reg == 'northwest' else 0, 
          'region_southeast': 1 if reg == 'southeast' else 0, 
          'region_southwest': 1 if reg == 'southwest' else 0,
          'age': age,
          'bmi': bmi,
          'children': kid}

    st.markdown('#### User Selection DataFrame')
    x = pd.DataFrame(data = DF, index = [0])

    st.dataframe(x)

    # Create new processed model without scaling:
    X2 = cat_processing(data).drop('remainder__charges', axis = 1)
    y2 = cat_processing(data)['remainder__charges']
    X2train, X2test, y2train, y2test = train_test_split(X2, y2, test_size = selected_size)
    lasso2 = Lasso(alpha = selected_alpha)
    lasso2.fit(X2train, y2train)

    st.markdown(f'Predicted Charges: &emsp; **:red[{round(lasso2.predict(x)[0], 2)}]**')

    st.markdown('---')


def dt_model(data):
    st.markdown('## 🌲 &nbsp; Decision Tree Regression')

    st.write('')

    # Set cleaned data and target:
    X = scaled_processing(data).drop('remainder__charges', axis = 1)
    y = scaled_processing(data)['remainder__charges']

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_depth = st.slider('Select maximum depth of tree:', min_value = 1, max_value = 20, value = 5)
        selected_mss = st.slider('Select minimum samples per split:', min_value = 1, max_value = 10, value = 2)
        selected_msl = st.slider('Select minimum samples per leaf:', min_value = 1, max_value = 10, value = 1)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = selected_size)

    # Initialise machine learning model
    tree_reg = DecisionTreeRegressor(max_depth = selected_depth, 
                                     min_samples_split = selected_mss, 
                                     min_samples_leaf = selected_msl)
    tree_reg.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    y_pred = tree_reg.predict(X_test)
    regression_report(y_test, y_pred, tree_reg, X, y, selected_cv)

    # Attempt predictions
    st.markdown('### 🤔 &nbsp; Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Select an age:', min_value = int(data['age'].min()), 
                                          max_value = int(data['age'].max()), 
                                          value = int(data['age'].mean()))
        sex = st.radio('Choose sex:', ['male', 'female'])

    with col2:
        bmi = st.slider('Select a BMI:', min_value = float(data['bmi'].min()), 
                                         max_value = float(data['bmi'].max()), 
                                         value = float(data['bmi'].mean()))
        smo = st.radio('Smoker?', ['yes', 'no'])

    with col3:
        kid = st.slider('Select no. of children:', min_value = int(data['children'].min()), 
                                                   max_value = int(data['children'].max()), 
                                                   value = int(data['children'].mean()))
        reg = st.radio('Choose Region:', ['northeast', 'northwest', 'southeast', 'southwest'], horizontal = True)

    DF = {'sex_male': 1 if sex == 'male' else 0, 
          'smoker_yes': 1 if smo == 'yes' else 0, 
          'region_northwest': 1 if reg == 'northwest' else 0, 
          'region_southeast': 1 if reg == 'southeast' else 0, 
          'region_southwest': 1 if reg == 'southwest' else 0,
          'age': age,
          'bmi': bmi,
          'children': kid}

    st.markdown('#### User Selection DataFrame')
    x = pd.DataFrame(data = DF, index = [0])

    st.dataframe(x)

    # Create new processed model without scaling:
    X2 = cat_processing(data).drop('remainder__charges', axis = 1)
    y2 = cat_processing(data)['remainder__charges']
    X2train, X2test, y2train, y2test = train_test_split(X2, y2, test_size = selected_size)
    tree_reg2 = DecisionTreeRegressor(max_depth = selected_depth, 
                                      min_samples_split = selected_mss, 
                                      min_samples_leaf = selected_msl)
    tree_reg2.fit(X2train, y2train)

    st.markdown(f'Predicted Charges: &emsp; **:red[{round(tree_reg2.predict(x)[0], 2)}]**')

    st.markdown('---')


def rf_model(data):
    st.markdown('## 🏞 &nbsp; Random Forest Regression')

    st.write('')

    # Set cleaned data and target:
    X = scaled_processing(data).drop('remainder__charges', axis = 1)
    y = scaled_processing(data)['remainder__charges']

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_size = st.slider('Select test size:', min_value = 0.15, max_value = 0.35, value = 0.25)
        selected_n_est = st.slider('Select number of estimators:', min_value = 1, max_value = 300, value = 100)
        selected_depth = st.slider('Select maximum depth of tree:', min_value = 1, max_value = 20, value = 5)
        selected_mss = st.slider('Select minimum samples per split:', min_value = 1, max_value = 10, value = 2)
        selected_cv = st.slider('Select number of K-Fold cross-validations:', min_value = 2, max_value = 30, value = 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = selected_size)

    # Initialise machine learning model
    forest_reg = RandomForestRegressor(n_estimators = selected_n_est,
                                       max_depth = selected_depth, 
                                       min_samples_split = selected_mss)
    forest_reg.fit(X_train, y_train)

    # Show results
    st.markdown('### 📊 &nbsp; Results')
    y_pred = forest_reg.predict(X_test)
    regression_report(y_test, y_pred, forest_reg, X, y, selected_cv)

    # Attempt predictions
    st.markdown('### 🤔 &nbsp; Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Select an age:', min_value = int(data['age'].min()), 
                                          max_value = int(data['age'].max()), 
                                          value = int(data['age'].mean()))
        sex = st.radio('Choose sex:', ['male', 'female'])

    with col2:
        bmi = st.slider('Select a BMI:', min_value = float(data['bmi'].min()), 
                                         max_value = float(data['bmi'].max()), 
                                         value = float(data['bmi'].mean()))
        smo = st.radio('Smoker?', ['yes', 'no'])

    with col3:
        kid = st.slider('Select no. of children:', min_value = int(data['children'].min()), 
                                                   max_value = int(data['children'].max()), 
                                                   value = int(data['children'].mean()))
        reg = st.radio('Choose Region:', ['northeast', 'northwest', 'southeast', 'southwest'], horizontal = True)

    DF = {'sex_male': 1 if sex == 'male' else 0, 
          'smoker_yes': 1 if smo == 'yes' else 0, 
          'region_northwest': 1 if reg == 'northwest' else 0, 
          'region_southeast': 1 if reg == 'southeast' else 0, 
          'region_southwest': 1 if reg == 'southwest' else 0,
          'age': age,
          'bmi': bmi,
          'children': kid}

    st.markdown('#### User Selection DataFrame')
    x = pd.DataFrame(data = DF, index = [0])

    st.dataframe(x)

    # Create new processed model without scaling:
    X2 = cat_processing(data).drop('remainder__charges', axis = 1)
    y2 = cat_processing(data)['remainder__charges']
    X2train, X2test, y2train, y2test = train_test_split(X2, y2, test_size = selected_size)
    forest_reg2 = RandomForestRegressor(n_estimators = selected_n_est,
                                        max_depth = selected_depth, 
                                        min_samples_split = selected_mss)
    forest_reg2.fit(X2train, y2train)

    st.markdown(f'Predicted Charges: &emsp; **:red[{round(forest_reg2.predict(x)[0], 2)}]**')

    st.markdown('---')
    

if __name__ == "__main__":
    st.set_page_config(page_title = 'Insurance ML App', page_icon = '📋')
    main()