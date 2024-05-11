from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a secret key for the session

def visualize_data(dataframe):
    visualizations = ""
   
    categorical_columns = dataframe.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        value_counts = dataframe[column].value_counts()
        plt.figure(figsize=(10, 6))
        num_categories = len(value_counts)
        color_palette = sns.color_palette("husl", n_colors=num_categories)
        plt.bar(value_counts.index, value_counts.values,color=color_palette)
        plt.title(f'Bar Chart for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()

        encoded_img = base64.b64encode(img_buf.read()).decode()
        visualizations += f"<h3>Bar Chart for {column}</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"

    # Pie chart for binary categorical columns
    binary_categorical_columns = dataframe.select_dtypes(include=["object"]).columns
    for column in binary_categorical_columns:
        if len(dataframe[column].unique()) == 2:
            value_counts = dataframe[column].value_counts()
            num_categories = len(value_counts)
            color_palette = sns.color_palette("Set3", n_colors=num_categories)
            plt.figure(figsize=(6, 6))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=color_palette)
            plt.title(f'Pie Chart for {column}')
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            plt.close()
            encoded_img = base64.b64encode(img_buf.read()).decode()
            visualizations += f"<h3>Pie Chart for {column}</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"

    # Histogram for numerical columns
    numerical_columns = dataframe.select_dtypes(include=["number"]).columns
    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        color_palette = sns.color_palette("muted")
        plt.hist(dataframe[column], bins=20, color=color_palette[0], edgecolor='black')
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()

        encoded_img = base64.b64encode(img_buf.read()).decode()
        visualizations += f"<h3>Histogram for {column}</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"

    # Heatmap for correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = dataframe.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues' , linewidths=.5)
    plt.title('Heatmap for Correlation Matrix')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    encoded_img = base64.b64encode(img_buf.read()).decode()
    visualizations += f"<h3>Heatmap for Correlation Matrix</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"

    # Boxplot for numerical columns
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        color_palette = sns.color_palette("muted")
        sns.boxplot(x=dataframe[column], color=color_palette[4])
        plt.title(f'Boxplot for {column}')
        plt.xlabel(column)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()

        encoded_img = base64.b64encode(img_buf.read()).decode()
        visualizations += f"<h3>Boxplot for {column}</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"
        
     # Pair plot for numerical columns
    numerical_columns = dataframe.select_dtypes(include=["number"]).columns
    if len(numerical_columns) > 1:
        plt.figure(figsize=(12, 8))
        sns.pairplot(dataframe[numerical_columns])
        plt.suptitle('Pair Plot for Numerical Columns', y=1.02)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()

        encoded_img = base64.b64encode(img_buf.read()).decode()
        visualizations += f"<h3>Pair Plot for Numerical Columns</h3>\n<img src='data:image/png;base64,{encoded_img}'/><br><br>"


    return visualizations

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(dataframe, task_type):
    # Handle missing values
    dataframe = dataframe.dropna()

    # Handle duplicate values
    dataframe = dataframe.drop_duplicates()

    # Detect and handle outliers
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_mask = (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        dataframe.loc[outliers_mask, column] = dataframe[column].median()

    #Handle categorical values using label encoding
    categorical_columns = dataframe.select_dtypes(include=["object"]).columns
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    #dataframe = pd.get_dummies(dataframe)

    # Text preprocessing for relevant text columns
    text_columns = dataframe.select_dtypes(include=["object"]).columns
    for column in text_columns:
        dataframe[column] = dataframe[column].apply(preprocess_text)

    return dataframe

def plot_accuracy_graph_all(algorithms, accuracy_values):
    plt.figure(figsize=(10, 6))
    plt.plot(algorithms, accuracy_values, marker='o', linestyle='-', color='blue')
    plt.title('Result of Various Model')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    # Encode the image data using Base64
    encoded_img = base64.b64encode(img_buf.read()).decode()

    return encoded_img


def plot_accuracy_graph_best(best_model,best_accuracy):
    plt.figure(figsize=(10, 8))
    plt.bar(best_model, best_accuracy, color='green')
    plt.title('Accuracy for Best Models')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    # Encode the image data using Base64
    encoded_img = base64.b64encode(img_buf.read()).decode()

    return encoded_img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        global filename
        filename = os.path.splitext(file.filename)[0]
        #print(filename)
        target_column = request.form['target_column']
        task_type = request.form['task_type']
        
        if file:
            # try:
            # Read the uploaded CSV file into a DataFrame
            print(filename)
            dataframe = pd.read_csv(file)
            # Display the first 5 rows of the dataset
            first_5_rows = dataframe.head().to_html(classes='table table-bordered', index=False)
            last_5_rows = dataframe.tail().to_html(classes='table table-bordered', index=False)

            # Get the shape of the dataset
            dataset_shape = dataframe.shape

            # Show the data types of the columns
            column_types = dataframe.dtypes.to_frame().reset_index().rename(columns={0: 'Data Types'}).to_html(classes='table table-bordered', index=False)

            # Check for missing values
            missing_values = dataframe.isnull().sum().to_frame().reset_index().rename(columns={0: 'Missing Values'}).to_html(classes='table table-bordered', index=False)

            # Check for duplicate values
            duplicate_values = dataframe.duplicated().sum()

            # Show basic statistics of numerical columns
            numeric_stats = dataframe.describe().transpose().reset_index().to_html(classes='table table-bordered', index=False)
            column_names = dataframe.columns.tolist()
            
            

            # Preprocess the data (handle missing, duplicate, and outliers)
            dataframe = preprocess_data(dataframe, task_type)
            
            #dataframes_table = dataframe.dataframes.to_html(classes='table table-bordered', index=False)
            visualizations = visualize_data(dataframe)

            result_str = ""
            # Split data into features (X) and target variable (y)
            X = dataframe.drop(columns=[target_column])
            y = dataframe[target_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalize the features
            numeric_columns = X_train.select_dtypes(include=["number"]).columns
            scaler = StandardScaler()
            X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
            X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

            # Regression models
            # ...

        # Regression models
            if task_type == "regression":
                regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Support Vector Regression": SVR(),
                    # "Lasso Regression": Lasso(),
                    "Random Forest Regressor": RandomForestRegressor()
                }

                results_regression = {}
                r2_values = []  # Initialize the list for R2 values
                best_r2 = float('-inf')
                best_model = ""

                for model_name, model in regression_models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    r2_values.append(r2)
                    results_regression[model_name] = f"{model_name} - Accuracy: {r2*100:.2f}"

                    # Update the best model if a higher R2 score is found
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name

                results = results_regression

                # Suggest the best algorithm and its R2 score
                best_algorithm_suggestion = f"Best Algorithm: {best_model} with Accuracy: {best_r2*100:.2f}"
                results["Best Algorithm Suggestion"] = best_algorithm_suggestion

                # Plot the R2 score graph
                accuracy_graph_reg = plot_accuracy_graph_all(list(regression_models.keys()), r2_values)
                accuracy_graph = plot_accuracy_graph_best(best_model, best_r2)


            # Classification models
            elif task_type == "classification":
                classification_models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Neural Network": MLPClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier()
                }

                results_classification = {}
                accuracy_values = []  # Initialize the list for accuracy values
                best_accuracy = 0
                best_model = ""

                for model_name, model in classification_models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracy_values.append(accuracy)
                    results_classification[model_name] = f"{model_name} - Accuracy: {accuracy*100:.2f}"

                    # Update the best model if a higher accuracy is found
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name

                results = results_classification

                # Suggest the best algorithm and its accuracy
                best_algorithm_suggestion = f"Best Algorithm: {best_model} with Accuracy: {best_accuracy * 100:.2f}%"
                results["Best Algorithm Suggestion"] = best_algorithm_suggestion

                # Plot the accuracy/MSE graph
                
                accuracy_graph_cls = plot_accuracy_graph_all(list(classification_models.keys()), accuracy_values)
                accuracy_graph = plot_accuracy_graph_best(best_model, best_accuracy)
                
        # ...


            # Feature selection using SelectKBest (for classification)
            if task_type == "classification":
                k_best_selector = SelectKBest(f_classif, k= 5)  # You can adjust the number of features (k)
                X_train_k_best = k_best_selector.fit_transform(X_train, y_train)
                X_test_k_best = k_best_selector.transform(X_test)
            else:
                X_train_k_best, X_test_k_best = X_train, X_test  # No feature selection for regression


            # PCA (Principal Component Analysis)
            pca = PCA(n_components=3)  # You can adjust the number of components
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Update the statistics after handling missing, duplicate, and outlier values, balancing, normalization, feature selection, and PCA
            updated_numeric_stats = dataframe.describe().transpose().reset_index().to_html(classes='table table-bordered', index=False)
            pca_columns = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
            X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns).to_html(classes='table table-bordered', index=False)

            result_str += (
         
                "<table border='1' style='border-collapse: collapse;'>"
                "<tr><th>Algorithm and Result</th></tr>"
            )

            for model_name, model_result in results.items():
                result_str += f"<tr><td>{model_result}</td></tr>"

            result_str += "</table><br><br>"
            if task_type == "classification":
                result_str += (        

                f"<h3>Performance of various Algorithm:</h3>\n<img src='data:image/png;base64,{accuracy_graph_cls}'/>"
                f"<h3>Accuracy Graph for Best Algorithm:</h3>\n<img src='data:image/png;base64,{accuracy_graph}'/>"
            )
            else:
                result_str += (
                    f"<h3>Performance of various Algorithm:</h3>\n<img src='data:image/png;base64,{accuracy_graph_reg}'/>"
                    f"<h3>Accuracy Graph for Best Algorithm:</h3>\n<img src='data:image/png;base64,{accuracy_graph}'/>"
                )                
            
            return render_template('index.html', filename = filename,first_5_rows=first_5_rows, last_5_rows = last_5_rows,dataset_shape = dataset_shape , column_types = column_types ,missing_values = missing_values ,
                                   duplicate_values = duplicate_values ,numeric_stats = numeric_stats, column_names = column_names ,visualizations = visualizations ,updated_numeric_stats = updated_numeric_stats,result_str = result_str)
        

    return render_template('index.html')


    
    
if __name__ == '__main__':
     app.run(debug=True )
