from django.shortcuts import render
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'index.html')

def result(request):
    return render(request, 'predict.html')

file_path = 'C:/Users/diopm/OneDrive/Bureau/DA_PJ/diabetes.csv'
data = pd.read_csv(file_path)

columns_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in columns_with_zeroes:
    median = data[column].median()
    data[column] = data[column].replace(0, median)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

def predict(request):
    if request.method == 'GET':
        n1 = float(request.GET.get('n1', 0))
        n2 = float(request.GET.get('n2', 0))
        n3 = float(request.GET.get('n3', 0))
        n4 = float(request.GET.get('n4', 0))
        n5 = float(request.GET.get('n5', 0))
        n6 = float(request.GET.get('n6', 0))
        n7 = float(request.GET.get('n7', 0))
        n8 = float(request.GET.get('n8', 0))

        new_data = {
            'Pregnancies': [n1],
            'Glucose': [n2],
            'BloodPressure': [n3],
            'SkinThickness': [n4],
            'Insulin': [n5],
            'BMI': [n6],
            'DiabetesPedigreeFunction': [n7],
            'Age': [n8]
        }

        new_data_df = pd.DataFrame(new_data)

        prediction = model.predict(new_data_df)[0]

        probabilities = model.predict_proba(new_data_df)[0] 

        if prediction == 1:
            result = 'Positive (Risque de diabète)'
            probability = probabilities[1]  
        else:
            result = 'Negative (Pas de risque de diabète)'
            probability = probabilities[0]  

        return render(request, 'predict.html', {'result': result, 'probability': probability * 100})

    return render(request, 'error.html', {'error': 'Méthode non autorisée'})
