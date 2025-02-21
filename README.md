# Bike-Rental-Prediction

🚲 Bike Rental Prediction using Deep Learning & Streamlit

This project predicts the number of rented bikes in Seoul based on weather and temporal features using a deep learning model. The application is built using PyTorch and Streamlit for interactive visualization.

📌 Features

🧠 Deep Learning Model: A fully connected neural network (FCNN) trained on SeoulBikeData.

📊 Feature Engineering: Includes cyclical transformations for time-based features and one-hot encoding for categorical variables.

🏛️ User Input Interface: Users can select hour, month, season, and holiday status for rental predictions.

📉 Visualization: Streamlit generates bar charts comparing predicted rentals across different seasons.

📌 Train the Model

If you want to retrain the model, run:

python model_train.py

This will:

Preprocess the SeoulBikeData.csv file.

Train a deep learning model using PyTorch.

Save the trained model in saved_models/.

📌 Run the Streamlit App

Launch the Streamlit web app with:

streamlit run bike_rental_app.py

This will open an interactive web app where users can input time, season, and weather conditions to predict bike rentals.


📈 Technologies Used

Python 🐍

PyTorch 🔥

Streamlit 🌐

Pandas & NumPy 📊

Scikit-learn 🏆

📄 Author

Lachiket Narendra Warule🔗 LinkedIn

📝 License

This project is open-source under the MIT License.

