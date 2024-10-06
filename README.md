# Hotel Booking Cancellation Prediction App

## Overview

This Flask application implements a logistic regression model to predict the likelihood of hotel booking cancellations based on various factors. The app provides an interactive interface for users to input their booking details and receive predictions regarding the probability of cancellation.

## Key Features

- **Logistic Regression Model**: Trained on historical booking data to learn patterns and make predictions about booking cancellations.
- **Data Preprocessing and Normalization**: Ensures data is prepared accurately for model training and predictions.
- **Interactive Form Interface**: Users can easily input booking details, such as lead time and average price, to receive predictions.
- **Model Evaluation**: Displays performance metrics like accuracy, confusion matrix, and classification report for transparency and reliability.
- **Clear Output Messages**: Provides users with intuitive feedback based on predicted cancellation probabilities.

## Dependencies

To run this application, ensure you have the following libraries installed:

- **Flask**: Web framework for building the application.
- **NumPy**: Library for numerical computing.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: Provides machine learning algorithms like Logistic Regression.
- **Joblib**: Facilitates model persistence and loading.

You can install these dependencies using pip:

```bash
pip install Flask numpy pandas scikit-learn joblib
```

## Model Training (Not Included in Flask App)

### Data Preprocessing

1. **Read the CSV Data File**: Load your dataset (`data.csv`) containing booking information.
2. **Drop Irrelevant Features**: Clean the dataset by removing non-informative columns.
3. **Normalize Features**: Normalize continuous features (`lead time` and `average price`) using `MinMaxScaler`.
4. **Handle Missing Values**: Clean the dataset to ensure no missing values affect model performance.
5. **Split Data**: Divide the prepared DataFrame into training and testing sets for model training and evaluation.

### Logistic Regression Training

1. **Sigmoid Function**: Implement the sigmoid function for hypothesis calculation.
2. **Cost Function**: Define the compute cost function to measure model error.
3. **Gradient Descent**: Implement the gradient descent function to iteratively minimize the cost function.
4. **Train Model**: Train the model using `train_logistic_regression` with appropriate hyperparameters (learning rate, iterations).

### Model Evaluation

1. **Make Predictions**: Use the trained model to predict outcomes on the testing set.
2. **Performance Metrics**: Calculate accuracy, confusion matrix, and classification report to assess model performance.
3. **Model Persistence**: Use `joblib.dump` to save the trained weights for faster deployment in the Flask app.

## Flask App Development

### App Initialization

1. **Create Flask Instance**: Initialize the Flask app using `Flask(__name__)`.

### Data Loading and Preprocessing

- Load preprocessed data using `pd.read_csv`.
- Access trained weights from `weights.pkl` using `joblib.load`.

### Routing

- **Main Page (`/`)**: Renders the `hotelform.html` template, which displays the user input form.
- **Prediction Page (`/predictpage`)**: Handles form submission using the POST method.

### Prediction Logic

1. **Extract User Input**: Convert form inputs to appropriate data types (int, float).
2. **Normalize Input Data**: Normalize `lead_time` and `average_price` using the pre-loaded scaler.
3. **Prepare Input Array**: Create an input array for prediction with the normalized values.
4. **Make Prediction**: Use the trained model to predict outcomes and calculate the probability.

### User Feedback

Based on the prediction, users receive feedback:
- **High Probability**: "You have a high probability of canceling your reservation."
- **Low Probability**: "That's the spirit! You wouldn't cancel your precious reservation."

## Running the App

1. **Save the Code**: Save the code as a Python file (e.g., `hotel_booking_cancellation_app.py`).
2. **Ensure Accessibility**: Ensure the preprocessed data and trained model weights are accessible.
3. **Run the Application**:
   - Open a terminal or command prompt and navigate to the directory where you saved the file.
   - Run the app using the command:
   ```bash
   python hotel_booking_cancellation_app.py
   ```
4. **Access the App**: Open a web browser and navigate to `http://localhost:5000/` (default Flask port).

## Further Considerations

- Fine-tune model hyperparameters for improved performance.
- Add error handling for invalid user inputs.
- Explore other machine learning algorithms for comparative analysis.
- Implement more comprehensive data visualizations to explore booking trends and cancellation factors.
- Consider deploying the app to a production environment for broader use.
