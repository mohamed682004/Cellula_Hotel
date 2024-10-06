# Hotel Booking Cancellation Prediction App

## Overview

This Flask application implements a logistic regression model to predict the likelihood of hotel booking cancellations based on various factors. Additionally, it features an interactive dashboard for visualizing hotel booking data, enabling users to explore trends and insights related to guest behaviour.

## Key Features

- **Logistic Regression Model**: Trained on historical booking data to learn patterns and make predictions about booking cancellations.
- **Data Preprocessing and Normalization**: Ensures data is prepared accurately for model training and predictions.
- **Interactive Form Interface**: Allows users to input booking details and receive predictions regarding the probability of cancellation.
- **Model Evaluation**: Displays performance metrics like accuracy, confusion matrix, and classification report for transparency and reliability.
- **Interactive Dashboard**: Visualizes booking data through various charts, allowing users to explore relationships between different features (e.g., guest demographics, room types, market segments).
- **Clear Output Messages**: Provides users with intuitive feedback based on predicted cancellation probabilities.

## Dependencies

To run this application, ensure you have the following libraries installed:

- **Flask**: Web framework for building the application.
- **NumPy**: Library for numerical computing.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: Provides machine learning algorithms like Logistic Regression.
- **Joblib**: Facilitates model persistence and loading.
- **Dash**: For building the interactive dashboard.
- **Plotly**: For creating visualizations within the dashboard.

You can install these dependencies using pip:

```bash
pip install Flask numpy pandas scikit-learn joblib dash plotly
```

## Hotel Booking Dashboard

This code creates an interactive dashboard to visualize hotel booking data using Dash, a Python framework for building web applications.

### Data Preparation

1. **Library Imports**: The code imports necessary libraries, including:
   - `pandas` for data manipulation,
   - `plotly.express` and `plotly.graph_objs` for creating visualizations,
   - `Dash` for building the web application.
   
2. **Data Loading**: Loads the CSV data file containing hotel booking information.

3. **Preprocessing**: Cleans and formats the data, including:
   - Converting date formats,
   - Handling missing values,
   - Transforming categorical features into numerical representations.

### Interactive Dashboard Layout

- **App Structure**: The `app.layout` defines the structure of the dashboard, which includes:
  - A title: "Hotel Booking Dashboard".
  - Two rows allowing users to filter data:
    - **DatePickerRange**: For selecting a date range for reservations.
    - **RangeSlider**: For filtering based on the number of weekend nights desired.

- **Interactive Charts**: Four separate `dcc.Graph` components display interactive charts:
  - **Adults and Children Chart**: Shows the total number of adults and children for the selected date range and weekend night filter.
  - **Room Types Chart**: A bar chart depicting the distribution of booked room types for the filtered data.
  - **Market Segment Chart**: A pie chart visualizing the distribution of bookings across different market segments (Offline, Online, Corporate, etc.).
  - **Booking Status Chart**: Displays a pie chart showing the percentage of canceled and non-canceled bookings within the filtered data.

### Data Filtering and Chart Updates

- **Callback Function**: The `@app.callback` decorator defines a function that updates the charts based on user interactions. It takes input from:
  - The selected date range (start_date and end_date) from the DatePickerRange.
  - The chosen weekend night range (weekend_nights) from the RangeSlider.
  
- **Data Filtering**: The function filters the original data frame (`df`) based on the selected criteria, creating a new `filtered_df`.

- **Chart Generation**: It generates figures for each chart based on the filtered data:
  - Adults and Children Chart: A bar chart representing the total number of adults and children.
  - Room Types Chart: A bar chart showing the distribution of room types.
  - Market Segment Chart: A pie chart depicting market segment distribution.
  - Booking Status Chart: A pie chart visualizing the percentage of canceled and non-canceled bookings.

- **Return Updated Figures**: The function returns a tuple containing the updated figures for all four charts.

### Running the Application

1. **Execution**: The `if __name__ == '__main__':` block executes the code when the script is run directly. It starts the Dash app server in debug mode (`debug=True`) on port 8052.
  
2. **Accessing the Dashboard**:
   - Save the code as a Python file (e.g., `hotel_booking_dashboard.py`).
   - Open a terminal or command prompt and navigate to the directory where you saved the file.
   - Run the script using the command:
     ```bash
     python hotel_booking_dashboard.py
     ```
   - This will launch the web application in your default browser, typically at `http://localhost:8052/`.

### Interacting with the Dashboard

- **User Input**: Use the date picker to select a desired reservation date range.
- **Adjust Filters**: Adjust the range slider to filter bookings based on the number of weekend nights.
- **Dynamic Updates**: Observe how the charts update dynamically to reflect the chosen filters.

This interactive dashboard provides a user-friendly way to explore hotel booking data and gain insights into guest behaviour, room type preferences, market segment performance, and booking trends.

## Running the Flask App

1. **Save the Code**: Save the Flask app code as a Python file (e.g., `main.py`).
2. **Ensure Accessibility**: Ensure the preprocessed data and trained model weights are accessible.
3. **Run the Application**:
   - Open a terminal or command prompt and navigate to the directory where you saved the file.
   - Run the app using the command:
     ```bash
     python main.py
     ```
4. **Access the App**: Open a web browser and navigate to `http://localhost:5000/` (default Flask port).

## Further Considerations

- Fine-tune model hyperparameters for improved performance.
- Add error handling for invalid user inputs.
- Explore other machine learning algorithms for comparative analysis.
- Implement more comprehensive data visualizations to explore booking trends and cancellation factors.
- Consider deploying the app to a production environment for broader use.

---

This README now fully incorporates the details of the interactive dashboard and how it fits into the overall hotel booking cancellation prediction application. Let me know if you need any additional modifications!
