# Data manipulation and analysis
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Machine learning
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Utilities
import time
from typing import Any

def create_ml_knowledgecheck():
    """
    Creates an interactive knowledge check about ML concepts with two buttons and feedback.
    Returns the display elements for use in a Jupyter notebook.
    """
    output = widgets.Output()

    question = widgets.HTML(
        "Which type of machine learning analysis is most appropriate for this scenario?"
    )
    
    # Create buttons
    classification_button = widgets.Button(
        description='Classification',
        layout=widgets.Layout(width='200px', height='200px', margin='10px')
    )
    
    regression_button = widgets.Button(
        description='Regression',
        layout=widgets.Layout(width='200px', height='200px', margin='10px')
    )
    
    def show_feedback(is_correct):
        """Helper function to display feedback"""
        with output:
            clear_output(wait=True)
            if is_correct:
                display(HTML("""
                    <div class="alert alert-info" role="feedback">
                        <p class="admonition-title" style="font-weight:bold">Correct</p>
                        <p>This scenario requires a numerical output, so we will use a regression algorithm for this scenario.</p>
                    </div>
                """))
            else:
                display(HTML("""
                    <div class="alert alert-info" role="feedback">
                        <p class="admonition-title" style="font-weight:bold">Incorrect</p>
                        <p>Classification tasks work for scenarios that require classifying data into categories. 
                        This task needs a <i>numerical value </i>for output, and therefore requires a different approach.</p>
                    </div>
                """))
    
    # Define click handlers
    classification_button.on_click(lambda b: show_feedback(False))
    regression_button.on_click(lambda b: show_feedback(True))
    
    # Create button container
    buttons = widgets.HBox([classification_button, regression_button])
    
    return question, buttons, output

def display_knowledgecheck():
    """Creates and displays the knowledge check in the notebook."""
    question, buttons, output = create_ml_knowledgecheck()
    display(question, buttons, output)

def create_weather_visualization_controls():
    """Creates control widgets for the Mt. Mitchell weather data visualization."""
    # Variable dropdown
    var_dropdown = widgets.Dropdown(
        options=[
            ('Temperature (F)', 'MITC_airtemp_degF'),
            ('Average Wind Speed (mph)', 'MITC_windspeed_mph'),
            ('Wind Gust (mph)', 'MITC_windgust_mph'),
            ('Relative Humidity (%)', 'MITC_rh_percent'),
            ('Precipitation (in)', 'MITC_precip_in')
        ],
        description='Variable:',
        disabled=False
    )

    # Plot type dropdown
    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Time Series'],
        description='Plot type:',
        disabled=False
    )

    # Button for plotting
    plot_button = widgets.Button(description="Plot")
    
    # Output widget to render plots
    output = widgets.Output()
    
    return var_dropdown, plot_dropdown, plot_button, output

def display_mt_mitchell_weather_dashboard(weather_data):
    """Creates and displays interactive dashboard for Mt. Mitchell weather data."""
    # Create interface controls
    var_dropdown, plot_dropdown, plot_button, output = create_weather_visualization_controls()
    
    def on_plot_button_click(b):
        with output:
            clear_output(wait=True)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.hist(weather_data[var_dropdown.value], bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {var_dropdown.label} at Mt. Mitchell (MITC)", fontsize=14)
                ax.set_xlabel(var_dropdown.label)
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Time Series':
                xdates = pd.to_datetime(weather_data['observation_datetime'])
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.plot(xdates[::100], weather_data[var_dropdown.value][::100], 
                       label=var_dropdown.label, color='orange')
                ax.set_title(f"Time Series of {var_dropdown.label} at Mt. Mitchell (MITC)", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel(var_dropdown.label)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Mt. Mitchell</h3>"), 
           var_dropdown, 
           plot_dropdown, 
           plot_button, 
           output)

def display_input_stations_dashboard(weather_data):
    """Creates and displays interactive dashboard for input stations weather data."""
    # Create interface controls
    station_dropdown, var_dropdown, plot_dropdown, plot_button, output = create_input_station_controls()
    
    def on_plot_button_click(b):
        # Construct variable name by combining station and weather variable
        selected_var = f"{station_dropdown.value}_{var_dropdown.value}"
        
        with output:
            clear_output(wait=True)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.hist(weather_data[selected_var], bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {var_dropdown.label} at {station_dropdown.value}", fontsize=14)
                ax.set_xlabel(var_dropdown.label)
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Time Series':
                xdates = pd.to_datetime(weather_data['observation_datetime'])
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.plot(xdates[::100], weather_data[selected_var][::100], 
                       label=var_dropdown.label, color='orange')
                ax.set_title(f"Time Series of {var_dropdown.label} at {station_dropdown.value}", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel(var_dropdown.label)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Input Stations</h3>"), 
           station_dropdown,
           var_dropdown, 
           plot_dropdown, 
           plot_button, 
           output)

def create_percentage_widget():
    """Creates widget for specifying training/validation/testing splits."""
    # Create text widgets for percentages
    training = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Training %:',
        layout=widgets.Layout(width='200px')
    )

    validation = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Validation %:',
        layout=widgets.Layout(width='200px')
    )

    testing = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Testing %:',
        layout=widgets.Layout(width='200px')
    )

    submit_button = widgets.Button(description="Submit")
    output = widgets.Output()
    
    def check_percentages(change=None):
        with output:
            output.clear_output()
            total = training.value + validation.value + testing.value
            print(f"Total: {total}%")
                
    def on_submit_clicked(b):
        with output:
            output.clear_output()
            check_percentages()
            total = training.value + validation.value + testing.value
            print("✓ Submitted" if total == 100 else 
                  "⚠️ Make sure the percentages sum to 100% and resubmit.")
    
    # Add observers
    training.observe(check_percentages, names='value')
    validation.observe(check_percentages, names='value')
    testing.observe(check_percentages, names='value')
    submit_button.on_click(on_submit_clicked)
    
    # Layout
    widget_box = widgets.VBox([
        widgets.HTML(value="<h3>Dataset Split Percentages</h3>"),
        training,
        validation,
        testing,
        output,
        submit_button
    ])
    
    display(widget_box)
    
    def get_decimal_values():
        return {
            'training': training.value / 100,
            'validation': validation.value / 100,
            'testing': testing.value / 100
        }
    
    return widget_box, get_decimal_values

def algorithm_selection():
    """Creates widget for algorithm selection and returns selected value via callback."""
    algorithm_options = {
        "Multivariate Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost"
    }
    # Use a list to store the selection (mutable)
    selection = [None]
    output = widgets.Output()

    buttons = [
        widgets.Button(
            description=name,
            layout=widgets.Layout(width='300px', height='125px', margin='10px')
        )
        for name in algorithm_options
    ]

    def on_button_clicked(b):
        selection[0] = algorithm_options[b.description]
        for button in buttons:
            button.style.button_color = '#b2ebf2' if button == b else None
        with output:
            clear_output(wait=True)
            print(f"Selected Algorithm: {b.description}")

    for button in buttons:
        button.on_click(on_button_clicked)

    display(widgets.HBox(buttons), output)
    
    # Function to get current selection
    def get_selection():
        return selection[0]
        
    return get_selection

def create_station_selector():
    """Creates grid of checkboxes for station selection."""
    checkboxes = {
        station: widgets.Checkbox(
            value=False,
            description=station,
            disabled=False,
            indent=False
        ) 
        for station in STATIONS
    }
    
    checkbox_grid = widgets.GridBox(
        children=[checkboxes[station] for station in STATIONS],
        layout=widgets.Layout(
            grid_template_columns='repeat(3, auto)',
            grid_gap='10px'
        )
    )
    
    output = widgets.Output()
    
    def on_change(change):
        with output:
            output.clear_output()
            selected = [station for station, checkbox in checkboxes.items() if checkbox.value]
            print(f"Selected stations: {', '.join(selected) if selected else 'None'}")
    
    for checkbox in checkboxes.items():
        checkbox[1].observe(on_change, names='value')
    
    display(widgets.VBox([
        widgets.HTML(value="<h3>Select Weather Stations</h3>"),
        checkbox_grid,
        output
    ]))
    
    return checkboxes

selected_model = None  # Global variable for model access

def train_button(selected_algo, X_train_filtered, y_train):
    """Creates a single 'Train ML Model' button, using the provided selected_algo."""
    global selected_model
    selected_model = None  # Reset the model at start
    output = widgets.Output()

    # Create a label to show model status
    status_label = widgets.Label(value='Click button to train model')
    
    train_button = widgets.Button(
        description='Train Algorithm', 
        layout=widgets.Layout(width='200px', height='200px')
    )

    def train_model(b):
        global selected_model
        with output:
            clear_output()
            # Use warning filter to suppress convergence warnings
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            
            # Filter out the specific convergence warning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            try:
                if selected_algo == "xgboost":
                    print("Running XGBoost model...")
                    selected_model = XGBClassifier(
                        n_estimators=100,
                        tree_method='hist',
                        random_state=42
                    )
                    selected_model.fit(X_train_filtered, y_train)
                    print("XGBoost model training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                elif selected_algo == "logistic_regression":
                    print("Running Logistic Regression model...")
                    # Increase max_iter to help with convergence
                    selected_model = LogisticRegression(max_iter=1000)
                    selected_model.fit(X_train_filtered, y_train)
                    print("Logistic Regression training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                else:
                    print("No algorithm selected. Cannot train.")
                    selected_model = None
                    status_label.value = 'No algorithm selected'
                    
            except Exception as e:
                print(f"Error during training: {str(e)}")
                selected_model = None
                status_label.value = 'Error during training'
            
            # Print final status of selected_model
            print("\nFinal status:")
            print(f"selected_model object exists: {selected_model is not None}")
            if selected_model is not None:
                print(f"Model type: {type(selected_model)}")
                
            # Update button state
            train_button.description = 'Model Trained'
            train_button.disabled = True

    train_button.on_click(train_model)

    display(widgets.VBox([train_button, status_label]))
    display(output)

    def get_model():
        """Function to get the trained model"""
        return selected_model

    return get_model  # Return the function instead of the model directly

## Data

def train_val_test_split(df, 
                         y_col='ptype', 
                         train_size=0.7, 
                         val_size=0.15, 
                         test_size=0.15, 
                         random_state=None):
    """
    Split a pandas DataFrame into features and target sets for training, validation, and testing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to split.
    y_col : str, default='ptype'
        Column name to use as the target variable.
    train_size : float, default=0.7
        Proportion of the dataset to include in the training split.
    val_size : float, default=0.15
        Proportion of the dataset to include in the validation split.
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Features for training.
    y_train : pandas.Series
        Target for training.
    X_val : pandas.DataFrame
        Features for validation.
    y_val : pandas.Series
        Target for validation.
    X_test : pandas.DataFrame
        Features for testing.
    y_test : pandas.Series
        Target for testing.
    """
    # Verify that the proportions sum to 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("train_size, val_size, and test_size should sum to 1.0")
    
    # First, split into training and temp (validation + test)
    relative_test_size = test_size / (val_size + test_size)
    
    df_train, df_temp = train_test_split(
        df, 
        train_size=train_size,
        test_size=val_size + test_size,
        random_state=random_state
    )
    
    # Then split the temp set into validation and test
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_size,
        random_state=random_state
    )
    
    # Split each dataframe into X and y
    X_train = df_train.drop(columns=[y_col])
    y_train = df_train[y_col]
    
    X_val = df_val.drop(columns=[y_col])
    y_val = df_val[y_col]
    
    X_test = df_test.drop(columns=[y_col])
    y_test = df_test[y_col]
    
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df):.1%})")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df):.1%})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test