from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Dict, List, Tuple, Callable

from xgboost import XGBRegressor
import time 
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# Constants for configuration
WEATHER_VARS = [
    ('Temperature (F)', 'airtemp_degF'),
    ('Average Wind Speed (mph)', 'windspeed_mph'),
    ('Wind Gust (mph)', 'windgust_mph'),
    ('Relative Humidity (%)', 'rh_percent'),
    ('Precipitation (in)', 'precip_in')
]

STATIONS = ['BEAR', 'BURN', 'FRYI', 'JEFF', 'NCAT', 'SALI', 'SASS', 'UNCA', 'WINE']

def create_ml_quiz():
    """
    Creates an interactive quiz about ML concepts with two buttons and feedback.
    Returns the display elements for use in a Jupyter notebook.
    """
    output = widgets.Output()
    
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
    
    return buttons, output

def display_quiz():
    """Creates and displays the quiz in the notebook."""
    buttons, output = create_ml_quiz()
    display(buttons, output)

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

def create_input_station_controls():
    """Creates control widgets for input stations data visualization."""
    var_dropdown = widgets.Dropdown(
        options=WEATHER_VARS,
        description='Variable:',
        disabled=False
    )

    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Time Series'],
        description='Plot type:',
        disabled=False
    )

    station_dropdown = widgets.Dropdown(
        options=STATIONS,
        description='Station:',
        disabled=False
    )

    plot_button = widgets.Button(description="Plot")
    output = widgets.Output()
    
    return station_dropdown, var_dropdown, plot_dropdown, plot_button, output

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

def create_correlation_plot_controls():
    """Creates control widgets for correlation plots."""
    var_dropdown = widgets.Dropdown(
        options=[
            ('Temperature (F)', 'airtemp_degF'),
            ('Precipitation (in)', 'precip_in'),
            ('Relative Humidity (%)', 'rh_percent'),
            ('Wind Gust (mph)', 'windgust_mph'),
            ('Average Wind Speed (mph)', 'windspeed_mph')
        ],
        description='Variable:',
        disabled=False
    )

    plot_button = widgets.Button(description="Plot")
    output = widgets.Output()
    
    return var_dropdown, plot_button, output

def display_correlation_plot_dashboard(base_url="https://elearning.unidata.ucar.edu/dataeLearning/Cybertraining/analysis/media/pairplot_"):
    """Creates and displays interactive dashboard for correlation plots."""
    # Create interface controls
    var_dropdown, plot_button, output = create_correlation_plot_controls()
    
    def update_image(_):
        selected_var = var_dropdown.value
        image_url = f"{base_url}{selected_var}.png"
        
        with output:
            output.clear_output(wait=True)
            display(HTML(
                f'<center><i>Click to enlarge</i><br>'
                f'<a href="{image_url}" target="blank">'
                f'<img src="{image_url}" width="600px"></a></center>'
            ))
    
    plot_button.on_click(update_image)
    display(var_dropdown, plot_button, output)

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
        "Multi-Linear Regressor": "linear_regression",
        "XGBoost": "xgboost"
    }
    
    # Use a list to store the selection (mutable)
    selection = [None]
    output = widgets.Output()

    buttons = [
        widgets.Button(
            description=name,
            layout=widgets.Layout(width='200px', height='200px', margin='10px')
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

def train_model_button(selected_algo, X_train_filtered, y_train):  # selected_algo is now a parameter
    """Creates a single 'Train ML Model' button, using the provided selected_algo."""
    global selected_model
    output = widgets.Output()

    train_button = widgets.Button(description='Train Algorithm', layout=widgets.Layout(width='200px', height='200px'))

    def train_model(b):
        global selected_model
        with output:
            clear_output()
            if selected_algo == "xgboost":
                print("Running XGBoost model...")
                base_model = XGBRegressor(
                    n_estimators=100,
                    tree_method='hist',
                    random_state=42
                )
                selected_model = MultiXGBRegressor(base_model) # Assuming this is defined
                selected_model.fit(X_train_filtered, y_train) # Make sure these variables are available
                print("XGBoost model training completed!")
            elif selected_algo == "linear_regression":
                print("Running Linear Regression model...")
                selected_model = MultiLinearRegressor() # Assuming this is defined
                selected_model.fit(X_train_filtered, y_train) # Make sure these variables are available
                print("Linear Regression training completed!")
            else:
                print("No algorithm selected. Cannot train.")

    train_button.on_click(train_model)

    display(train_button)
    display(output)

    return selected_model  # Return the trained model (or None if not trained)

class MultiXGBRegressor(MultiOutputRegressor):
    def __init__(self, estimator):
        super().__init__(estimator)
        self.estimators_ = []

    def fit(self, X, y):
        start_time = time.time()
        print("\nStarting Multi-Target XGBoost Training Process...")
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        n_outputs = y_np.shape[1]
        target_names = y.columns if hasattr(y, 'columns') else [f"target_{i}" for i in range(n_outputs)]
        
        self.estimators_ = [
            XGBRegressor(**{k: v for k, v in self.estimator.get_params().items() 
                          if k != 'verbose'}) 
            for _ in range(n_outputs)
        ]
        
        for i, (est, target) in enumerate(zip(self.estimators_, target_names)):
            target_start = time.time()
            print(f"\nTraining target {i+1}/{n_outputs}: {target}", flush=True)
            est.fit(X, y_np[:, i], verbose=False)
            target_time = time.time() - target_start
            print(f"Target completed in {target_time:.2f} seconds", flush=True)

        total_time = time.time() - start_time
        print(f"\nTotal training completed in {total_time:.2f} seconds")
        return self
    

class MultiLinearRegressor(MultiOutputRegressor):
    def __init__(self):
        super().__init__(LinearRegression())
        self.estimators_ = []

    def fit(self, X, y):
        start_time = time.time()
        print("\nStarting Multi-Target Linear Regression Training...")
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        n_outputs = y_np.shape[1]
        target_names = y.columns if hasattr(y, 'columns') else [f"target_{i}" for i in range(n_outputs)]
        
        self.estimators_ = [LinearRegression() for _ in range(n_outputs)]
        
        for i, (est, target) in enumerate(zip(self.estimators_, target_names)):
            target_start = time.time()
            print(f"\nTraining target {i+1}/{n_outputs}: {target}", flush=True)
            est.fit(X, y_np[:, i])
            target_time = time.time() - target_start
            print(f"Target completed in {target_time:.2f} seconds", flush=True)

        total_time = time.time() - start_time
        print(f"\nTotal training completed in {total_time:.2f} seconds")
        return self