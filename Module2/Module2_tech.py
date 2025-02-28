# Data manipulation and analysis
import re
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

### Columns ###

column_list = ['TEMP_C_0_m', 'TEMP_C_1000_m', 'TEMP_C_5000_m', 
               'T_DEWPOINT_C_0_m', 'T_DEWPOINT_C_1000_m', 'T_DEWPOINT_C_5000_m', 
               'UGRD_m/s_0_m', 'UGRD_m/s_1000_m', 'UGRD_m/s_5000_m', 
               'VGRD_m/s_0_m', 'VGRD_m/s_1000_m', 'VGRD_m/s_5000_m', 
               'PRES_Pa_0_m', 'PRES_Pa_1000_m', 'PRES_Pa_5000_m']

def create_column_filter_widget(columns=column_list):
    """
    Create a simple grid of checkboxes for column selection.
    
    Parameters:
    columns (list): List of column names to include as selectable options.
    
    Returns:
    tuple: (widget, get_selected_columns) where get_selected_columns is a function
           that returns the current list of selected column names.
    """
    # Group columns by altitude level for section headers
    columns_by_altitude = {
        '0_m': [col for col in columns if '_0_m' in col],
        '1000_m': [col for col in columns if '_1000_m' in col],
        '5000_m': [col for col in columns if '_5000_m' in col]
    }
    
    # Create widgets for each section
    sections = []
    all_checkboxes = {}
    
    # Function to create a section of checkboxes
    def create_section(title, cols):
        # Create checkboxes for this section
        section_checkboxes = {
            col: widgets.Checkbox(
                value=False,
                description=col.replace('_0_m', '').replace('_1000_m', '').replace('_5000_m', ''),
                disabled=False,
                indent=False
            ) for col in cols
        }
        
        # Add these checkboxes to the global dictionary
        all_checkboxes.update(section_checkboxes)
        
        # Create a grid for this section
        grid = widgets.GridBox(
            children=list(section_checkboxes.values()),
            layout=widgets.Layout(
                grid_template_columns='repeat(3, auto)',
                grid_gap='5px',
                width='100%',
                padding='2px'
            )
        )
        
        # Return section widget
        return widgets.VBox([
            widgets.HTML(value=f"<b>{title}</b>"),
            grid
        ])
    
    # Create each section
    if columns_by_altitude['0_m']:
        sections.append(create_section("Surface Level (0m)", columns_by_altitude['0_m']))
    
    if columns_by_altitude['1000_m']:
        sections.append(create_section("1000m Level", columns_by_altitude['1000_m']))
    
    if columns_by_altitude['5000_m']:
        sections.append(create_section("5000m Level", columns_by_altitude['5000_m']))
    
    # Create output widget for selection summary
    output = widgets.Output()
    
    # Update function
    def update_display(change=None):
        selected_columns = [col for col, checkbox in all_checkboxes.items() if checkbox.value]
        
        with output:
            output.clear_output()
            if selected_columns:
                print(f"Selected {len(selected_columns)} of {len(columns)} columns:")
                
                # Group selected columns by altitude for display
                selected_by_altitude = {
                    "0m": [c for c in selected_columns if "_0_m" in c],
                    "1000m": [c for c in selected_columns if "_1000_m" in c],
                    "5000m": [c for c in selected_columns if "_5000_m" in c]
                }
                
                for level, cols in selected_by_altitude.items():
                    if cols:
                        print(f"  - {level}: {len(cols)} columns selected")
            else:
                print("No columns selected")
    
    # Connect event handlers
    for checkbox in all_checkboxes.values():
        checkbox.observe(update_display, names='value')
    
    # Create dividers between sections
    dividers = [widgets.HTML(value="<hr style='margin: 10px 0'>") for _ in range(len(sections)-1)]
    
    # Interleave sections and dividers
    container_items = []
    for i, section in enumerate(sections):
        container_items.append(section)
        if i < len(dividers):
            container_items.append(dividers[i])
    
    # Add output at the end
    container_items.append(widgets.HTML(value="<hr style='margin: 10px 0'>"))
    container_items.append(output)
    
    # Create main container
    main_widget = widgets.VBox(container_items)
    
    # Function to get selected columns
    def get_selected_columns():
        return [col for col, checkbox in all_checkboxes.items() if checkbox.value]
    
    # Initial display update
    update_display()
    
    return main_widget, get_selected_columns

class HistogramWidget:
    def __init__(self, df, bins=21, figsize=(7, 4)):
        """
        Initialize the histogram widget for plotting precipitation data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the data to plot
        bins : int, optional
            Number of bins for the histogram (default: 21)
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (7, 4))
        """
        self.df = df
        self.bins = bins
        self.figsize = figsize
        
        # Verify the dataframe has the 'ptype' column
        if 'ptype' not in df.columns:
            raise ValueError("Dataframe must contain a 'ptype' column with 'rain' and 'snow' values")
        
        # Get all columns except 'ptype' for dropdown
        self.numeric_columns = [col for col in df.columns if col != 'ptype' and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in dataframe besides 'ptype'")
        
        # Initialize widget components
        self.value_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.numeric_columns[0],
            description='Column:',
            disabled=False
        )
        
        # Separate opacity sliders for rain and snow
        self.rain_alpha_slider = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Rain opacity:',
            disabled=False,
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        self.snow_alpha_slider = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Snow opacity:',
            disabled=False,
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        self.plot_button = widgets.Button(
            description='Update Plot',
            disabled=False,
            button_style='', 
            tooltip='Click to update the plot'
        )
        
        # Set up the layout
        self.plot_output = widgets.Output()
        self.plot_button.on_click(self.update_plot)
        
        # Display the widget
        self.widget = widgets.VBox([
            self.value_dropdown,
            self.rain_alpha_slider,
            self.snow_alpha_slider,
            self.plot_button,
            self.plot_output
        ])
    
    def update_plot(self, b):
        """Update the histogram plot based on current widget values."""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Get current values from widgets
            value_column = self.value_dropdown.value
            rain_alpha = self.rain_alpha_slider.value
            snow_alpha = self.snow_alpha_slider.value
            
            # Split data by ptype and plot
            rain_data = self.df[self.df['ptype'] == 'rain'][value_column].dropna()
            snow_data = self.df[self.df['ptype'] == 'snow'][value_column].dropna()
            
            # Determine bin edges based on the full dataset
            all_data = self.df[value_column].dropna()
            bin_edges = np.histogram_bin_edges(all_data, bins=self.bins)
            
            # Plot snow histogram FIRST (on the bottom) with specific color #ee8866 and its own opacity
            if not snow_data.empty:
                ax.hist(snow_data, bins=bin_edges, alpha=snow_alpha, color='#ee8866', label='Snow')
            
            # Plot rain histogram SECOND (on top) with specific color #77aadd and its own opacity
            if not rain_data.empty:
                ax.hist(rain_data, bins=bin_edges, alpha=rain_alpha, color='#77aadd', label='Rain')
            
            # Add labels and legend
            ax.set_xlabel(value_column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {value_column} by Precipitation Type')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        # Generate initial plot
        self.update_plot(None)

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
            
            # Print class names information
            unique_classes = np.unique(y_train)
            print(f"Class names in y_train: {unique_classes} (Shape: {y_train.shape})")
            
            try:
                if selected_algo == "xgboost":
                    print("Running XGBoost model...")
                    # Convert class labels to 0 and 1 for XGBoost
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    
                    # Clear mapping showing which class name maps to which number
                    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
                    print(f"Label encoding: {class_mapping}")
                    print(f"Class '{le.classes_[0]}' → 0, Class '{le.classes_[1]}' → 1")
                    
                    selected_model = XGBClassifier(
                        n_estimators=100,
                        tree_method='hist',
                        random_state=42
                    )
                    # Store the label encoder with the model for later use
                    selected_model.fit(X_train_filtered, y_train_encoded)
                    selected_model.label_encoder_ = le
                    print("XGBoost model training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                elif selected_algo == "logistic_regression":
                    print("Running Logistic Regression model...")
                    # Increase max_iter to help with convergence
                    selected_model = LogisticRegression(max_iter=1000)
                    selected_model.fit(X_train_filtered, y_train)
                    
                    # Show how LogisticRegression has mapped the classes internally
                    classes_mapping = dict(zip(selected_model.classes_, range(len(selected_model.classes_))))
                    print(f"Logistic Regression class mapping: {classes_mapping}")
                    print(f"Class '{selected_model.classes_[0]}' → 0, Class '{selected_model.classes_[1]}' → 1")
                    
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
                
            # Reset button state to allow retraining
            train_button.description = 'Train Again'
            train_button.disabled = False

    train_button.on_click(train_model)

    display(widgets.VBox([train_button, status_label]))
    display(output)

    def get_model():
        """Function to get the trained model"""
        return selected_model

    return get_model  # Return the function instead of the model directly

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