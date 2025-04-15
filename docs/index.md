# SBYB Documentation

Welcome to the documentation for SBYB (Step-By-Your-Byte), a comprehensive machine learning library that unifies the entire ML pipeline.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [CLI Reference](#cli-reference)
7. [Tutorials](#tutorials)
8. [Examples](#examples)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

SBYB (Step-By-Your-Byte) is a comprehensive machine learning library designed to unify the entire ML pipeline from data preprocessing to model deployment. It provides a seamless experience for data scientists and ML engineers, with intelligent defaults and automated processes that make machine learning more accessible and efficient.

### Key Features

- **Unified Data Preprocessing**: Automatic handling of missing values, outliers, encoding, and scaling
- **Task Type & Data Type Auto-detection**: Intelligent identification of ML tasks and data characteristics
- **AutoML Engine**: Automated model selection, hyperparameter tuning, and ensemble creation
- **Evaluation & Explainability**: Comprehensive metrics and model interpretation tools
- **Deployment & Serving**: Easy model export and deployment options
- **Zero-code UI Generation**: Automatic creation of user interfaces for models
- **Project Scaffolding**: Quick setup of new ML projects with best practices
- **EDA Tools**: Powerful data profiling and visualization capabilities
- **Plugin System**: Extensible architecture for custom components
- **Local Experiment Tracking**: Track, compare, and visualize ML experiments
- **CLI & Programmatic API**: Multiple interfaces for different workflows

### Why SBYB?

SBYB stands out from other ML libraries by:

1. **Unifying the Entire Pipeline**: From data preprocessing to deployment in a single library
2. **Working Offline**: No internet connection required for core functionality
3. **Providing Intelligent Defaults**: Sensible choices that work well for most datasets
4. **Supporting Multiple Interfaces**: CLI, API, and UI options for different users
5. **Being Highly Extensible**: Plugin system for custom components and integrations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install sbyb
```

### Install from Source

```bash
git clone https://github.com/sbyb/sbyb.git
cd sbyb
pip install -e .
```

## Quick Start

### Using the CLI

```bash
# Create a new project
sbyb project create --name my_project --template classification

# Run AutoML on a dataset
sbyb automl run --data data.csv --target target_column

# Generate a UI for a model
sbyb ui generate --model model.pkl --output ui_app
```

### Using the API

```python
from sbyb.api import SBYB

# Initialize SBYB
sbyb = SBYB()

# Preprocess data
import pandas as pd
data = pd.read_csv("data.csv")
preprocessed_data = sbyb.preprocess_data(data)

# Run AutoML
result = sbyb.run_automl(
    data=preprocessed_data,
    target="target_column",
    output_dir="output"
)

# Generate UI
sbyb.generate_ui(
    model=result.model,
    output_dir="ui_app",
    ui_type="dashboard",
    framework="streamlit"
)
```

## Core Components

SBYB consists of several core components that work together to provide a comprehensive ML solution:

### Data Preprocessing

The preprocessing module handles all aspects of data preparation, including:

- Missing value detection and imputation
- Outlier detection and handling
- Categorical encoding
- Feature scaling
- Feature engineering
- Preprocessing pipelines

```python
from sbyb.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()
preprocessed_data = pipeline.fit_transform(data)
```

### Task Detection

The task detection module automatically identifies the appropriate machine learning task based on data characteristics:

- Classification detection
- Regression detection
- Clustering detection
- NLP task detection
- Computer vision detection
- Time series detection

```python
from sbyb.task_detection import TaskDetector

detector = TaskDetector()
task_type = detector.detect(X, y)
```

### AutoML Engine

The AutoML engine automates the model selection and optimization process:

- Model selection based on task type
- Hyperparameter optimization
- Feature selection
- Model stacking and ensembling

```python
from sbyb.automl import AutoMLEngine

automl = AutoMLEngine(task_type="classification")
result = automl.fit(X, y)
```

### Evaluation & Explainability

The evaluation module provides comprehensive metrics and model interpretation tools:

- Task-specific metrics calculation
- Visualization of model performance
- Model explainability with SHAP, LIME, and ELI5

```python
from sbyb.evaluation import Evaluator, Explainer

evaluator = Evaluator()
evaluation = evaluator.evaluate(model, X_test, y_test)

explainer = Explainer()
explanation = explainer.explain(model, X_test)
```

### Deployment & Serving

The deployment module enables easy model export and serving:

- Model export to various formats
- API generation for model serving
- Containerization and orchestration
- Model monitoring

```python
from sbyb.deployment import ModelExporter, ModelServer

exporter = ModelExporter()
exporter.to_onnx(model, "model.onnx")

server = ModelServer(server_type="fastapi")
server.serve(model, host="0.0.0.0", port=8000)
```

### UI Generator

The UI generator creates interactive interfaces for models:

- Dashboard generation
- Form generation
- Component library
- Theming system

```python
from sbyb.ui_generator import DashboardGenerator

generator = DashboardGenerator(framework="streamlit")
generator.generate(model, output_dir="dashboard")
```

### Project Scaffolding

The scaffolding module helps set up new ML projects:

- Project templates
- Configuration generation
- Environment setup

```python
from sbyb.scaffolding import ProjectGenerator

generator = ProjectGenerator()
generator.create_project(name="my_project", template="classification")
```

### EDA Tools

The EDA module provides powerful data profiling and visualization:

- Data profiling
- Visualization
- Statistical analysis

```python
from sbyb.eda import DataProfiler, Visualizer

profiler = DataProfiler()
profile = profiler.generate_profile(data)

visualizer = Visualizer()
visualizer.plot_correlation_matrix(data)
```

### Plugin System

The plugin system enables extension of SBYB's functionality:

- Plugin discovery and loading
- Hook system
- Custom component registration

```python
from sbyb.plugins import PluginManager

manager = PluginManager()
manager.install_plugin("sbyb-plugin-example")
```

### Experiment Tracking

The tracking module allows tracking and comparison of ML experiments:

- Experiment and run management
- Metric and parameter logging
- Artifact storage
- Visualization

```python
from sbyb.tracking import ExperimentTracker

tracker = ExperimentTracker()
tracker.create_experiment(name="My Experiment")
tracker.create_run()
tracker.log_metric("accuracy", 0.95)
```

## API Reference

### SBYB Class

The main interface for the SBYB library.

```python
from sbyb.api import SBYB

sbyb = SBYB(config=None)
```

#### Project Management

```python
# Create a new project
project_dir = sbyb.create_project(
    name="my_project",
    template="classification",
    output_dir=".",
    description="My ML project"
)
```

#### Data Preprocessing

```python
# Preprocess data
preprocessed_data = sbyb.preprocess_data(
    data=data,
    config=None,
    output_file="preprocessed.csv",
    save_pipeline="pipeline.pkl"
)

# Generate data profile
profile = sbyb.profile_data(
    data=data,
    output_dir="profile",
    format="html"
)
```

#### AutoML

```python
# Run AutoML
result = sbyb.run_automl(
    data=data,
    target="target_column",
    task=None,  # Auto-detected if None
    config=None,
    output_dir="output",
    time_limit=3600,
    track=True,
    experiment_name="My Experiment"
)
```

#### Evaluation

```python
# Evaluate model
evaluation = sbyb.evaluate_model(
    model=model,
    data=test_data,
    target="target_column",
    metrics=["accuracy", "f1"],
    output_dir="evaluation"
)

# Explain model
explanation = sbyb.explain_model(
    model=model,
    data=data,
    method="shap",
    output_dir="explanation"
)
```

#### Deployment

```python
# Export model
path = sbyb.export_model(
    model=model,
    format="onnx",
    output="model.onnx"
)

# Serve model
server = sbyb.serve_model(
    model=model,
    host="0.0.0.0",
    port=8000,
    server_type="fastapi"
)
server.start()
```

#### UI Generation

```python
# Generate UI
ui_path = sbyb.generate_ui(
    model=model,
    output_dir="ui",
    ui_type="dashboard",
    framework="streamlit",
    theme="light"
)
```

#### Plugin Management

```python
# List plugins
plugins = sbyb.list_plugins(category=None)

# Install plugin
success = sbyb.install_plugin(
    source="sbyb-plugin-example",
    force=False
)

# Uninstall plugin
success = sbyb.uninstall_plugin(
    name="sbyb-plugin-example",
    category=None
)

# Create plugin template
success = sbyb.create_plugin_template(
    name="my-plugin",
    output_dir=".",
    category="custom",
    description="My custom plugin",
    author="John Doe"
)
```

#### Experiment Tracking

```python
# Create experiment
experiment = sbyb.create_experiment(
    name="My Experiment",
    description="Experiment description",
    tags=["tag1", "tag2"]
)

# Create run
run = sbyb.create_run(
    experiment_id=experiment.experiment_id,
    name="Run 1",
    description="Run description",
    tags=["tag1"]
)

# Start run
sbyb.start_run()

# Log metrics and parameters
sbyb.log_metric("accuracy", 0.95)
sbyb.log_parameter("learning_rate", 0.01)

# Log model
model_path = sbyb.log_model(model, "model.pkl")

# End run
sbyb.end_run()

# Visualize experiment
report_path = sbyb.visualize_experiment(
    experiment_id=experiment.experiment_id,
    metrics=["accuracy", "f1"],
    params=["learning_rate", "max_depth"],
    output_dir="visualization",
    format="html"
)
```

### Component APIs

Each component of SBYB also has its own API that can be used directly:

- `sbyb.preprocessing`: Data preprocessing components
- `sbyb.task_detection`: Task detection components
- `sbyb.automl`: AutoML components
- `sbyb.evaluation`: Evaluation and explainability components
- `sbyb.deployment`: Deployment and serving components
- `sbyb.ui_generator`: UI generation components
- `sbyb.scaffolding`: Project scaffolding components
- `sbyb.eda`: EDA components
- `sbyb.plugins`: Plugin system components
- `sbyb.tracking`: Experiment tracking components

## CLI Reference

SBYB provides a comprehensive command-line interface for all major functionality.

### Global Options

```
--help, -h: Show help message
```

### Project Commands

```bash
# Create a new project
sbyb project create --name my_project --template classification --output . --description "My ML project"
```

### Data Commands

```bash
# Preprocess data
sbyb data preprocess --data data.csv --config config.json --output preprocessed.csv --save-pipeline pipeline.pkl

# Generate data profile
sbyb data profile --data data.csv --output profile --format html
```

### AutoML Commands

```bash
# Run AutoML
sbyb automl run --data data.csv --target target_column --task classification --config config.json --output output --time-limit 3600 --track --experiment "My Experiment"
```

### Evaluation Commands

```bash
# Evaluate model
sbyb eval run --model model.pkl --data test.csv --target target_column --output evaluation --metrics accuracy,f1

# Explain model
sbyb eval explain --model model.pkl --data data.csv --output explanation --method shap
```

### Deployment Commands

```bash
# Export model
sbyb deploy export --model model.pkl --format onnx --output model.onnx

# Serve model
sbyb deploy serve --model model.pkl --host 0.0.0.0 --port 8000 --server fastapi
```

### UI Commands

```bash
# Generate UI
sbyb ui generate --model model.pkl --output ui --type dashboard --framework streamlit --theme light
```

### Plugin Commands

```bash
# List plugins
sbyb plugin list --category custom

# Install plugin
sbyb plugin install sbyb-plugin-example --force

# Uninstall plugin
sbyb plugin uninstall sbyb-plugin-example --category custom

# Create plugin template
sbyb plugin create my-plugin --category custom --output . --description "My custom plugin" --author "John Doe"
```

### Tracking Commands

```bash
# Create experiment
sbyb track experiment create --name "My Experiment" --description "Experiment description" --tags tag1,tag2

# List experiments
sbyb track experiment list

# Delete experiment
sbyb track experiment delete --id experiment_id

# Create run
sbyb track run create --experiment experiment_id --name "Run 1" --description "Run description" --tags tag1

# List runs
sbyb track run list --experiment experiment_id

# Delete run
sbyb track run delete --id run_id

# Visualize experiment
sbyb track visualize --experiment experiment_id --metrics accuracy,f1 --params learning_rate,max_depth --output visualization --format html
```

### Version Command

```bash
# Show version information
sbyb version
```

## Tutorials

### Basic Tutorial: Classification

This tutorial walks through a complete classification workflow using SBYB.

#### 1. Create a Project

```bash
sbyb project create --name classification_project --template classification
cd classification_project
```

#### 2. Prepare Your Data

Place your dataset in the `data` directory or use a sample dataset:

```python
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Save to CSV
data.to_csv("data/iris.csv", index=False)
```

#### 3. Explore Your Data

```bash
sbyb data profile --data data/iris.csv --output profile
```

Open `profile/data_profile.html` in your browser to view the data profile.

#### 4. Preprocess Your Data

```bash
sbyb data preprocess --data data/iris.csv --output data/preprocessed.csv
```

#### 5. Run AutoML

```bash
sbyb automl run --data data/preprocessed.csv --target target --output output --track
```

#### 6. Evaluate the Model

```bash
sbyb eval run --model output/model.pkl --data data/preprocessed.csv --target target --output evaluation
```

#### 7. Explain the Model

```bash
sbyb eval explain --model output/model.pkl --data data/preprocessed.csv --output explanation
```

#### 8. Generate a UI

```bash
sbyb ui generate --model output/model.pkl --output ui --type dashboard
```

Run the UI:

```bash
cd ui
streamlit run app.py
```

#### 9. Deploy the Model

```bash
sbyb deploy export --model output/model.pkl --format onnx --output model.onnx
sbyb deploy serve --model output/model.pkl
```

### Advanced Tutorial: Custom Components

This tutorial shows how to extend SBYB with custom components.

#### 1. Create a Plugin Template

```bash
sbyb plugin create my-preprocessor --category preprocessing
cd my-preprocessor
```

#### 2. Implement Your Custom Preprocessor

Edit `my_preprocessor.py`:

```python
from sbyb.preprocessing.base import BasePreprocessor
from sbyb.plugins.decorators import register_preprocessor

@register_preprocessor("my_custom_preprocessor")
class MyCustomPreprocessor(BasePreprocessor):
    def __init__(self, param1=1, param2="default"):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y=None):
        # Implement fitting logic
        return self
        
    def transform(self, X):
        # Implement transformation logic
        return X
```

#### 3. Install Your Plugin

```bash
pip install -e .
```

#### 4. Use Your Custom Preprocessor

```python
from sbyb.api import SBYB
from sbyb.preprocessing import PreprocessingPipeline

# Initialize SBYB
sbyb = SBYB()

# Create a preprocessing pipeline with your custom preprocessor
pipeline = PreprocessingPipeline()
pipeline.add_step("my_custom_preprocessor", param1=2, param2="custom")

# Use the pipeline
preprocessed_data = pipeline.fit_transform(data)
```

## Examples

### Classification Example

```python
from sbyb.api import SBYB
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data.csv")

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize SBYB
sbyb = SBYB()

# Preprocess data
preprocessed_train = sbyb.preprocess_data(train_data)
preprocessed_test = sbyb.preprocess_data(test_data)

# Run AutoML
result = sbyb.run_automl(
    data=preprocessed_train,
    target="target_column",
    task="classification",
    output_dir="output",
    track=True,
    experiment_name="Classification Example"
)

# Evaluate model
evaluation = sbyb.evaluate_model(
    model=result.model,
    data=preprocessed_test,
    target="target_column",
    output_dir="evaluation"
)

# Generate UI
sbyb.generate_ui(
    model=result.model,
    output_dir="ui",
    ui_type="dashboard",
    framework="streamlit"
)
```

### Regression Example

```python
from sbyb.api import SBYB
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data.csv")

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize SBYB
sbyb = SBYB()

# Preprocess data
preprocessed_train = sbyb.preprocess_data(train_data)
preprocessed_test = sbyb.preprocess_data(test_data)

# Run AutoML
result = sbyb.run_automl(
    data=preprocessed_train,
    target="target_column",
    task="regression",
    output_dir="output",
    track=True,
    experiment_name="Regression Example"
)

# Evaluate model
evaluation = sbyb.evaluate_model(
    model=result.model,
    data=preprocessed_test,
    target="target_column",
    output_dir="evaluation"
)

# Explain model
explanation = sbyb.explain_model(
    model=result.model,
    data=preprocessed_test,
    method="shap",
    output_dir="explanation"
)

# Generate UI
sbyb.generate_ui(
    model=result.model,
    output_dir="ui",
    ui_type="dashboard",
    framework="streamlit"
)
```

### Time Series Example

```python
from sbyb.api import SBYB
import pandas as pd

# Load data
data = pd.read_csv("time_series_data.csv")

# Initialize SBYB
sbyb = SBYB()

# Preprocess data
preprocessed_data = sbyb.preprocess_data(data)

# Run AutoML
result = sbyb.run_automl(
    data=preprocessed_data,
    target="target_column",
    task="time_series",
    output_dir="output",
    track=True,
    experiment_name="Time Series Example"
)

# Generate UI
sbyb.generate_ui(
    model=result.model,
    output_dir="ui",
    ui_type="dashboard",
    framework="streamlit"
)

# Deploy model
server = sbyb.serve_model(
    model=result.model,
    host="0.0.0.0",
    port=8000,
    server_type="fastapi"
)
server.start()
```

## Contributing

We welcome contributions to SBYB! Here's how you can contribute:

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/sbyb/sbyb.git
cd sbyb

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
black sbyb
flake8 sbyb
```

### Submitting a Pull Request

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-branch`
7. Submit a pull request

## License

SBYB is released under the MIT License. See the LICENSE file for details.
