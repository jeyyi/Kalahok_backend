# Backend of NLP part of Kalahok E participation 2.0
## Prerequisites
Before you get started, make sure you have the following installed:

Python 3.6 or higher
pip (Python package manager)
Git
### Step 1: Clone the repository
Clone the repository containing your FastAPI app to your local machine using the following command:


#### git clone <repository-url>
Replace <repository-url> with the URL of your repository. This will create a new directory with the name of your repository.

### Step 2: Create a virtual environment
Creating a virtual environment is good practice when working with Python projects, as it allows you to keep your dependencies separate from other projects on your machine.


#### python -m venv env
This command creates a new virtual environment named "env" in the current directory.

### Step 3: Activate the virtual environment
You can activate the virtual environment by running the following command:

#### source env/bin/activate
This will change your command prompt to indicate that you are now working inside the virtual environment.

### Step 4: Install dependencies
To install the dependencies for your FastAPI app, run the following command:


#### pip install -r requirements.txt

This will install all the packages listed in your requirements.txt file.

### Step 5: Start the app
To start the FastAPI app, navigate to the directory containing the app and run the following command:
 
#### uvicorn app.main:app --reload
This will start the development server and automatically reload the app whenever you make changes to the code.

You can now visit http://localhost:8000 in your web browser to see your app in action!
