# Latent Dirichlet Allocation
Final project for Advanced Machine Learning @ KTH.

## Using Poetry
* Install it - instructions here: https://python-poetry.org/docs/#installation
* Inside the repository, run `poetry install` to create a new virtual environment and install our dependencies in it
* Whenever you want to activate that virtual environment, run `poetry shell`
* When you want to deactive the virtual environment, just do `exit` or `ctrl+D`

### Adding dependencies
**Don't** `pip install` anything. Instead:
* If the dependency is needed for running our code, run `poetry add <dependency_name>`
* Otherwise, if the dependency is only for something related to our development process (eg. testing), run `poetry add --dev <dependency_name>`

After adding dependencies, make sure to commit the changes that `poetry` automatically made to `pyproject.toml` and `poetry.lock`. After the others pull your changes, they'll need to run `poetry update` to make sure they're in sync - but `poetry` will tell them automatically when they need to do it.
