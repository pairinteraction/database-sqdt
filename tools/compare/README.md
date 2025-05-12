This is a simple script to compare old to new databases.
This can be helpful to see, if and what changed from a previous version to a newer version of the database files.

# Run the compare script
First copy a new and a old version of the databases inside this folder (e.g. "Rb_v1.0/" and "Rb_v1.1/")
and adapt the `new_path` and `old_path` variables inside the `compare.py` script.
Then make sure you have the `generate_database_sqdt` package installed in your python environment.
Finally, you can run `python compare.py`.

Optionally, you can adjust the relative (rtol) and absolute (atol) tolerances, when comparing numerical values.
