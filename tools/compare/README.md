This is a collection of simple scripts to compare old to new databases (for states, matrix elements and wigner tables).
This can be helpful to see, if and what changed from a previous version to a newer version of the database files.

# Run the compare script
First copy a new and an old version of the databases inside this folder (e.g. "Rb_v1.0/" and "Rb_v1.1/")
and adapt the `new_path` and `old_path` variables inside the corresponding `compare_....py` script.
Now, you can run `uv run compare_....py`.

Optionally, you can adjust some of the arguments to the compare functions (e.g. the relative (rtol) and absolute (atol) tolerances) to your needs.
