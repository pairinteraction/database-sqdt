# Tool for generating the SQDT database tables in the cloud

Create the database including tables of states and matrix elements calculated with single-channel quantum defect theory.
Database tables are available through [GitHub Releases](https://github.com/pairinteraction/database-sqdt/releases).

## Generate new tables locally
Run the script to generate the tables via
```bash
uv run generate_database <species> --n-max 220
```

## Generate a new release
To generate a new release with all the important tables, simply create and push a new annotated tag with a tag name of the form "v*.*" .
This will run the `generate_database.yml` workflow, where first all tables are created (this happens for all commits, not only tags),
and then in addition uploads the zipped versions of the tables to a new release with name "v*.*" .
The release is created in draft mode, so you can double-check, that all database tables are included and optionally add a release text.
Once you are happy with the release draft, don't forget to publish the release.

## Misc

### Profiling / Benchmarking
To check the performance of this tool and to look for bottlenecks in the code, you can use [py-spy](https://github.com/benfred/py-spy) to profile the code.
To do so install py-spy from pip and then run the following command:

```bash
py-spy record -o profiling.svg -- python3 generate_database.py <species> --n-max <n-max>
```
